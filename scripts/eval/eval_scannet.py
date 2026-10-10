"""ScanNet-1500 relative pose benchmark: accuracy and speed of EfficientLoFTR, LoFTR, Coarse LoFTR, LoMa and RoMa v2,
in torch and TensorRT.

Follows the upstream EfficientLoFTR ScanNet test (third_party/EfficientLoFTR/scripts/reproduce_test/indoor_full_auc.sh):
the 1500 pairs are resized to 640x480, the essential matrix is estimated with OpenCV RANSAC (0.5 px, confidence
0.99999) on 5 shuffles of the matches, and accuracy is the AUC of the pose error max(rotation, translation angle) up to
5/10/20 degrees. Precision is the fraction of matches with a symmetric epipolar distance below 5e-4. Latency is measured
per pair with CUDA events, from the images on the GPU to the filtered matches.

Every TensorRT engine is compared with the torch model it was exported from, at the same input size and matching
settings (the engine's thr and border removal, upstream defaults 0.2 and 2). The upstream ScanNet results use other
settings for EfficientLoFTR (thr 0.1, no border removal, mixed precision) and the ScanNet-trained LoFTR weights;
--paper adds these two torch rows. LoMa and RoMa v2 engines are compared with upstream in its default (bf16) mixed
precision on the same RGB inputs; RoMa v2 matches are sampled with upstream's sampler from a fixed seed.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from eloftr.matcher import engine_metadata
from eloftr.pose import pose_auc, pose_error, symmetric_epipolar_distance
from eloftr.trt_runtime import TRTEngine
from eloftr.upstream import DEFAULT_CKPT, UPSTREAM_DIR, WEIGHTS_DIR as ELOFTR_WEIGHTS, load_upstream, run_upstream

ROOT = Path(__file__).resolve().parents[2]
ASSETS = UPSTREAM_DIR / 'assets' / 'scannet_test_1500'
LOFTR_WEIGHTS = ROOT / 'weights' / 'loftr'
LOMA_WEIGHTS = ROOT / 'weights' / 'loma'
ROMAV2_WEIGHTS = ROOT / 'weights' / 'romav2'
LEGACY_ENGINE = ROOT / 'weights' / 'LoFTR_teacher.engine'
LEGACY_WEIGHTS = ROOT / 'weights' / 'LoFTR_teacher.pt'
HEIGHT, WIDTH = 480, 640  # resolution of the ScanNet intrinsics; keypoints are mapped back to it
RANSAC_THR, RANSAC_CONF, EPI_THR = 0.5, 0.99999, 5e-4
LEGACY_CONF_THR = 0.01  # loftr.utils.helpers.get_coarse_match
# Acceptance targets for TensorRT engines against their torch model, in AUC percentage points and match count ratio.
# Only checked on the full 1500 pairs.
TARGETS = {'auc@5_delta': (-1.0, None), 'auc@10_delta': (-1.0, None), 'auc@20_delta': (-1.0, None),
           'count_ratio': (0.97, 1.03)}


def default_data_root():
    """$DATASET_DIR/scannet, with the default of run/env.sh."""
    return Path(os.environ.get('DATASET_DIR', Path.home() / 'dataset')) / 'scannet'


def load_pairs(data_root):
    """ScanNet-1500 pairs with intrinsics K (640x480) and the relative pose T_0to1 (upstream ScanNetDataset)."""
    names = np.load(ASSETS / 'test.npz')['name']
    intrinsics = dict(np.load(ASSETS / 'intrinsics.npz'))
    pairs = []
    for scene, sub, id0, id1 in names:
        scene = f'scene{scene:04d}_{sub:02d}'
        cam2world0, cam2world1 = (np.loadtxt(data_root / scene / 'pose' / f'{i}.txt') for i in (id0, id1))
        pairs.append({'image0': data_root / scene / 'color' / f'{id0}.jpg',
                      'image1': data_root / scene / 'color' / f'{id1}.jpg',
                      'K': intrinsics[scene].reshape(3, 3),
                      'T_0to1': np.linalg.inv(cam2world1) @ cam2world0})
    return pairs


class PairImages(Dataset):
    """uint8 images of a pair resized to the method input size: grayscale like upstream read_scannet_gray, or RGB."""

    def __init__(self, pairs, height, width, color=False):
        self.pairs, self.size, self.color = pairs, (width, height), color

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        images = []
        for key in ('image0', 'image1'):
            image = cv2.imread(str(self.pairs[index][key]), cv2.IMREAD_COLOR if self.color else cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(self.pairs[index][key])
            image = cv2.resize(image, self.size)
            images.append(torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.color else image))
        return images


class Method:
    """A matcher at a fixed input size. `load()` returns match(image0, image1) for [1, C, H, W] float CUDA images in
    [0, 1] (grayscale, or RGB if `color`) -> (kpts0, kpts1 [M, 2] in input pixels, conf [M]); TensorRT methods name
    their torch `reference`. Keypoints of `pixel_center` 0.5 methods are OpenCV pixels (centres at integers), rescaled
    exactly to the intrinsics frame; the others are scaled about the image origin as upstream EfficientLoFTR does."""

    def __init__(self, name, height, width, load, reference=None, color=False, pixel_center=0.0):
        self.name, self.height, self.width, self.load, self.reference = name, height, width, load, reference
        self.color, self.pixel_center = color, pixel_center


def size_suffix(height, width):
    return '' if (height, width) == (HEIGHT, WIDTH) else f' {width}x{height}'


def eloftr_torch(model_type, mp, height, width, thr=None, border_rm=None, npe=None, ckpt=DEFAULT_CKPT):
    settings = '' if thr is None and border_rm is None else f' (thr {thr}, border {border_rm})'
    name = f'eloftr {model_type} torch {"mp" if mp else "fp32"}{size_suffix(height, width)}{settings}'

    def load():
        model = load_upstream(ckpt, model_type, npe, mp=mp, thr=thr, border_rm=border_rm)

        def match(image0, image1):
            batch = run_upstream(model, image0, image1, mp=mp)
            return batch['mkpts0_f'], batch['mkpts1_f'], batch['mconf']
        return match
    return Method(name, height, width, load)


def loftr_torch(pretrained, height, width):
    def load():
        from loftr_full.upstream import load_upstream as load_loftr, run_upstream as run_loftr
        model = load_loftr(pretrained)

        def match(image0, image1):
            out = run_loftr(model, image0, image1)
            return out['keypoints0'], out['keypoints1'], out['confidence']
        return match
    return Method(f'loftr {pretrained} torch fp32{size_suffix(height, width)}', height, width, load)


def static_engine(path, cuda_graph):
    """EfficientLoFTR or LoFTR engine: one candidate per coarse cell of image0 and a `valid` mask."""
    meta = engine_metadata(path)
    height, width = meta['height'], meta['width']
    variant = meta['model_type'] if meta['model'] == 'eloftr' else meta['pretrained']
    name = f'{meta["model"]} {variant} trt {meta["precision"]}{size_suffix(height, width)}'

    def load():
        engine = TRTEngine(path)
        if cuda_graph:
            engine.capture_cuda_graph()

        def match(image0, image1):
            out = engine(image0, image1)
            keypoints0, keypoints1, confidence, valid = (out[k] for k in meta['outputs'])
            return keypoints0[valid], keypoints1[valid], confidence[valid]
        return match

    if meta['model'] == 'eloftr':
        reference = eloftr_torch(meta['model_type'], False, height, width, npe=meta.get('npe'))
    else:
        reference = loftr_torch(meta['pretrained'], height, width)
    return Method(name, height, width, load, reference.name), reference


def loma_torch(variant, height, width, num_keypoints):
    """Upstream LoMa in its default mixed precision (bf16 autocast) on the RGB images, as its tensor API matches."""
    def load():
        from loma_trt.upstream import load_upstream, run_upstream, to_pixels
        model = load_upstream(variant)

        def match(image0, image1):
            out = run_upstream(model, image0, image1, num_keypoints)
            valid = out['matches0'] > -1
            kpts0 = to_pixels(out['keypoints0'][valid], height, width)
            kpts1 = to_pixels(out['keypoints1'][out['matches0'][valid]], height, width)
            return kpts0, kpts1, out['scores0'][valid]
        return match
    name = f'loma {variant} torch mp{size_suffix(height, width)}'
    return Method(name, height, width, load, color=True, pixel_center=0.5)


def loma_engine(path, cuda_graph):
    meta = engine_metadata(path)
    height, width = meta['height'], meta['width']

    def load():
        engine = TRTEngine(path)
        if cuda_graph:
            engine.capture_cuda_graph()

        def match(image0, image1):
            out = engine(image0, image1)
            keypoints0, keypoints1, confidence, valid = (out[k] for k in meta['outputs'])
            return keypoints0[valid], keypoints1[valid], confidence[valid]
        return match
    reference = loma_torch(meta['variant'], height, width, meta['num_keypoints'])
    name = f'loma {meta["variant"]} trt {meta["precision"]}{size_suffix(height, width)}'
    return Method(name, height, width, load, reference.name, color=True, pixel_center=0.5), reference


def romav2_torch(height, width, bidirectional, num_matches):
    """Upstream RoMa v2 forward in its default mixed precision (bf16), matches sampled with upstream's sampler."""
    def load():
        from romav2_trt.evaluation import upstream_outputs
        from romav2_trt.matcher import sample_matches
        from romav2_trt.upstream import load_upstream, run_upstream
        model = load_upstream(bidirectional=bidirectional)

        def match(image0, image1):
            return sample_matches(upstream_outputs(run_upstream(model, image0, image1)), num_matches)
        return match
    name = f'romav2{" bidir" if bidirectional else ""} torch mp{size_suffix(height, width)}'
    return Method(name, height, width, load, color=True, pixel_center=0.5)


def romav2_engine(path, cuda_graph, num_matches):
    meta = engine_metadata(path)
    height, width = meta['height'], meta['width']

    def load():
        from romav2_trt.matcher import sample_matches
        engine = TRTEngine(path)
        if cuda_graph:
            engine.capture_cuda_graph()

        def match(image0, image1):
            out = engine(image0, image1)
            return sample_matches(tuple(out[k] for k in meta['outputs']), num_matches)
        return match
    reference = romav2_torch(height, width, meta['bidirectional'], num_matches)
    name = f'romav2{" bidir" if meta["bidirectional"] else ""} trt {meta["precision"]}{size_suffix(height, width)}'
    return Method(name, height, width, load, reference.name, color=True, pixel_center=0.5), reference


def coarse_matches(conf_matrix, width, stride):
    """loftr.utils.helpers.get_coarse_match on the GPU: every entry of the [1, L, S] confidence above 0.01."""
    _, i, j = torch.nonzero(conf_matrix > LEGACY_CONF_THR, as_tuple=True)
    wc = width // stride
    kpts0 = torch.stack([i % wc, i // wc], 1).float() * stride
    kpts1 = torch.stack([j % wc, j // wc], 1).float() * stride
    return kpts0, kpts1, conf_matrix[0, i, j]


def legacy_config():
    from loftr.utils.cvpr_ds_config import default_cfg
    from loftr.utils.helpers import make_student_config
    return make_student_config(deepcopy(default_cfg))  # make_student_config edits the nested dicts in place


def legacy_torch(weights=LEGACY_WEIGHTS):
    config = legacy_config()
    height, width, stride = config['input_height'], config['input_width'], config['resolution'][0]

    def load():
        from loftr import LoFTR
        model = LoFTR(config)
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False)['model_state_dict'])
        model = model.eval().cuda()

        def match(image0, image1):
            conf_matrix, _ = model(image0, image1)
            return coarse_matches(conf_matrix, width, stride)
        return match
    return Method('coarse-loftr student torch fp32', height, width, load)


def legacy_engine(path, cuda_graph):
    config = legacy_config()
    height, width, stride = config['input_height'], config['input_width'], config['resolution'][0]

    def load():
        engine = TRTEngine(path)
        if cuda_graph:
            engine.capture_cuda_graph()

        def match(image0, image1):
            conf_matrix = next(iter(engine(image0, image1).values()))  # outputs: conf_matrix, similarity matrix
            return coarse_matches(conf_matrix, width, stride)
        return match
    reference = legacy_torch()
    return Method('coarse-loftr student trt fp16', height, width, load, reference.name), reference


def make_methods(opt):
    """TensorRT engines, each preceded by its torch model (EfficientLoFTR also in upstream mixed precision)."""
    engines = opt.engine if opt.engine is not None else (
        sorted(ELOFTR_WEIGHTS.glob('*.engine')) + sorted(LOFTR_WEIGHTS.glob('*.engine'))
        + sorted(LOMA_WEIGHTS.glob('*.engine')) + sorted(ROMAV2_WEIGHTS.glob('*.engine'))
        + ([LEGACY_ENGINE] if LEGACY_ENGINE.exists() else []))
    methods = {}
    for path in engines:
        meta = engine_metadata(path)
        if meta.get('model') in ('eloftr', 'loftr'):
            method, reference = static_engine(path, opt.cuda_graph)
        elif meta.get('model') == 'loma':
            method, reference = loma_engine(path, opt.cuda_graph)
        elif meta.get('model') == 'romav2':
            method, reference = romav2_engine(path, opt.cuda_graph, opt.romav2_matches)
        else:
            method, reference = legacy_engine(path, opt.cuda_graph)
        if not opt.no_torch:
            methods.setdefault(reference.name, reference)
            if meta.get('model') == 'eloftr':
                mp = eloftr_torch(meta['model_type'], True, method.height, method.width, npe=meta.get('npe'))
                methods.setdefault(mp.name, mp)
        methods[method.name] = method
    if opt.paper:
        for method in (eloftr_torch('full', True, HEIGHT, WIDTH, thr=0.1, border_rm=0),
                       loftr_torch('indoor_new', HEIGHT, WIDTH)):
            methods.setdefault(method.name, method)
    return list(methods.values())


@torch.no_grad()
def run_matching(method, pairs, opt):
    """Matches of every pair in the 640x480 frame of the intrinsics, and the latency of each pair in ms."""
    match = method.load()
    loader = DataLoader(PairImages(pairs, method.height, method.width, method.color), batch_size=None,
                        num_workers=opt.workers, pin_memory=True)
    scale = np.array([WIDTH / method.width, HEIGHT / method.height], dtype=np.float32)
    center = method.pixel_center
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    matches, times = [], []
    for index, images in enumerate(loader):
        images = (x.cuda(non_blocking=True) for x in images)
        layout = (lambda x: x.permute(2, 0, 1)[None]) if method.color else (lambda x: x[None, None])
        image0, image1 = (layout(x).float() / 255. for x in images)
        if index == 0:
            for _ in range(opt.warmup):
                match(image0, image1)
        start.record()
        kpts0, kpts1, _ = match(image0, image1)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        matches.append(tuple((k.float().cpu().numpy() + center) * scale - center for k in (kpts0, kpts1)))
    return matches, np.array(times)


def pair_metrics(index, kpts0, kpts1, pair, ransac_times):
    """Pose errors of `ransac_times` RANSAC runs on shuffled matches (seeded by the pair index) and the precision."""
    K, T_0to1 = pair['K'], pair['T_0to1']
    rng = np.random.default_rng(index)
    errors = []
    for _ in range(ransac_times):
        order = rng.permutation(len(kpts0))
        errors.append(pose_error(kpts0[order], kpts1[order], K, K, T_0to1, RANSAC_THR, RANSAC_CONF))
    distances = symmetric_epipolar_distance(kpts0, kpts1, T_0to1, K, K)
    return errors, float((distances < EPI_THR).mean()) if len(distances) else 0.0


def summarize(matches, times, pairs, opt):
    with ThreadPoolExecutor(opt.threads) as pool:  # OpenCV releases the GIL
        results = list(pool.map(lambda i: pair_metrics(i, *matches[i], pairs[i], opt.ransac_times), range(len(pairs))))
    errors = np.array([e for r in results for e in r[0]])
    row = {f'auc@{t}': 100 * auc for t, auc in pose_auc(errors).items()}
    row['precision'] = 100 * float(np.mean([r[1] for r in results]))
    row['matches_mean'] = float(np.mean([len(m[0]) for m in matches]))
    row['latency_ms_median'] = float(np.median(times))
    row['latency_ms_mean'] = float(np.mean(times))
    row['pose_failures'] = int(np.isinf(errors).sum())
    return row


def compare_to_references(methods, rows):
    for method in methods:
        ref = rows.get(method.reference)
        if ref is None:
            continue
        row = rows[method.name]
        for t in (5, 10, 20):
            row[f'auc@{t}_delta'] = row[f'auc@{t}'] - ref[f'auc@{t}']
        row['count_ratio'] = row['matches_mean'] / ref['matches_mean'] if ref['matches_mean'] else float('nan')
        row['reference'] = method.reference


def check(rows):
    failures = []
    for name, row in rows.items():
        if 'reference' not in row:
            continue
        for key, (low, high) in TARGETS.items():
            if (low is not None and row[key] < low) or (high is not None and row[key] > high):
                failures.append(f'{name}: {key}={row[key]:.3f} outside [{low}, {high}] vs {row["reference"]}')
    return failures


def print_table(title, rows):
    columns = [('matches_mean', 'matches', '.0f'), ('auc@5', 'AUC@5', '.2f'), ('auc@10', 'AUC@10', '.2f'),
               ('auc@20', 'AUC@20', '.2f'), ('precision', 'P@5e-4', '.2f'), ('latency_ms_median', 'ms', '.2f'),
               ('latency_ms_mean', 'ms mean', '.2f'), ('auc@5_delta', 'dAUC@5', '+.2f'),
               ('auc@10_delta', 'dAUC@10', '+.2f'), ('auc@20_delta', 'dAUC@20', '+.2f')]
    width = max(len(n) for n in rows) + 2
    print(f'\n{title}\nAUC of the pose error at 5/10/20 deg and precision in %, latency per pair (median, mean) from '
          f'images on the GPU to filtered matches;\ndAUC: TensorRT minus its torch model at the same settings.')
    print(' ' * width + ''.join(f'{label:>9}' for _, label, _ in columns))
    for name, row in rows.items():
        cells = ''.join((format(row[k], fmt) if k in row else '-').rjust(9) for k, _, fmt in columns)
        print(f'{name:<{width}}{cells}')


def main():
    parser = argparse.ArgumentParser(description='ScanNet-1500 relative pose accuracy and speed of torch models and '
                                                 'TensorRT engines.')
    parser.add_argument('--engine', type=Path, nargs='*', default=None,
                        help='Engines to evaluate, default: all in weights/eloftr, weights/loftr, weights/loma and '
                             'weights/romav2, and the legacy weights/LoFTR_teacher.engine.')
    parser.add_argument('--romav2-matches', type=int, default=5000,
                        help='Matches sampled from the RoMa v2 warps (upstream ScanNet protocol: 5000).')
    parser.add_argument('--no-torch', action='store_true', help='Only evaluate the engines.')
    parser.add_argument('--paper', action='store_true',
                        help='Add the published ScanNet settings: EfficientLoFTR full in mixed precision with thr 0.1 '
                             'and no border removal, and LoFTR with the ScanNet weights (indoor_new).')
    parser.add_argument('--data-root', type=Path, default=default_data_root(),
                        help='ScanNet-1500 test images and poses, default: $DATASET_DIR/scannet.')
    parser.add_argument('--limit', type=int, default=None, help='Evaluate N evenly spaced pairs (quick check).')
    parser.add_argument('--ransac-times', type=int, default=5, help='RANSAC runs per pair on shuffled matches.')
    parser.add_argument('--warmup', type=int, default=20, help='Warm-up iterations of each method on the first pair.')
    parser.add_argument('--cuda-graph', action='store_true', help='Run the TensorRT engines as CUDA graphs.')
    parser.add_argument('--workers', type=int, default=4, help='Image loading processes.')
    parser.add_argument('--threads', type=int, default=min(16, os.cpu_count()), help='Pose estimation threads.')
    parser.add_argument('--out', type=Path, default=Path('outputs/eval'))
    opt = parser.parse_args()

    if not (opt.data_root / 'scene0707_00').is_dir():
        sys.exit(f'ScanNet-1500 not found in {opt.data_root}, run: pixi run download-scannet')
    pairs = load_pairs(opt.data_root)
    if opt.limit:
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, min(opt.limit, len(pairs))).round().astype(int)]
    methods = make_methods(opt)
    if not methods:
        sys.exit('No engines found, run: pixi run build-default')
    torch.backends.cudnn.benchmark = True  # fixed input sizes; upstream times its models the same way

    rows = {}
    for method in methods:
        print(f'{method.name}: matching {len(pairs)} pairs', flush=True)
        matches, times = run_matching(method, pairs, opt)
        rows[method.name] = summarize(matches, times, pairs, opt)
        torch.cuda.empty_cache()
    compare_to_references(methods, rows)

    opt.out.mkdir(parents=True, exist_ok=True)
    report = opt.out / (f'scannet_{len(pairs)}.json' if opt.limit else 'scannet.json')
    protocol = {'pairs': len(pairs), 'ransac_times': opt.ransac_times, 'ransac_thr_px': RANSAC_THR,
                'ransac_conf': RANSAC_CONF, 'epipolar_thr': EPI_THR, 'cuda_graph': opt.cuda_graph,
                'gpu': torch.cuda.get_device_name()}
    report.write_text(json.dumps({'protocol': protocol, 'rows': rows}, indent=2))
    print_table(f'ScanNet-1500, {len(pairs)} pairs at {WIDTH}x{HEIGHT} on {torch.cuda.get_device_name()}', rows)
    print(f'Report: {report}')

    if opt.limit:
        print('\nAcceptance targets are only checked on all 1500 pairs.')
        return
    failures = check(rows)
    if failures:
        print('\nFAILED acceptance targets:\n  ' + '\n  '.join(failures))
        sys.exit(1)
    print('\nAll TensorRT engines meet the acceptance targets.')


if __name__ == '__main__':
    main()
