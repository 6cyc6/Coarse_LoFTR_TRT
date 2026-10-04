import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from eloftr.evaluation import compare, load_gray, sample_pairs, static_matches, summarize, to_tensor, upstream_matches
from eloftr.matcher import engine_metadata
from eloftr.static_model import PRECISIONS, StaticELoFTR
from eloftr.trt_runtime import TRTEngine
from eloftr.upstream import DEFAULT_CKPT, WEIGHTS_DIR, load_upstream, run_upstream

# Reference: the upstream torch model computed entirely in fp32. Upstream's default inference runs the
# fine-level einsums under fp16 autocast, which already changes the fine argmax for ~11% of the matches
# (equally valid pixel pairs); that row and upstream mixed precision calibrate the fine-level noise.
REFERENCE = 'torch upstream fp32 strict'
NOISE = ('torch upstream fp32', 'torch upstream mp')
# Acceptance targets for TensorRT engines against the torch reference. Ground-truth accuracy on the synthetic
# homographies must match; fine-level disagreement may not exceed upstream's own precision noise by much.
TARGETS = {'count_ratio': (0.97, 1.03), 'jaccard_mean': (0.97, None), 'homography_p1_delta': (-0.005, None),
           'homography_p3_delta': (-0.005, None), 'homography_err_delta': (None, 0.02),
           'flow_gt1px_excess': (None, 0.03)}


def random_homography(rng, height, width, perturb=0.15):
    """Random perspective warp: move the image corners by up to `perturb` of the shorter side."""
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    moved = corners + rng.uniform(-perturb, perturb, corners.shape).astype(np.float32) * min(height, width)
    return cv2.getPerspectiveTransform(corners, moved)


def fundamental_inliers(matches):
    if len(matches['i']) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(matches['kpts0'], matches['kpts1'], cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    return int(mask.sum()) if mask is not None else 0


def homography_errors(matches, homography):
    """Distance between kpts1 and the ground-truth projection of kpts0."""
    if len(matches['i']) == 0:
        return np.zeros(0)
    projected = cv2.perspectiveTransform(matches['kpts0'][None].astype(np.float64), homography)[0]
    return np.linalg.norm(projected - matches['kpts1'], axis=1)


@torch.no_grad()
def latency_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def filtered(keypoints0, keypoints1, confidence, valid):
    return keypoints0[valid], keypoints1[valid], confidence[valid]


def engine_outputs(engine, names, image0, image1):
    out = engine(image0, image1)
    return [out[k] for k in names]


def make_methods(height, width, model_type, engines, ckpt):
    """name -> (matches(image0, image1) -> dict, run(image0, image1) for timing, optional run with CUDA graph)."""
    upstream = load_upstream(ckpt, model_type)
    upstream_mp = load_upstream(ckpt, model_type, mp=True)
    methods = {
        'torch upstream fp32 strict': (lambda a, b: upstream_matches(run_upstream(upstream, a, b, strict_fp32=True)),
                                       lambda a, b: run_upstream(upstream, a, b, strict_fp32=True)),
        'torch upstream fp32': (lambda a, b: upstream_matches(run_upstream(upstream, a, b)),
                                lambda a, b: run_upstream(upstream, a, b)),
        'torch upstream mp': (lambda a, b: upstream_matches(run_upstream(upstream_mp, a, b, mp=True)),
                              lambda a, b: run_upstream(upstream_mp, a, b, mp=True)),
    }
    for precision, groups in PRECISIONS.items():
        model = StaticELoFTR(upstream, height, width, groups).cuda().eval()
        methods[f'torch static {precision}'] = (lambda a, b, m=model: static_matches(*m(a, b)),
                                                lambda a, b, m=model: filtered(*m(a, b)))
    for path in engines:
        meta = engine_metadata(path)
        engine = TRTEngine(path)
        names = meta['outputs']
        methods[f'trt {meta["precision"]} ({path.name})'] = (
            lambda a, b, e=engine, n=names: static_matches(*engine_outputs(e, n, a, b)),
            lambda a, b, e=engine, n=names: filtered(*engine_outputs(e, n, a, b)),
            engine)
    return methods


@torch.no_grad()
def evaluate(height, width, model_type, engines, pairs, opt):
    methods = make_methods(height, width, model_type, engines, opt.ckpt)
    images = {p: to_tensor(load_gray(p, height, width)) for pair in pairs for p in pair}
    rng = np.random.default_rng(0)
    homographies = {p: random_homography(rng, height, width) for p in images}
    warped = {p: to_tensor(cv2.warpPerspective(load_gray(p, height, width), h, (width, height)))
              for p, h in homographies.items()}

    outputs = {name: {'pairs': [], 'homography': []} for name in methods}
    for name, (matches, *_) in methods.items():
        for p0, p1 in pairs:
            outputs[name]['pairs'].append(matches(images[p0], images[p1]))
        for p in images:
            outputs[name]['homography'].append(matches(images[p], warped[p]))

    ref = outputs[REFERENCE]
    rows = {}
    for name, (_, run, *engine) in methods.items():
        out = outputs[name]
        row = summarize([compare(r, o) for r, o in zip(ref['pairs'], out['pairs'])])
        row['matches_mean'] = float(np.mean([len(m['i']) for m in out['pairs']]))
        row['f_inliers_mean'] = float(np.mean([fundamental_inliers(m) for m in out['pairs']]))
        errors = [homography_errors(m, homographies[p]) for m, p in zip(out['homography'], images)]
        for t in (1, 3, 5):
            row[f'homography_p{t}'] = float(np.mean([(e < t).mean() if len(e) else 0.0 for e in errors]))
        # mean error of the matches within 5 px, i.e. the sub-pixel accuracy of correct matches
        errors = np.concatenate(errors)
        row['homography_err'] = float(errors[errors < 5].mean())
        image0, image1 = images[pairs[0][0]], images[pairs[0][1]]
        row['latency_ms'] = latency_ms(lambda: run(image0, image1), opt.warmup, opt.iters)
        if engine and opt.cuda_graph:
            engine[0].capture_cuda_graph()
            row['latency_cuda_graph_ms'] = latency_ms(lambda: run(image0, image1), opt.warmup, opt.iters)
        rows[name] = row
    noise = max(rows[name]['flow_gt1px'] for name in NOISE)
    for row in rows.values():
        for key in ('homography_p1', 'homography_p3', 'homography_err'):
            row[f'{key}_delta'] = row[key] - rows[REFERENCE][key]
        row['flow_gt1px_excess'] = row['flow_gt1px'] - noise
    return rows


def check(rows):
    failures = []
    for name, row in rows.items():
        if not name.startswith('trt '):
            continue
        for key, (low, high) in TARGETS.items():
            if (low is not None and row[key] < low) or (high is not None and row[key] > high):
                failures.append(f'{name}: {key}={row[key]:.4f} outside [{low}, {high}]')
    return failures


def print_table(title, rows):
    columns = [('matches_mean', 'matches', '.0f'), ('count_ratio', 'count', '.3f'), ('jaccard_mean', 'jacc', '.3f'),
               ('jaccard_min', 'jacc min', '.3f'), ('flow_p95', 'flow p95', '.2f'), ('flow_gt1px', 'flow>1px', '.3f'),
               ('f_inliers_mean', 'F inl', '.0f'), ('homography_p1', 'H@1', '.4f'), ('homography_p3', 'H@3', '.4f'),
               ('homography_err', 'H err', '.3f'), ('latency_ms', 'ms', '.2f'), ('latency_cuda_graph_ms', 'ms graph', '.2f')]
    width = max(len(n) for n in rows) + 2
    print(f'\n{title}\n'
          f'Agreement with {REFERENCE}: count ratio, Jaccard of coarse matches, displacement difference (px) of\n'
          f'shared matches. Ground truth on synthetic homographies: precision at 1/3 px, mean error (px) below 5 px.')
    print(' ' * width + ''.join(f'{label:>10}' for _, label, _ in columns))
    for name, row in rows.items():
        cells = ''.join(f'{row[k]:>10{fmt}}' if k in row else f'{"-":>10}' for k, _, fmt in columns)
        print(f'{name:<{width}}{cells}')


def main():
    parser = argparse.ArgumentParser(description='Compare EfficientLoFTR TensorRT engines with the torch model.')
    parser.add_argument('--engine', type=Path, nargs='*', default=None,
                        help='Engines to evaluate, default: all engines in weights/eloftr.')
    parser.add_argument('--pair', type=Path, nargs=2, action='append', default=None,
                        help='Image pair to use instead of the upstream sample pairs (repeatable).')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--no-cuda-graph', dest='cuda_graph', action='store_false')
    parser.add_argument('--out', type=Path, default=Path('outputs/eval'))
    opt = parser.parse_args()

    engines = opt.engine if opt.engine else sorted(WEIGHTS_DIR.glob('*.engine'))
    if not engines:
        sys.exit('No engines found, run: pixi run build-default')
    groups = {}
    for path in engines:
        meta = engine_metadata(path)
        groups.setdefault((meta['height'], meta['width'], meta['model_type']), []).append(path)

    pairs = [tuple(p) for p in opt.pair] if opt.pair else sample_pairs()
    opt.out.mkdir(parents=True, exist_ok=True)
    failures = []
    for (height, width, model_type), paths in groups.items():
        rows = evaluate(height, width, model_type, paths, pairs, opt)
        title = f'EfficientLoFTR {model_type} {height}x{width}, {len(pairs)} pairs on {torch.cuda.get_device_name()}'
        print_table(title, rows)
        report = opt.out / f'eloftr_{model_type}_{height}x{width}.json'
        report.write_text(json.dumps({'pairs': [[str(a), str(b)] for a, b in pairs], 'rows': rows}, indent=2))
        print(f'Report: {report}')
        failures += check(rows)

    if failures:
        print('\nFAILED acceptance targets:\n  ' + '\n  '.join(failures))
        sys.exit(1)
    print('\nAll TensorRT engines meet the acceptance targets.')


if __name__ == '__main__':
    main()
