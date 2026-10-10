import argparse
import json
from pathlib import Path

import onnx
import torch

from eloftr import sha256, sidecar_path
from eloftr.evaluation import load_rgb, sample_pairs, to_tensor
from loma_trt.evaluation import aligned_compare, summarize_aligned, upstream_outputs
from loma_trt.static_model import GROUPS, PRECISIONS, StaticLoMa, half_groups, precision_label
from loma_trt.upstream import VARIANTS, WEIGHTS_DIR, checkpoint_path, load_upstream, upstream_commit

OUTPUT_NAMES = ['keypoints0', 'keypoints1', 'confidence', 'valid']


@torch.no_grad()
def self_check(upstream, height, width, num_keypoints, pairs):
    """StaticLoMa in fp32 must reproduce the upstream (all-fp32) matches."""
    model = StaticLoMa(upstream, height, width, num_keypoints).cuda().eval()
    results = []
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # TF32 rounding differs between fused and unfused conv+BN
    for path0, path1 in pairs:
        image0, image1 = (to_tensor(load_rgb(p, height, width)) for p in (path0, path1))
        ref = upstream_outputs(upstream, image0, image1, num_keypoints)
        results.append(aligned_compare(ref, model(image0, image1)))
    torch.backends.cudnn.allow_tf32 = tf32
    summary = summarize_aligned(results)
    print(f'Self-check vs upstream fp32 on {len(pairs)} pairs: {summary}')
    if (summary['aligned_min'] < 0.99 or summary['jaccard_min'] < 0.99 or summary['flow_median'] > 1e-2
            or summary['flow_gt1px'] > 1e-3):
        raise RuntimeError('StaticLoMa does not reproduce upstream LoMa, refusing to export')


def main():
    parser = argparse.ArgumentParser(description='Export a fixed-shape LoMa ONNX model.')
    parser.add_argument('--variant', choices=list(VARIANTS), default='b128',
                        help='b128: LoMa-B128 (DeDoDe-B descriptor); b: LoMa-B (DeDoDe-G descriptor with DINOv2).')
    parser.add_argument('--height', type=int, default=384, help='Input height, multiple of 8 (56 for LoMa-B).')
    parser.add_argument('--width', type=int, default=384, help='Input width, multiple of 8 (56 for LoMa-B).')
    parser.add_argument('--num-keypoints', type=int, default=2048, help='Keypoints per image (upstream default 2048).')
    parser.add_argument('--precision', choices=list(PRECISIONS), default='fp16',
                        help='fp16 (default, faster and closer to fp32) or bf16 (upstream mixed precision): '
                             'half-precision CNNs, DINOv2 and transformer; fp32 scoremap, softmaxes, sampling and '
                             'matching.')
    parser.add_argument('--half-groups', type=str, default=None,
                        help='Override the half-precision module groups, comma-separated subset of '
                             f'{",".join(GROUPS)}.')
    parser.add_argument('--out', type=Path, default=None, help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--skip-check', action='store_true', help='Skip the comparison against upstream.')
    opt = parser.parse_args()
    print(opt)

    groups = half_groups(opt.precision, opt.half_groups)
    label = precision_label(opt.precision, groups)
    suffix = '' if opt.num_keypoints == 2048 else f'_k{opt.num_keypoints}'
    out = opt.out or WEIGHTS_DIR / f'loma_{opt.variant}_{label}_{opt.height}x{opt.width}{suffix}.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)

    upstream = load_upstream(opt.variant, fp32=True)
    pairs = sample_pairs()
    if not opt.skip_check:
        self_check(upstream, opt.height, opt.width, opt.num_keypoints, pairs)

    model = StaticLoMa(upstream, opt.height, opt.width, opt.num_keypoints, opt.precision, opt.half_groups).cuda().eval()
    inputs = tuple(to_tensor(load_rgb(p, opt.height, opt.width)) for p in pairs[0])
    with torch.no_grad():
        torch.onnx.export(model, inputs, str(out), input_names=['image0', 'image1'], output_names=OUTPUT_NAMES,
                          opset_version=opt.opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(out))

    ckpt = checkpoint_path(opt.variant)
    metadata = {'model': 'loma', 'variant': opt.variant, 'height': opt.height, 'width': opt.width, 'channels': 3,
                'precision': label, 'dtype': opt.precision, 'half_groups': list(groups),
                'num_keypoints': opt.num_keypoints, 'thr': model.threshold, 'outputs': OUTPUT_NAMES,
                'pixel_coordinates': 'opencv', 'upstream_commit': upstream_commit(), 'torch': torch.__version__,
                'opset': opt.opset, 'checkpoint': str(ckpt), 'checkpoint_sha256': sha256(ckpt),
                'onnx_sha256': sha256(out)}
    sidecar_path(out).write_text(json.dumps(metadata, indent=2))
    print(f'Exported {out}')


if __name__ == '__main__':
    main()
