import argparse
import json
from pathlib import Path

import onnx
import torch

from eloftr import sha256, sidecar_path
from eloftr.evaluation import compare, load_gray, sample_pairs, static_matches, summarize, to_tensor, upstream_matches
from eloftr.static_model import StaticELoFTR, precision_groups, precision_label
from eloftr.upstream import DEFAULT_CKPT, WEIGHTS_DIR, load_upstream, run_upstream, upstream_commit

OUTPUT_NAMES = ['keypoints0', 'keypoints1', 'confidence', 'valid']


@torch.no_grad()
def self_check(upstream, height, width, pairs):
    """StaticELoFTR in fp32 must reproduce upstream (all-fp32 reference) matches."""
    model = StaticELoFTR(upstream, height, width).cuda().eval()
    results = []
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # TF32 rounding differs between fused and unfused conv+BN
    for path0, path1 in pairs:
        image0, image1 = (to_tensor(load_gray(p, height, width)) for p in (path0, path1))
        ref = upstream_matches(run_upstream(upstream, image0, image1, strict_fp32=True))
        results.append(compare(ref, static_matches(*model(image0, image1))))
    torch.backends.cudnn.allow_tf32 = tf32
    summary = summarize(results)
    print(f'Self-check vs upstream fp32 on {len(pairs)} pairs: {summary}')
    if summary['jaccard_min'] < 0.99 or summary['flow_median'] > 1e-2 or summary['flow_gt1px'] > 1e-3:
        raise RuntimeError('StaticELoFTR does not reproduce upstream EfficientLoFTR, refusing to export')


def main():
    parser = argparse.ArgumentParser(description='Export a fixed-shape EfficientLoFTR ONNX model.')
    parser.add_argument('--height', type=int, default=480, help='Input height, multiple of 32.')
    parser.add_argument('--width', type=int, default=640, help='Input width, multiple of 32.')
    parser.add_argument('--model-type', choices=['full', 'opt'], default='full',
                        help='full: dual-softmax coarse matching (accuracy); opt: raw similarity (speed).')
    parser.add_argument('--precision', choices=['fp32', 'mixed'], default='mixed',
                        help='mixed: fp16 backbone, transformer and fine FPN; fp32 matching, LayerNorms and softmaxes.')
    parser.add_argument('--fp16', type=str, default=None,
                        help='Override the fp16 module groups, comma-separated subset of '
                             'backbone,coarse,fine,fine_matching.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT, help='Path to eloftr_outdoor.ckpt.')
    parser.add_argument('--npe', type=int, nargs=4, default=None,
                        help='RoPE train/test sizes; upstream suggests [832, 832, long_side, long_side] above 832.')
    parser.add_argument('--out', type=Path, default=None, help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--skip-check', action='store_true', help='Skip the comparison against upstream.')
    opt = parser.parse_args()
    print(opt)

    groups = precision_groups(opt.precision, opt.fp16)
    label = precision_label(groups)
    out = opt.out or WEIGHTS_DIR / f'eloftr_{opt.model_type}_{label}_{opt.height}x{opt.width}.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)

    upstream = load_upstream(opt.ckpt, opt.model_type, opt.npe)
    pairs = sample_pairs()
    if not opt.skip_check:
        self_check(upstream, opt.height, opt.width, pairs)

    model = StaticELoFTR(upstream, opt.height, opt.width, groups).cuda().eval()
    inputs = tuple(to_tensor(load_gray(p, opt.height, opt.width)) for p in pairs[0])
    with torch.no_grad():
        torch.onnx.export(model, inputs, str(out), input_names=['image0', 'image1'], output_names=OUTPUT_NAMES,
                          opset_version=opt.opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(out))

    metadata = {'model': 'eloftr', 'height': opt.height, 'width': opt.width, 'model_type': opt.model_type,
                'precision': label, 'fp16_groups': list(groups), 'thr': model.thr,
                'npe': upstream.config['coarse']['npe'],
                'outputs': OUTPUT_NAMES, 'upstream_commit': upstream_commit(), 'torch': torch.__version__,
                'opset': opt.opset, 'checkpoint': str(opt.ckpt), 'checkpoint_sha256': sha256(opt.ckpt),
                'onnx_sha256': sha256(out)}
    sidecar_path(out).write_text(json.dumps(metadata, indent=2))
    print(f'Exported {out}')


if __name__ == '__main__':
    main()
