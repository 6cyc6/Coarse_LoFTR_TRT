import argparse
import json
from pathlib import Path

import onnx
import torch

from eloftr import sha256, sidecar_path
from eloftr.evaluation import load_rgb, sample_pairs, to_tensor
from loma_trt.static_model import PRECISIONS, half_groups, precision_label
from romav2_trt.evaluation import dense_compare, summarize_dense, upstream_outputs
from romav2_trt.static_model import GROUPS, StaticRoMaV2, group_dtypes
from romav2_trt.upstream import WEIGHTS_DIR, checkpoint_path, load_upstream, run_upstream, upstream_commit


@torch.no_grad()
def self_check(upstream, height, width, bidirectional, pairs):
    """StaticRoMaV2 in fp32 must reproduce the upstream (all-fp32) warps and confidences."""
    model = StaticRoMaV2(upstream, height, width, bidirectional=bidirectional, batch_images=False).cuda().eval()
    results = []
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # TF32 rounding differs between fused and unfused conv+BN
    for path0, path1 in pairs:
        image0, image1 = (to_tensor(load_rgb(p, height, width)) for p in (path0, path1))
        ref = upstream_outputs(run_upstream(upstream, image0, image1))
        results.append(dense_compare(ref, model(image0, image1)))
    torch.backends.cudnn.allow_tf32 = tf32
    summary = summarize_dense(results)
    print(f'Self-check vs upstream fp32 on {len(pairs)} pairs: {summary}')
    if summary['epe_median'] > 1e-3 or summary['epe_gt1px'] > 1e-3 or summary['overlap_agree'] < 0.999:
        raise RuntimeError('StaticRoMaV2 does not reproduce upstream RoMa v2, refusing to export')


def main():
    parser = argparse.ArgumentParser(description='Export a fixed-shape RoMa v2 ONNX model.')
    parser.add_argument('--height', type=int, default=384, help='Input height, multiple of 16.')
    parser.add_argument('--width', type=int, default=384, help='Input width, multiple of 16.')
    parser.add_argument('--bidirectional', action='store_true', help='Also predict the warp of image1 into image0.')
    parser.add_argument('--precision', choices=list(PRECISIONS), default='fp16',
                        help='fp16 (default; DINOv3 stays bf16, its activations exceed the fp16 range) or bf16 '
                             '(upstream mixed precision): half-precision DINOv3, matcher, DPT head and refiner '
                             'convolutions; fp32 matching, sampling, local correlation and warp arithmetic.')
    parser.add_argument('--half-groups', type=str, default=None,
                        help='Override the half-precision module groups, comma-separated subset of '
                             f'{",".join(GROUPS)}.')
    parser.add_argument('--out', type=Path, default=None, help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--skip-check', action='store_true', help='Skip the comparison against upstream.')
    opt = parser.parse_args()
    print(opt)

    groups = half_groups(opt.precision, opt.half_groups, GROUPS)
    label = precision_label(opt.precision, groups, GROUPS)
    suffix = '_bidir' if opt.bidirectional else ''
    out = opt.out or WEIGHTS_DIR / f'romav2_{label}_{opt.height}x{opt.width}{suffix}.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)

    upstream = load_upstream(fp32=True, bidirectional=opt.bidirectional)
    pairs = sample_pairs()
    if not opt.skip_check:
        self_check(upstream, opt.height, opt.width, opt.bidirectional, pairs)

    model = StaticRoMaV2(upstream, opt.height, opt.width, opt.precision, opt.half_groups, opt.bidirectional)
    model = model.cuda().eval()
    output_names = ['warp_AB', 'confidence_AB'] + (['warp_BA', 'confidence_BA'] if opt.bidirectional else [])
    inputs = tuple(to_tensor(load_rgb(p, opt.height, opt.width)) for p in pairs[0])
    with torch.no_grad():
        torch.onnx.export(model, inputs, str(out), input_names=['image0', 'image1'], output_names=output_names,
                          opset_version=opt.opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(str(out))

    ckpt = checkpoint_path()
    metadata = {'model': 'romav2', 'height': opt.height, 'width': opt.width, 'channels': 3,
                'bidirectional': opt.bidirectional, 'precision': label, 'dtype': opt.precision,
                'half_groups': list(groups),
                'group_dtypes': {g: str(d).removeprefix('torch.') for g, d in group_dtypes(opt.precision,
                                                                                          opt.half_groups).items()},
                'outputs': output_names, 'upstream_commit': upstream_commit(),
                'torch': torch.__version__, 'opset': opt.opset, 'checkpoint': str(ckpt),
                'checkpoint_sha256': sha256(ckpt), 'onnx_sha256': sha256(out)}
    sidecar_path(out).write_text(json.dumps(metadata, indent=2))
    print(f'Exported {out}')


if __name__ == '__main__':
    main()
