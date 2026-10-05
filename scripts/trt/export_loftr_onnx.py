import argparse
import json
from pathlib import Path

import kornia
import onnx
import torch

from eloftr import sha256, sidecar_path
from eloftr.evaluation import compare, load_gray, sample_pairs, static_matches, summarize, to_tensor
from eloftr.static_model import precision_groups, precision_label
from loftr_full.static_model import GROUPS, PRECISIONS, StaticLoFTR
from loftr_full.upstream import PRETRAINED, WEIGHTS_DIR, checkpoint_path, load_upstream, run_upstream, upstream_matches

OUTPUT_NAMES = ['keypoints0', 'keypoints1', 'confidence', 'valid']


@torch.no_grad()
def self_check(upstream, height, width, pairs):
    """StaticLoFTR in fp32 must reproduce the kornia LoFTR matches."""
    model = StaticLoFTR(upstream, height, width).cuda().eval()
    results = []
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    for path0, path1 in pairs:
        image0, image1 = (to_tensor(load_gray(p, height, width)) for p in (path0, path1))
        ref = upstream_matches(run_upstream(upstream, image0, image1), width)
        results.append(compare(ref, static_matches(*model(image0, image1))))
    torch.backends.cudnn.allow_tf32 = tf32
    summary = summarize(results)
    print(f'Self-check vs kornia LoFTR fp32 on {len(pairs)} pairs: {summary}')
    if summary['jaccard_min'] < 0.99 or summary['flow_median'] > 1e-2 or summary['flow_gt1px'] > 1e-3:
        raise RuntimeError('StaticLoFTR does not reproduce kornia LoFTR, refusing to export')


def main():
    parser = argparse.ArgumentParser(description='Export a fixed-shape LoFTR (coarse + fine) ONNX model.')
    parser.add_argument('--height', type=int, default=480, help='Input height, multiple of 8.')
    parser.add_argument('--width', type=int, default=640, help='Input width, multiple of 8.')
    parser.add_argument('--pretrained', choices=PRETRAINED, default='outdoor',
                        help='Released weights: outdoor (MegaDepth), indoor_new or indoor (ScanNet).')
    parser.add_argument('--precision', choices=list(PRECISIONS), default='mixed',
                        help='mixed: fp16 backbone and transformer linear layers; fp32 attention, matching, '
                             'LayerNorms and softmaxes.')
    parser.add_argument('--fp16', type=str, default=None,
                        help='Override the fp16 module groups, comma-separated subset of backbone,coarse,fine.')
    parser.add_argument('--ckpt', type=Path, default=None,
                        help='Local LoFTR checkpoint instead of the released weights, e.g. outdoor_ds.ckpt.')
    parser.add_argument('--out', type=Path, default=None, help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--skip-check', action='store_true', help='Skip the comparison against kornia.')
    opt = parser.parse_args()
    print(opt)

    groups = precision_groups(opt.precision, opt.fp16, GROUPS, PRECISIONS)
    label = precision_label(groups, PRECISIONS)
    out = opt.out or WEIGHTS_DIR / f'loftr_{opt.pretrained}_{label}_{opt.height}x{opt.width}.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)

    upstream = load_upstream(opt.pretrained, opt.ckpt)
    ckpt = opt.ckpt or checkpoint_path(opt.pretrained)
    pairs = sample_pairs()
    if not opt.skip_check:
        self_check(upstream, opt.height, opt.width, pairs)

    model = StaticLoFTR(upstream, opt.height, opt.width, groups).cuda().eval()
    inputs = tuple(to_tensor(load_gray(p, opt.height, opt.width)) for p in pairs[0])
    with torch.no_grad():
        torch.onnx.export(model, inputs, str(out), input_names=['image0', 'image1'], output_names=OUTPUT_NAMES,
                          opset_version=opt.opset, do_constant_folding=True)
    onnx.checker.check_model(str(out))

    metadata = {'model': 'loftr', 'pretrained': opt.pretrained, 'height': opt.height, 'width': opt.width,
                'precision': label, 'fp16_groups': list(groups), 'thr': model.thr, 'outputs': OUTPUT_NAMES,
                'kornia': kornia.__version__, 'torch': torch.__version__, 'opset': opt.opset,
                'checkpoint': str(ckpt), 'checkpoint_sha256': sha256(ckpt), 'onnx_sha256': sha256(out)}
    sidecar_path(out).write_text(json.dumps(metadata, indent=2))
    print(f'Exported {out}')


if __name__ == '__main__':
    main()
