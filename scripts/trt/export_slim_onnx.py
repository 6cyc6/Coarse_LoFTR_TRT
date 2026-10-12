import argparse
import json
from pathlib import Path

import onnx
import torch

from eloftr import sha256, sidecar_path
from eloftr.evaluation import load_gray, sample_pairs, summarize, to_tensor
from eloftr.static_model import precision_groups, precision_label
from slim.evaluation import compare, static_matches, upstream_matches
from slim.ops import PLUGIN_LIBRARY
from slim.static_model import GROUPS, PRECISIONS, StaticSLiM
from slim.upstream import DEFAULT_CKPT, ROOT, WEIGHTS_DIR, load_upstream, run_upstream, upstream_commit

OUTPUT_NAMES = ['keypoints0', 'keypoints1', 'confidence', 'valid']


@torch.no_grad()
def self_check(upstream, height, width, pairs):
    """StaticSLiM in fp32 must reproduce upstream SLiM. With mamba_ssm's Mamba2 kernels it does so exactly; the
    exportable Mamba2 (plain fp32 matmuls) differs from them by their TF32 rounding."""
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    images = [[to_tensor(load_gray(p, height, width)) for p in pair] for pair in pairs]
    refs = [upstream_matches(run_upstream(upstream, *pair)) for pair in images]
    summaries = {}
    for exportable in (False, True):
        model = StaticSLiM(upstream, height, width, exportable=exportable).cuda().eval()
        results = [compare(ref, static_matches(*model(*pair))) for ref, pair in zip(refs, images)]
        summaries[exportable] = summarize(results)
        print(f'Self-check vs upstream fp32 on {len(pairs)} pairs ({"exportable" if exportable else "mamba_ssm"} '
              f'Mamba2): {summaries[exportable]}')
    torch.backends.cudnn.allow_tf32 = tf32
    exact, exportable = summaries[False], summaries[True]
    if exact['jaccard_min'] < 0.99 or exact['flow_median'] > 1e-2 or exact['flow_gt1px'] > 1e-3:
        raise RuntimeError('StaticSLiM does not reproduce upstream SLiM, refusing to export')
    if exportable['jaccard_mean'] < 0.97 or exportable['flow_median'] > 1e-2:
        raise RuntimeError('The exportable Mamba2 changes the matches beyond the TF32 noise of upstream')


def main():
    parser = argparse.ArgumentParser(description='Export a fixed-shape SLiM ONNX model (indoor test settings).')
    parser.add_argument('--height', type=int, default=480, help='Input height, multiple of 32.')
    parser.add_argument('--width', type=int, default=640, help='Input width, multiple of 32.')
    parser.add_argument('--precision', choices=list(PRECISIONS), default='mixed',
                        help='mixed: fp16 convolutions and projections; fp32 scans, matching, LayerNorms and softmaxes.')
    parser.add_argument('--fp16', type=str, default=None,
                        help='Override the fp16 module groups, comma-separated subset of backbone,refine.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT, help='SLiM checkpoint.')
    parser.add_argument('--config', default='indoor_test', help='Upstream config (configs/cfg_<config>.py).')
    parser.add_argument('--out', type=Path, default=None, help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--skip-check', action='store_true', help='Skip the comparison against upstream.')
    opt = parser.parse_args()
    print(opt)

    groups = precision_groups(opt.precision, opt.fp16, GROUPS, PRECISIONS)
    label = precision_label(groups, PRECISIONS)
    variant = opt.config.removesuffix('_test')
    out = opt.out or WEIGHTS_DIR / f'slim_{variant}_{label}_{opt.height}x{opt.width}.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)

    upstream = load_upstream(opt.ckpt, opt.config)
    pairs = sample_pairs()
    if not opt.skip_check:
        self_check(upstream, opt.height, opt.width, pairs)

    model = StaticSLiM(upstream, opt.height, opt.width, groups).cuda().eval()
    inputs = tuple(to_tensor(load_gray(p, opt.height, opt.width)) for p in pairs[0])
    with torch.no_grad():
        torch.onnx.export(model, inputs, str(out), input_names=['image0', 'image1'], output_names=OUTPUT_NAMES,
                          opset_version=opt.opset, custom_opsets={'slim': 1}, do_constant_folding=True)
    onnx.checker.check_model(str(out))

    metadata = {'model': 'slim', 'variant': variant, 'config': opt.config, 'height': opt.height, 'width': opt.width,
                'precision': label, 'fp16_groups': list(groups), 'max_coarse_matches': model.max_coarse,
                'max_fine_matches': model.max_fine, 'coarse_thr': model.coarse_thr, 'fine_thr': model.fine_thr,
                'refine_iters': model.refine_iters, 'outputs': OUTPUT_NAMES,
                'plugins': [str(PLUGIN_LIBRARY.relative_to(ROOT))], 'upstream_commit': upstream_commit(),
                'torch': torch.__version__, 'opset': opt.opset, 'checkpoint': str(opt.ckpt),
                'checkpoint_sha256': sha256(opt.ckpt), 'onnx_sha256': sha256(out)}
    sidecar_path(out).write_text(json.dumps(metadata, indent=2))
    print(f'Exported {out}')


if __name__ == '__main__':
    main()
