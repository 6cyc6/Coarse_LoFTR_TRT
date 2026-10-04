import argparse
import json
import time
from pathlib import Path

import tensorrt as trt
import torch

from eloftr import sidecar_path
from eloftr.trt_runtime import TRTEngine, build_engine


@torch.no_grad()
def smoke_test(engine_path, meta, num_pairs=4):
    """Compare the engine with StaticELoFTR in torch at the same precision."""
    from eloftr.evaluation import compare, load_gray, sample_pairs, static_matches, summarize, to_tensor
    from eloftr.static_model import StaticELoFTR
    from eloftr.upstream import load_upstream

    height, width = meta['height'], meta['width']
    upstream = load_upstream(meta['checkpoint'], meta['model_type'], meta['npe'])
    model = StaticELoFTR(upstream, height, width, meta['fp16_groups']).cuda().eval()
    engine = TRTEngine(engine_path)
    results = []
    for path0, path1 in sample_pairs()[:num_pairs]:
        image0, image1 = (to_tensor(load_gray(p, height, width)) for p in (path0, path1))
        out = engine(image0, image1)
        if not all(torch.isfinite(out[k]).all() for k in ('keypoints0', 'keypoints1', 'confidence')):
            raise RuntimeError('TensorRT produced non-finite outputs')
        ref = static_matches(*model(image0, image1))
        results.append(compare(ref, static_matches(*(out[k] for k in meta['outputs']))))
    summary = summarize(results)
    print(f'Smoke test vs torch StaticELoFTR ({meta["precision"]}): {summary}')
    if summary['jaccard_mean'] < 0.9 or summary['flow_median'] > 0.5:
        raise RuntimeError('TensorRT engine disagrees with the torch model')
    return summary


def main():
    parser = argparse.ArgumentParser(description='Build a TensorRT engine from an ONNX model.')
    parser.add_argument('--onnx', type=Path, required=True, help='Input ONNX model.')
    parser.add_argument('--engine', type=Path, default=None, help='Output engine, default: ONNX path with .engine.')
    parser.add_argument('--fp16', action='store_true',
                        help='Weakly-typed FP16 builder flag for untyped legacy ONNX models (TensorRT 10 only). '
                             'EfficientLoFTR exports carry their precision in the ONNX types instead.')
    parser.add_argument('--tf32', action='store_true', help='Allow TF32 for fp32 layers (lower accuracy).')
    parser.add_argument('--workspace-gib', type=float, default=4.0)
    parser.add_argument('--opt-level', type=int, default=3, help='Builder optimization level 0-5.')
    parser.add_argument('--timing-cache', type=Path, default=None,
                        help='Timing cache to reuse between builds, default: trt_timing.cache next to the engine.')
    parser.add_argument('--skip-check', action='store_true', help='Skip the comparison against torch.')
    parser.add_argument('--verbose', action='store_true')
    opt = parser.parse_args()
    print(opt)

    engine_path = opt.engine or opt.onnx.with_suffix('.engine')
    timing_cache = opt.timing_cache or engine_path.parent / 'trt_timing.cache'
    sidecar = sidecar_path(opt.onnx)
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    started = time.perf_counter()
    build_engine(opt.onnx, engine_path, strongly_typed=not opt.fp16, fp16=opt.fp16, tf32=opt.tf32,
                 workspace_gib=opt.workspace_gib, opt_level=opt.opt_level, timing_cache=timing_cache,
                 verbose=opt.verbose)
    meta.update({'onnx': str(opt.onnx), 'tensorrt': trt.__version__, 'strongly_typed': not opt.fp16,
                 'fp16_flag': opt.fp16, 'tf32': opt.tf32, 'opt_level': opt.opt_level,
                 'gpu': torch.cuda.get_device_name(), 'build_seconds': round(time.perf_counter() - started, 1)})
    print(f'Built {engine_path} in {meta["build_seconds"]} s')

    if 'model_type' in meta and not opt.skip_check:
        meta['smoke_test'] = smoke_test(engine_path, meta)
    sidecar_path(engine_path).write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
