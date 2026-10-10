"""Compare the dense outputs of upstream RoMa v2, StaticRoMaV2 and TensorRT engines.

The warps are compared by their end-point error in pixels of the target image, on the pixels the reference predicts
to overlap (overlap probability above 0.5); the overlap probabilities are compared everywhere.
"""
import numpy as np
import torch

from eloftr.evaluation import load_rgb, sample_pairs, to_tensor

OVERLAP_THR = 0.5


def dense_compare(ref, other):
    """ref, other: (warp_AB, confidence_AB[, warp_BA, confidence_BA]) with warps [1, H, W, 2] in normalized coordinates
    and confidences [1, H, W, 4] (overlap logit first)."""
    epe, overlap_diff, agree, covered = [], [], [], []
    for (warp_r, conf_r), (warp_o, conf_o) in zip(zip(ref[::2], ref[1::2]), zip(other[::2], other[1::2])):
        height, width = warp_r.shape[1:3]
        scale = torch.tensor([width / 2, height / 2], device=warp_r.device)
        overlap_r, overlap_o = torch.sigmoid(conf_r[..., 0].float()), torch.sigmoid(conf_o[..., 0].float())
        mask = overlap_r > OVERLAP_THR
        epe.append((((warp_r.float() - warp_o.float()) * scale).norm(dim=-1)[mask]).cpu().numpy())
        overlap_diff.append((overlap_r - overlap_o).abs().flatten().cpu().numpy())
        agree.append(float(((overlap_o > OVERLAP_THR) == mask).float().mean()))
        covered.append(float(mask.float().mean()))
    return {'epe': np.concatenate(epe), 'overlap_diff': np.concatenate(overlap_diff),
            'overlap_agree': float(np.mean(agree)), 'overlap_frac': float(np.mean(covered))}


def summarize_dense(results):
    epe = np.concatenate([r['epe'] for r in results])
    overlap_diff = np.concatenate([r['overlap_diff'] for r in results])
    return {'overlap_frac': float(np.mean([r['overlap_frac'] for r in results])),
            'overlap_agree': float(np.mean([r['overlap_agree'] for r in results])),
            'overlap_mae': float(overlap_diff.mean()),
            'epe_median': float(np.median(epe)) if len(epe) else 0.0,
            'epe_p95': float(np.percentile(epe, 95)) if len(epe) else 0.0,
            'epe_gt1px': float((epe > 1).mean()) if len(epe) else 0.0}


def upstream_outputs(preds):
    """run_upstream's dict as the output tuple of StaticRoMaV2."""
    keys = ('warp_AB', 'confidence_AB', 'warp_BA', 'confidence_BA')
    return tuple(preds[k] for k in keys if preds[k] is not None)


@torch.no_grad()
def smoke_test(engine_path, meta, num_pairs=4):
    """Compare a RoMa v2 engine with StaticRoMaV2 in fp32 (= upstream fp32, checked at export); a half-precision engine
    has to agree with it about as well as StaticRoMaV2 in torch at the same precision does. With DINOv3 in bf16 (as in
    every half-precision mode) the warp of some hard pairs jumps by pixels in whole regions, in upstream's own bf16
    too, so the share of pixels off by more than 1 px is compared relative to that noise."""
    from eloftr.trt_runtime import TRTEngine
    from romav2_trt.static_model import StaticRoMaV2
    from romav2_trt.upstream import load_upstream

    height, width, bidirectional = meta['height'], meta['width'], meta['bidirectional']
    upstream = load_upstream(fp32=True)
    reference = StaticRoMaV2(upstream, height, width, bidirectional=bidirectional).cuda().eval()
    same = None
    if meta['half_groups']:
        same = StaticRoMaV2(upstream, height, width, meta['dtype'], ','.join(meta['half_groups']),
                            bidirectional).cuda().eval()
    del upstream
    engine = TRTEngine(engine_path)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # the engines are built without TF32
    results, noise = [], []
    for path0, path1 in sample_pairs()[:num_pairs]:
        image0, image1 = (to_tensor(load_rgb(p, height, width)) for p in (path0, path1))
        out = engine(image0, image1)
        out = [out[k] for k in meta['outputs']]
        if not all(torch.isfinite(t).all() for t in out):
            raise RuntimeError('TensorRT produced non-finite outputs')
        ref = reference(image0, image1)
        results.append(dense_compare(ref, out))
        if same is not None:
            noise.append(dense_compare(ref, same(image0, image1)))
    torch.backends.cudnn.allow_tf32 = tf32
    summary = summarize_dense(results)
    floor = summarize_dense(noise) if same is not None else {'epe_median': 0.0, 'epe_gt1px': 0.0, 'overlap_agree': 1.0}
    print(f'Smoke test vs torch StaticRoMaV2 fp32: {summary}')
    if same is not None:
        print(f'  torch StaticRoMaV2 {meta["precision"]} vs fp32 (precision noise): {floor}')
    if (summary['epe_median'] > 2 * floor['epe_median'] + 0.05 or summary['epe_gt1px'] > 1.5 * floor['epe_gt1px'] + 0.02
            or summary['overlap_agree'] < floor['overlap_agree'] - 0.02):
        raise RuntimeError('TensorRT engine disagrees with the torch model beyond its precision noise')
    return {'engine_vs_fp32': summary, 'torch_same_precision_vs_fp32': floor}
