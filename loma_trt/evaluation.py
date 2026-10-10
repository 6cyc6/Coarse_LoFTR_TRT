"""Compare LoMa match sets of upstream, StaticLoMa and TensorRT engines.

LoMa matches keypoints that each model detects itself, in its own order (top-k by probability), so match sets are
compared through the keypoints of image0: two keypoints are the same detection when they are mutual nearest
neighbours within `tol` pixels. Matches are then keyed by the detection (eloftr.evaluation.compare), and their
displacements kpts1 - kpts0 are compared.
"""
import numpy as np
import torch

from eloftr.evaluation import compare, load_rgb, sample_pairs, static_matches, summarize, to_tensor
from loma_trt.upstream import run_upstream, to_pixels


def keypoint_alignment(ref, other, tol=0.5):
    """For each keypoint of ref ([K, 2]), the index of its mutual nearest neighbour in other within tol, else -1."""
    distance = torch.cdist(ref.double(), other.double())
    nearest, nearest_back = distance.argmin(1), distance.argmin(0)
    ids = torch.arange(len(ref), device=ref.device)
    aligned = (nearest_back[nearest] == ids) & (distance[ids, nearest] < tol)
    return torch.where(aligned, nearest, torch.full_like(nearest, -1))


def upstream_outputs(model, image0, image1, num_keypoints=None):
    """Upstream LoMa results in the output format of StaticLoMa (all keypoints of image0, pixels, valid mask)."""
    height, width = image0.shape[-2:]
    out = run_upstream(model, image0, image1, num_keypoints)
    matches0 = out['matches0']
    valid = matches0 > -1
    keypoints0 = to_pixels(out['keypoints0'], height, width)
    keypoints1 = to_pixels(out['keypoints1'], height, width)[matches0.clamp(min=0)]
    return keypoints0, keypoints1, out['scores0'].float(), valid


def aligned_compare(ref, other, tol=0.5):
    """compare() of two outputs (keypoints0, keypoints1, confidence, valid), with ref's keypoints keyed by the index of
    the aligned keypoint in other (unaligned ones get keys no other keypoint has)."""
    alignment = keypoint_alignment(ref[0].float(), other[0].float(), tol)
    ref_matches, other_matches = static_matches(*ref), static_matches(*other)
    keys = alignment.cpu().numpy()[ref_matches['i']]
    unaligned = keys < 0
    keys[unaligned] = len(other[0]) + np.arange(unaligned.sum())  # never shared
    ref_matches['i'] = keys
    result = compare(ref_matches, other_matches)
    result['aligned'] = float((alignment >= 0).float().mean())
    return result


def summarize_aligned(results):
    summary = summarize(results)
    summary['aligned_min'] = float(np.min([r['aligned'] for r in results]))
    return summary


@torch.no_grad()
def smoke_test(engine_path, meta, num_pairs=4):
    """Compare a LoMa engine with StaticLoMa in fp32 (= upstream fp32, checked at export).

    Which 2048 keypoints survive the top-k and where their sub-pixel refinement lands is sensitive to the precision:
    upstream's own bf16 inference keeps only ~80% of the fp32 keypoints. A half-precision engine therefore has to agree
    with fp32 about as well as StaticLoMa in torch at the same precision does, not exactly.
    """
    from eloftr.trt_runtime import TRTEngine
    from loma_trt.static_model import StaticLoMa
    from loma_trt.upstream import load_upstream

    height, width = meta['height'], meta['width']
    upstream = load_upstream(meta['variant'], fp32=True)
    reference = StaticLoMa(upstream, height, width, meta['num_keypoints']).cuda().eval()
    same = None
    if meta['half_groups']:
        same = StaticLoMa(upstream, height, width, meta['num_keypoints'], meta['dtype'],
                          ','.join(meta['half_groups'])).cuda().eval()
    del upstream
    engine = TRTEngine(engine_path)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False  # the engines are built without TF32
    results, noise = [], []
    for path0, path1 in sample_pairs()[:num_pairs]:
        image0, image1 = (to_tensor(load_rgb(p, height, width)) for p in (path0, path1))
        out = engine(image0, image1)
        out = [out[k] for k in meta['outputs']]
        if not all(torch.isfinite(t).all() for t in out[:3]):
            raise RuntimeError('TensorRT produced non-finite outputs')
        ref = reference(image0, image1)
        results.append(aligned_compare(ref, out))
        noise.append(aligned_compare(ref, same(image0, image1)) if same is not None else None)
    torch.backends.cudnn.allow_tf32 = tf32
    summary = summarize_aligned(results)
    exact = {'jaccard_mean': 1.0, 'aligned_min': 1.0, 'flow_gt1px': 0.0}
    floor = summarize_aligned(noise) if same is not None else exact
    print(f'Smoke test vs torch StaticLoMa fp32: {summary}')
    if same is not None:
        print(f'  torch StaticLoMa {meta["precision"]} vs fp32 (precision noise): {floor}')
    if (summary['jaccard_mean'] < floor['jaccard_mean'] - 0.1 or summary['aligned_min'] < floor['aligned_min'] - 0.1
            or summary['flow_gt1px'] > floor['flow_gt1px'] + 0.05):
        raise RuntimeError('TensorRT engine disagrees with the torch model beyond its precision noise')
    return {'engine_vs_fp32': summary, 'torch_same_precision_vs_fp32': floor}
