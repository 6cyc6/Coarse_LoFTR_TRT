"""Shared helpers to compare match sets of upstream EfficientLoFTR, StaticELoFTR and TensorRT engines."""
from itertools import combinations

import cv2
import numpy as np
import torch

from eloftr.upstream import SAMPLE_IMAGES_DIR


def load_gray(path, height, width):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_rgb(path, height, width):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def to_tensor(image, device='cuda'):
    """uint8 [H, W] grayscale or [H, W, 3] color image -> [1, C, H, W] float tensor in [0, 1]."""
    tensor = torch.from_numpy(image)
    tensor = tensor[None, None] if tensor.ndim == 2 else tensor.permute(2, 0, 1)[None]
    return tensor.to(device=device, dtype=torch.float32) / 255.


def sample_pairs(directory=SAMPLE_IMAGES_DIR):
    """All same-scene pairs of the upstream sample images (scene = file name without the two id parts)."""
    scenes = {}
    for path in sorted(directory.glob('*.jpg')):
        scenes.setdefault(path.stem.rsplit('_', 2)[0], []).append(path)
    return [pair for paths in scenes.values() for pair in combinations(paths, 2)]


def upstream_matches(batch):
    return {'i': batch['i_ids'].cpu().numpy(),
            'kpts0': batch['mkpts0_f'].float().cpu().numpy(),
            'kpts1': batch['mkpts1_f'].float().cpu().numpy(),
            'conf': batch['mconf'].float().cpu().numpy()}


def static_matches(keypoints0, keypoints1, confidence, valid):
    i = torch.nonzero(valid).squeeze(1)
    return {'i': i.cpu().numpy(),
            'kpts0': keypoints0[i].cpu().numpy(),
            'kpts1': keypoints1[i].cpu().numpy(),
            'conf': confidence[i].cpu().numpy()}


def compare(ref, other):
    """Match-set agreement keyed by the coarse cell of image0 (each cell has at most one match).

    The fine stage picks one pixel pair inside an 8x8 window by argmax, and precision changes often
    pick a different, equally valid pair (both endpoints move together). Fine agreement is therefore
    measured on the match displacement kpts1 - kpts0 rather than on the endpoints.
    """
    ref_i, other_i = ref['i'], other['i']
    shared, ref_at, other_at = np.intersect1d(ref_i, other_i, assume_unique=True, return_indices=True)
    union = len(ref_i) + len(other_i) - len(shared)
    flow_ref = ref['kpts1'][ref_at] - ref['kpts0'][ref_at]
    flow_other = other['kpts1'][other_at] - other['kpts0'][other_at]
    return {'n_ref': len(ref_i), 'n': len(other_i), 'n_shared': len(shared),
            'jaccard': len(shared) / union if union else 1.0,
            'flow_err': np.linalg.norm(flow_ref - flow_other, axis=1)}


def summarize(results):
    """Aggregate per-pair `compare` results into one row."""
    err = np.concatenate([r['flow_err'] for r in results]) if results else np.zeros(0)
    n_ref = sum(r['n_ref'] for r in results)
    n = sum(r['n'] for r in results)
    return {'count_ratio': n / n_ref if n_ref else float('nan'),
            'jaccard_mean': float(np.mean([r['jaccard'] for r in results])),
            'jaccard_min': float(np.min([r['jaccard'] for r in results])),
            'flow_median': float(np.median(err)) if len(err) else 0.0,
            'flow_p95': float(np.percentile(err, 95)) if len(err) else 0.0,
            'flow_gt1px': float((err > 1).mean()) if len(err) else 0.0}
