"""Comparison of SLiM match sets, in the format of eloftr.evaluation (compare, then eloftr.evaluation.summarize)."""
import numpy as np
import torch


def upstream_matches(data):
    """Matches of upstream SLiM's data dict (slim.upstream.run_upstream)."""
    return {'kpts0': data['fine_coord_0'].float().cpu().numpy(), 'kpts1': data['fine_coord_1'].float().cpu().numpy()}


def static_matches(keypoints0, keypoints1, confidence, valid):
    i = torch.nonzero(valid).squeeze(1)
    return {'kpts0': keypoints0[i].float().cpu().numpy(), 'kpts1': keypoints1[i].float().cpu().numpy(),
            'conf': confidence[i].float().cpu().numpy()}


def compare(ref, other, max_dist=1.0):
    """Match-set agreement: a match of `ref` is shared when `other` has a match with the same keypoint0 (a fine pixel
    centre, exact) and a refined keypoint1 within max_dist px. SLiM can match one coarse cell several times, so
    matches are keyed by their keypoints rather than by coarse cell as in eloftr.evaluation.compare."""
    table = {}
    for p0, p1 in zip(map(tuple, other['kpts0']), other['kpts1']):
        table.setdefault(p0, []).append(p1)
    errors = []
    for p0, p1 in zip(map(tuple, ref['kpts0']), ref['kpts1']):
        candidates = table.get(p0)
        if candidates:
            d = np.linalg.norm(np.stack(candidates) - p1, axis=1).min()
            if d < max_dist:
                errors.append(d)
    n_ref, n, n_shared = len(ref['kpts0']), len(other['kpts0']), len(errors)
    union = n_ref + n - n_shared
    return {'n_ref': n_ref, 'n': n, 'n_shared': n_shared, 'jaccard': n_shared / union if union else 1.0,
            'flow_err': np.array(errors)}
