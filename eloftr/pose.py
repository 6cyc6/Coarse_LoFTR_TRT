"""Relative pose metrics of the LoFTR / EfficientLoFTR test protocol (upstream src/utils/metrics.py), in numpy."""
import cv2
import numpy as np


def normalize_keypoints(kpts, K):
    return (kpts - K[[0, 1], [2, 2]][None]) / K[[0, 1], [0, 1]][None]


def estimate_pose(kpts0, kpts1, K0, K1, thresh=0.5, conf=0.99999):
    """Essential matrix by OpenCV RANSAC with a `thresh` pixel threshold -> (R, t, inlier mask), None on failure."""
    if len(kpts0) < 5:
        return None
    kpts0, kpts1 = normalize_keypoints(kpts0, K0), normalize_keypoints(kpts1, K1)
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    E, mask = cv2.findEssentialMat(kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        return None
    # the essential matrix candidates are stacked as 3x3 blocks; keep the one with the most cheirality inliers
    best_num_inliers, ret = 0, None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)  # same call as upstream
        if n > best_num_inliers:
            best_num_inliers, ret = n, (R, t[:, 0], mask.ravel() > 0)
    return ret


def relative_pose_error(T_0to1, R, t):
    """Angular errors in degrees of the translation direction (up to sign) and of the rotation."""
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # essential matrix sign ambiguity
    cos = np.clip((np.trace(np.dot(R.T, T_0to1[:3, :3])) - 1) / 2, -1.0, 1.0)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return t_err, R_err


def pose_error(kpts0, kpts1, K0, K1, T_0to1, thresh=0.5, conf=0.99999):
    """max(translation, rotation) angular error in degrees, inf when no pose is found."""
    ret = estimate_pose(kpts0, kpts1, K0, K1, thresh, conf)
    if ret is None:
        return np.inf
    return float(max(relative_pose_error(T_0to1, ret[0], ret[1])))


def symmetric_epipolar_distance(kpts0, kpts1, T_0to1, K0, K1):
    """Squared symmetric epipolar distance of the matches to the ground-truth essential matrix, in normalized
    coordinates (upstream compares it with 5e-4 on ScanNet and 1e-4 on MegaDepth)."""
    t = T_0to1[:3, 3]
    t_x = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = t_x @ T_0to1[:3, :3]
    p0 = np.concatenate([normalize_keypoints(kpts0, K0), np.ones((len(kpts0), 1))], 1)
    p1 = np.concatenate([normalize_keypoints(kpts1, K1), np.ones((len(kpts1), 1))], 1)
    Ep0 = p0 @ E.T
    Etp1 = p1 @ E
    p1Ep0 = np.sum(p1 * Ep0, -1)
    return p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2) + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2))


def pose_auc(errors, thresholds=(5, 10, 20)):
    """Area under the recall curve of the pose errors up to each threshold (degrees), normalized to [0, 1]."""
    errors = [0] + sorted(errors)
    recall = list(np.linspace(0, 1, len(errors)))
    aucs = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs[thr] = float(np.trapz(y, x) / thr)
    return aucs
