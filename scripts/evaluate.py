import numpy as np

THRESHOLDS = [
    (0.25, 2.0),
    (0.5, 5.0),
    (5.0, 10.0),
]


def quaternion_angular_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Returns angular error in degrees between two unit quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * np.degrees(np.arccos(dot))


def compute_recall(
    pred_poses: dict,
    gt_poses: dict,
) -> dict:
    """
    Returns:
        {
          "(0.25m, 2deg)": float,
          "(0.5m, 5deg)":  float,
          "(5m, 10deg)":   float,
          "n_query":       int,
          "n_localized":   int,
        }
    """
    results = {}
    n = len(gt_poses)
    for t_thr, r_thr in THRESHOLDS:
        count = 0
        for name, (t_pred, q_pred) in pred_poses.items():
            if name not in gt_poses:
                continue
            t_gt, q_gt = gt_poses[name]
            t_err = np.linalg.norm(t_pred - t_gt)
            r_err = quaternion_angular_error(q_pred, q_gt)
            if t_err < t_thr and r_err < r_thr:
                count += 1
        key = f"({t_thr}m, {int(r_thr)}deg)"
        results[key] = (count / n) if n > 0 else 0.0
    results["n_query"] = n
    results["n_localized"] = sum(
        1 for name, (t_pred, q_pred) in pred_poses.items()
        if name in gt_poses
        and np.linalg.norm(t_pred - gt_poses[name][0]) < THRESHOLDS[0][0]
        and quaternion_angular_error(q_pred, gt_poses[name][1]) < THRESHOLDS[0][1]
    )
    return results