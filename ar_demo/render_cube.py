import numpy as np
import cv2


def render_cube_on_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cube_center_world: np.ndarray,
    cube_size: float = 1.0,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Projects a unit cube into the image and blends it.
    """
    half = cube_size / 2.0
    corners_local = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float32) * half
    corners_world = corners_local + cube_center_world

    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(corners_world, rvec, t, camera_matrix, dist_coeffs)
    pts2d = pts2d.reshape(-1, 2).astype(int)

    EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    rendered = image.copy()
    for i, j in EDGES:
        cv2.line(rendered, tuple(pts2d[i]), tuple(pts2d[j]), (0, 255, 0), 2)

    overlay = rendered.copy()
    face = pts2d[[4, 5, 6, 7]]
    cv2.fillConvexPoly(overlay, face, (0, 200, 100))
    rendered = cv2.addWeighted(overlay, alpha, rendered, 1 - alpha, 0)

    return rendered


def rotmat_to_quaternion(R):
    """Convert rotation matrix to quaternion (qw, qx, qy, qz)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)