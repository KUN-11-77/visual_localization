"""
AR Demo: Render a semi-transparent 3D cube onto query images using estimated poses.

Reads pred_poses.json from run_pipeline.py and projects a 3D cube onto query images.
Localized frames get a green cube, failed frames get a red cube.
Frames are evenly sampled from all queries for consistent video quality.

Cube is placed per-frame in front of the camera (along view direction), not at a
global scene center, ensuring visibility regardless of camera position.

Per method.md: indoor (7Scenes) cube ~0.3m, outdoor (Cambridge) cube ~1.5m,
semi-transparent (alpha ~0.6) for spatial jitter perception.

Usage:
  python scripts/ar_demo.py --config configs/baseline_a.yaml \
      --poses outputs/results/baseline_a/pred_poses.json \
      --output outputs/ar_demo/baseline_a/ \
      --limit 30 --fps 5
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Experiment YAML config")
    p.add_argument("--poses", required=True, help="pred_poses.json from pipeline")
    p.add_argument("--output", required=True, help="Output directory for AR images")
    p.add_argument("--cube_size", type=float, default=None,
                   help="Cube side length in meters (auto: 1.5 outdoor / 0.3 indoor)")
    p.add_argument("--cube_distance", type=float, default=1.0,
                   help="Distance in meters to place cube in front of camera")
    p.add_argument("--limit", type=int, default=30, help="Max frames (evenly sampled, 0=all)")
    p.add_argument("--line_width", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Face fill opacity (0-1)")
    p.add_argument("--fps", type=int, default=5, help="FPS for output video (0=skip)")
    return p.parse_args()


def make_cube_geometry(center, size):
    """Return vertices, edges, and triangular faces for a cube."""
    d = size / 2.0
    cx, cy, cz = center
    verts = np.array([
        [cx - d, cy - d, cz - d], [cx + d, cy - d, cz - d],
        [cx + d, cy + d, cz - d], [cx - d, cy + d, cz - d],
        [cx - d, cy - d, cz + d], [cx + d, cy - d, cz + d],
        [cx + d, cy + d, cz + d], [cx - d, cy + d, cz + d],
    ], dtype=np.float64)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),  # front
        (5, 4, 7), (5, 7, 6),  # back
        (4, 0, 3), (4, 3, 7),  # left
        (1, 5, 6), (1, 6, 2),  # right
        (3, 2, 6), (3, 6, 7),  # top
        (4, 5, 1), (4, 1, 0),  # bottom
    ]
    return verts, edges, faces


def quat_to_rotmat(q):
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def project_points(pts3d_world, t_c2w, q_w2c, K):
    """Project world 3D points to 2D pixel coordinates.

    pts3d_world: (N, 3) world coordinates
    t_c2w: (3,) camera center in world
    q_w2c: (4,) world-to-camera quaternion [qw, qx, qy, qz]

    Returns: (N, 2) pixel coordinates, (N,) depth values
    """
    R_w2c = quat_to_rotmat(q_w2c)
    pts_cam = (R_w2c @ pts3d_world.T).T + (-R_w2c @ t_c2w)
    pts_img = (K @ pts_cam.T).T
    depth = pts_img[:, 2]
    pts2d = pts_img[:, :2] / depth[:, np.newaxis]
    return pts2d, depth


def cube_center_in_front(t_c2w, q_w2c, distance):
    """Compute cube center placed `distance` meters in front of the camera."""
    R_c2w = quat_to_rotmat(q_w2c).T  # world-to-cam → cam-to-world rotation
    forward = R_c2w[:, 2]  # 3rd column of R_c2w is the forward direction
    return t_c2w + forward * distance


def draw_cube(image, verts2d, edges, faces, depth, color, alpha, line_width):
    """Draw cube with semi-transparent filled faces and wireframe edges."""
    h, w = image.shape[:2]
    overlay = image.copy()
    color_bgr = tuple(int(c) for c in color)

    # Draw filled faces with alpha blending
    for tri in faces:
        pts = []
        valid = True
        for idx in tri:
            if depth[idx] <= 0:
                valid = False
                break
            x, y = verts2d[idx]
            if x < -50 or x > w + 50 or y < -50 or y > h + 50:
                valid = False
                break
            pts.append((int(x), int(y)))
        if valid and len(pts) == 3:
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color_bgr)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw wireframe edges on top
    for i, j in edges:
        if depth[i] <= 0 or depth[j] <= 0:
            continue
        p1 = tuple(verts2d[i].astype(int))
        p2 = tuple(verts2d[j].astype(int))
        if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
            cv2.line(image, p1, p2, color_bgr, line_width)

    return image


def load_image(root, scene, img_name, dataset_name):
    """Load query image with path resolution."""
    path = root / img_name
    img = cv2.imread(str(path))
    if img is None:
        alt_path = root / Path(img_name).name
        img = cv2.imread(str(alt_path))
    return img


def build_K(dataset_cfg):
    """Build camera intrinsic matrix from config (original resolution)."""
    if "camera_matrix" in dataset_cfg:
        return np.array(dataset_cfg["camera_matrix"], dtype=np.float64)
    fx = dataset_cfg.get("fx", 585.0)
    fy = dataset_cfg.get("fy", 585.0)
    cx = dataset_cfg.get("cx", 320.0)
    cy = dataset_cfg.get("cy", 240.0)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def auto_cube_size(dataset_name):
    """Return appropriate cube size for dataset type."""
    if dataset_name == "7scenes":
        return 0.3  # indoor, per method.md
    return 1.5  # outdoor Cambridge


def sample_frames(frame_items, limit):
    """Evenly sample frames from sorted items for consistent video coverage."""
    if limit <= 0 or limit >= len(frame_items):
        return frame_items
    step = len(frame_items) / limit
    sampled = [frame_items[int(i * step)] for i in range(limit)]
    return sampled


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg.get("name", "cambridge")
    root = Path(dataset_cfg["root"])
    scene = dataset_cfg.get("scene", "")
    K = build_K(dataset_cfg)

    cube_size = args.cube_size if args.cube_size is not None else auto_cube_size(dataset_name)

    img_root = root  # img names already include scene prefix for 7scenes

    with open(args.poses, encoding="utf-8") as f:
        pred_poses = json.load(f)

    all_frames = sorted(pred_poses.items())
    frames = sample_frames(all_frames, args.limit)

    verts_3d_base, edges, faces = make_cube_geometry((0, 0, 0), cube_size)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    green = (0, 255, 0)
    red = (0, 0, 255)
    video_frames = []

    n_localized = 0
    n_total = 0

    print(f"Rendering {len(frames)} AR frames (cube={cube_size}m, alpha={args.alpha}, "
          f"distance={args.cube_distance}m)...")
    for img_name, pose_data in tqdm(frames):
        image = load_image(img_root, scene, img_name, dataset_name)
        if image is None:
            continue

        t_c2w = np.array(pose_data["t"])
        q_w2c = np.array(pose_data["q"])

        # Place cube in front of camera, not at global origin
        center = cube_center_in_front(t_c2w, q_w2c, args.cube_distance)
        verts_3d, _, _ = make_cube_geometry(tuple(center), cube_size)

        pts2d, depth = project_points(verts_3d, t_c2w, q_w2c, K)

        if len(pts2d) == 0 or np.all(depth <= 0):
            continue

        n_total += 1
        n_visible = int(np.sum(depth > 0))
        is_localized = n_visible >= 4
        if is_localized:
            n_localized += 1
        color = green if is_localized else red

        image_ar = draw_cube(image.copy(), pts2d, edges, faces, depth,
                             color, args.alpha, args.line_width)

        status = "LOCALIZED" if is_localized else "FAILED"
        t_err_val = pose_data.get('t_err', None)
        t_err_str = f"{float(t_err_val):.3f}" if t_err_val is not None else "?"
        cv2.putText(image_ar, f"{status} | t_err={t_err_str}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out_path = output_dir / f"{Path(img_name).stem}_ar.jpg"
        cv2.imwrite(str(out_path), image_ar)
        video_frames.append(image_ar)

    print(f"Saved {len(video_frames)} AR images ({n_localized}/{n_total} localized) "
          f"to {output_dir}")

    if args.fps > 0 and len(video_frames) > 1:
        video_path = output_dir / "ar_demo.mp4"
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        for frm in video_frames:
            writer.write(frm)
        writer.release()
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
