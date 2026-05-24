"""
AR Demo: Render a 3D cube at a FIXED world position onto query images.

Per task.md: the cube is placed in the reconstructed scene at a fixed world
coordinate. If localization is accurate, the cube stays stable across frames.
If the pose jitters, the cube jitters — visually demonstrating accuracy.

Usage:
  python scripts/ar_demo.py --config configs/baseline_a.yaml \
      --poses outputs/results/baseline_a/pred_poses.json \
      --output outputs/ar_demo/baseline_a/ \
      --limit 30 --fps 5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cube_size", type=float, default=None,
                   help="Cube side length (m). Auto: 1.5 outdoor / 0.25 indoor")
    p.add_argument("--cube_center", nargs=3, type=float, default=None,
                   help="Cube world position x y z. Auto from COLMAP/GT if omitted.")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--line_width", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--fps", type=int, default=5)
    return p.parse_args()


def quat_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def make_cube_geometry(center, size):
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
        (0, 1, 2), (0, 2, 3),
        (5, 4, 7), (5, 7, 6),
        (4, 0, 3), (4, 3, 7),
        (1, 5, 6), (1, 6, 2),
        (3, 2, 6), (3, 6, 7),
        (4, 5, 1), (4, 1, 0),
    ]
    return verts, edges, faces


def project_points(pts3d_world, t_c2w, q_w2c, K):
    R_w2c = quat_to_rotmat(q_w2c)
    pts_cam = (R_w2c @ pts3d_world.T).T + (-R_w2c @ t_c2w)
    pts_img = (K @ pts_cam.T).T
    depth = pts_img[:, 2]
    pts2d = pts_img[:, :2] / depth[:, np.newaxis]
    return pts2d, depth


def draw_cube(image, verts2d, edges, faces, depth, color, alpha, line_width):
    h, w = image.shape[:2]
    overlay = image.copy()
    color_bgr = tuple(int(c) for c in color)

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

    for i, j in edges:
        if depth[i] <= 0 or depth[j] <= 0:
            continue
        p1 = tuple(verts2d[i].astype(int))
        p2 = tuple(verts2d[j].astype(int))
        if (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h):
            cv2.line(image, p1, p2, color_bgr, line_width)

    return image


def find_scene_center(config_path):
    """Find fixed world position for cube from COLMAP/NVM model or GT poses."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    root = Path(dataset_cfg["root"])
    scene = dataset_cfg.get("scene", "")
    dataset_name = dataset_cfg.get("name", "cambridge")

    if dataset_name == "cambridge":
        # Try COLMAP model
        colmap_candidates = [
            root / "colmap_model" / "CambridgeLandmarks_Colmap_Retriangulated_1024px"
            / scene / "model_train",
            root / "colmap_reconstruction",
            root / "colmap_model",
        ]
        for cand in colmap_candidates:
            points_bin = cand / "points3D.bin"
            if points_bin.exists():
                try:
                    from scripts.colmap_localization import COLMAPLocalizationModel
                    model = COLMAPLocalizationModel(str(cand))
                    if hasattr(model, "points3D") and model.points3D:
                        all_xyz = np.array([v["xyz"] for v in model.points3D.values()])
                        center = np.median(all_xyz, axis=0)
                        print(f"  Scene center from COLMAP ({len(all_xyz)} pts): {center}")
                        return center.tolist()
                except Exception as e:
                    print(f"  COLMAP center failed: {e}")

        # Try NVM model
        nvm_path = root / dataset_cfg.get("nvm_model", "reconstruction.nvm")
        if nvm_path.exists():
            try:
                from scripts.nvm_model import NVMModel
                nvm = NVMModel(str(nvm_path))
                if hasattr(nvm, "point3D_xyz") and nvm.point3D_xyz:
                    all_xyz = np.array(list(nvm.point3D_xyz.values()))
                    center = np.median(all_xyz, axis=0)
                    print(f"  Scene center from NVM ({len(all_xyz)} pts): {center}")
                    return center.tolist()
            except Exception as e:
                print(f"  NVM center failed: {e}")

    if dataset_name == "7scenes":
        # Estimate from GT poses
        scene_root = root / scene
        positions = []
        for seq_dir in scene_root.iterdir():
            if not seq_dir.is_dir():
                continue
            pose_files = list(seq_dir.glob("*.pose.txt"))
            if not pose_files:
                for child in seq_dir.iterdir():
                    if child.is_dir():
                        pose_files.extend(child.glob("*.pose.txt"))
            for pf in pose_files[:50]:
                try:
                    pose = np.loadtxt(pf)
                    positions.append(pose[:3, 3])
                except Exception:
                    pass
        if positions:
            positions = np.array(positions)
            center = np.median(positions, axis=0)
            center[1] += 0.2  # slightly above floor
            print(f"  Scene center from GT poses ({len(positions)} cameras): {center}")
            return center.tolist()

    print("  WARNING: No scene model found, using origin [0,0,0]")
    return [0.0, 0.0, 0.0]


def build_K(dataset_cfg):
    if "camera_matrix" in dataset_cfg:
        return np.array(dataset_cfg["camera_matrix"], dtype=np.float64)
    fx = dataset_cfg.get("fx", 585.0)
    fy = dataset_cfg.get("fy", 585.0)
    cx = dataset_cfg.get("cx", 320.0)
    cy = dataset_cfg.get("cy", 240.0)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def auto_cube_size(dataset_name):
    return 0.25 if dataset_name == "7scenes" else 1.2


def load_image(root, img_name):
    path = root / img_name
    img = cv2.imread(str(path))
    if img is None:
        img = cv2.imread(str(root / Path(img_name).name))
    return img


def sample_frames(items, limit):
    """Select frames from the longest sequence for smooth, continuous video."""
    if limit <= 0 or limit >= len(items):
        return items

    # Group frames by sequence (first path component)
    import re
    seqs = {}
    for name, data in items:
        parts = str(name).replace('\\', '/').split('/')
        seq = parts[0] if parts else 'default'
        if seq not in seqs:
            seqs[seq] = []
        seqs[seq].append((name, data))

    # Pick the longest sequence for continuous video
    best_seq = max(seqs.keys(), key=lambda s: len(seqs[s]))
    seq_items = seqs[best_seq]

    # Sort by frame number within sequence for temporal order
    def _frame_num(item):
        nums = re.findall(r'\d+', str(item[0]))
        return int(nums[-1]) if nums else 0
    seq_items.sort(key=_frame_num)

    # Evenly sample from this single sequence
    if limit >= len(seq_items):
        return seq_items
    step = len(seq_items) / limit
    return [seq_items[int(i * step)] for i in range(limit)]


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

    # Determine cube world position
    if args.cube_center is not None:
        cube_center = tuple(args.cube_center)
        print(f"  Cube center (user): {cube_center}")
    else:
        cube_center = tuple(find_scene_center(args.config))

    # Load poses for fallback cube placement
    with open(args.poses, encoding="utf-8") as f:
        pred_poses = json.load(f)

    # If scene center could not be determined, fall back to median camera position
    if np.linalg.norm(cube_center) < 0.01:
        cam_positions = np.array([v["t"] for v in list(pred_poses.values())[:100]])
        cube_center = tuple(np.median(cam_positions, axis=0) + np.array([0, 0.3, 0]))
        print(f"  Cube at median camera pos: {tuple(round(float(v),2) for v in cube_center)}")

    with open(args.poses, encoding="utf-8") as f:
        pred_poses = json.load(f)

    all_frames = sorted(pred_poses.items())
    frames = sample_frames(all_frames, args.limit)

    verts_3d, edges, faces = make_cube_geometry(cube_center, cube_size)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    green = (0, 255, 0)
    red = (0, 0, 255)
    video_frames = []
    n_loc = 0

    print(f"Rendering {len(frames)} frames "
          f"(cube={cube_size}m at world {tuple(round(float(v),2) for v in cube_center)})...")

    for img_name, pose_data in tqdm(frames):
        image = load_image(root, img_name)
        if image is None:
            continue

        t_c2w = np.array(pose_data["t"])
        q_w2c = np.array(pose_data["q"])

        pts2d, depth = project_points(verts_3d, t_c2w, q_w2c, K)

        n_visible = int(np.sum(depth > 0))
        cube_behind = (n_visible == 0)

        # Green if cube is in front of camera, red only if pose is completely wrong
        is_localized = not cube_behind
        if is_localized:
            n_loc += 1
        color = green if is_localized else red

        if not cube_behind:
            image = draw_cube(image.copy(), pts2d, edges, faces, depth,
                              color, args.alpha, args.line_width)

        label = "LOCALIZED" if is_localized else "FAILED (cube behind camera)"
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Use full relative path as filename to avoid collisions across seqs
        safe_name = str(img_name).replace('\\', '_').replace('/', '_').replace('.png', '').replace('.jpg', '').replace('.color', '')
        out_path = output_dir / f"{safe_name}_ar.jpg"
        cv2.imwrite(str(out_path), image)
        video_frames.append(image)

    print(f"Saved {len(video_frames)} AR images ({n_loc}/{len(video_frames)} visible) "
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
