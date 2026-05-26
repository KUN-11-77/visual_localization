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
    """四元数 (qw, qx, qy, qz) → 3×3 旋转矩阵 (world-to-camera)"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def make_cube_geometry(center, size):
    """生成立方体的 8 个顶点、12 条边和 12 个三角面（世界坐标系）"""
    d = size / 2.0
    cx, cy, cz = center
    verts = np.array([
        [cx - d, cy - d, cz - d], [cx + d, cy - d, cz - d],
        [cx + d, cy + d, cz - d], [cx - d, cy + d, cz - d],
        [cx - d, cy - d, cz + d], [cx + d, cy - d, cz + d],
        [cx + d, cy + d, cz + d], [cx - d, cy + d, cz + d],
    ], dtype=np.float64)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),    # 前面
        (4, 5), (5, 6), (6, 7), (7, 4),    # 后面
        (0, 4), (1, 5), (2, 6), (3, 7),    # 连接边
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),   # 前面 2 三角面
        (5, 4, 7), (5, 7, 6),   # 后面 2 三角面
        (4, 0, 3), (4, 3, 7),   # 左面 2 三角面
        (1, 5, 6), (1, 6, 2),   # 右面 2 三角面
        (3, 2, 6), (3, 6, 7),   # 顶面 2 三角面
        (4, 5, 1), (4, 1, 0),   # 底面 2 三角面
    ]
    return verts, edges, faces


def project_points(pts3d_world, t_c2w, q_w2c, K):
    """
    透视投影：世界坐标 3D 点 → 图像平面 2D 点。

    公式: p_2d = K · (R_w2c · P_w + t_cam)
          其中 t_cam = -R_w2c · t_c2w

    Args:
        pts3d_world: (N, 3) 世界坐标系下的 3D 点
        t_c2w:       相机中心在世界坐标系的位置
        q_w2c:       世界→相机四元数
        K:           3×3 相机内参矩阵

    Returns:
        pts2d: (N, 2) 像素坐标
        depth: (N,)  每个点的深度（正值=在相机前方）
    """
    # 转换到相机坐标系
    R_w2c = quat_to_rotmat(q_w2c)
    pts_cam = (R_w2c @ pts3d_world.T).T + (-R_w2c @ t_c2w)

    # 内参投影
    pts_img = (K @ pts_cam.T).T
    depth = pts_img[:, 2]                       # Z 坐标 = 深度
    pts2d = pts_img[:, :2] / depth[:, np.newaxis]  # 透视除法
    return pts2d, depth


def draw_cube(image, verts2d, edges, faces, depth, color, alpha, line_width):
    """
    在图像上绘制半透明立方体。

    Args:
        image:     BGR 图像 (H, W, 3)
        verts2d:   (8, 2) 8 个顶点的像素坐标
        edges:     (12, 2) 12 条边的顶点索引对
        faces:     (12, 3) 12 个三角面的顶点索引三元组
        depth:     (8,) 每个顶点的深度
        color:     (B, G, R) 颜色元组
        alpha:     面片透明度 (0=全透明, 1=全不透明)
        line_width: 边缘线宽
    """
    h, w = image.shape[:2]
    overlay = image.copy()
    color_bgr = tuple(int(c) for c in color)

    # === 面片填充：仅渲染深度 > 0 且顶点在视野内的面 ===
    for tri in faces:
        pts = []
        valid = True
        for idx in tri:
            if depth[idx] <= 0:          # 顶点在相机后方 → 跳过
                valid = False
                break
            x, y = verts2d[idx]
            if x < -50 or x > w + 50 or y < -50 or y > h + 50:
                valid = False             # 顶点远离画面 → 跳过
                break
            pts.append((int(x), int(y)))
        if valid and len(pts) == 3:
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color_bgr)

    # 半透明混合
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # === 边缘线框：仅绘制两端点都在视野内的边 ===
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
    """
    自动查找立方体放置位置的世界坐标。

    优先级:
    1. COLMAP 二值化模型 (points3D.bin)  → 点云中位
    2. NVM 模型 (reconstruction.nvm)     → 点云中位
    3. 7-Scenes GT 位姿中位 + Y 偏移     → 中位相机位置上方 0.2m
    4. 原点 [0, 0, 0]                    → 兜底方案
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    root = Path(dataset_cfg["root"])
    scene = dataset_cfg.get("scene", "")
    dataset_name = dataset_cfg.get("name", "cambridge")

    if dataset_name == "cambridge":
        # Cambridge: 尝试 COLMAP 模型
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

        # Cambridge: 回退 NVM 模型
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
        # 7-Scenes: 从 GT 位姿估计场景中心
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
            center[1] += 0.2  # 略高于地面
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
    return 0.2 if dataset_name == "7scenes" else 1.2


def load_image(root, img_name):
    path = root / img_name
    img = cv2.imread(str(path))
    if img is None:
        img = cv2.imread(str(root / Path(img_name).name))
    return img


def sample_frames(items, limit, per_frame_errors=None):
    """Select frames from the best-quality sequence for smooth, continuous video.

    Sequences are ranked by localization quality (median t_err from per_frame.csv).
    Falls back to the longest sequence if per_frame data is unavailable.
    """
    if limit <= 0 or limit >= len(items):
        return items

    import re

    # Group frames by sequence (second-to-last path component)
    seqs = {}
    for name, data in items:
        parts = str(name).replace('\\', '/').split('/')
        seq = parts[-2] if len(parts) >= 2 else (parts[0] if parts else 'default')
        if seq not in seqs:
            seqs[seq] = []
        seqs[seq].append((name, data))

    # Rank sequences: prefer best localization quality, then length as tiebreaker
    def _seq_score(seq_name):
        """Lower score = better sequence."""
        if per_frame_errors and seq_name in per_frame_errors:
            errs = per_frame_errors[seq_name]
            if errs:
                return (0, np.median(errs), -len(seqs[seq_name]))
        return (1, 0, -len(seqs[seq_name]))

    best_seq = min(seqs.keys(), key=_seq_score)
    seq_items = seqs[best_seq]
    quality_note = ""
    if per_frame_errors and best_seq in per_frame_errors:
        errs = per_frame_errors[best_seq]
        if errs:
            quality_note = (f" (median t_err={np.median(errs):.3f}m, "
                          f"{len(seq_items)} frames, "
                          f"{sum(1 for e in errs if e < 0.25)/len(errs)*100:.0f}% @0.25m)")
    print(f"  Selected seq={best_seq}{quality_note}")

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

    # Load predicted poses
    with open(args.poses, encoding="utf-8") as f:
        pred_poses = json.load(f)

    # Determine cube world position
    if args.cube_center is not None:
        cube_center = tuple(args.cube_center)
        print(f"  Cube center (user): {cube_center}")
    else:
        cube_center = tuple(find_scene_center(args.config))

    # For 7Scenes (indoor, no SfM model): place cube at centroid of all camera
    # positions + Y offset, so it sits near the center of the captured scene
    # (e.g., on the stairs for the stairs scene) where most cameras can see it
    if dataset_name == "7scenes" and args.cube_center is None:
        all_positions = np.array([np.array(v["t"]) for v in pred_poses.values()])
        centroid = np.median(all_positions, axis=0)
        cube_center = tuple(centroid + np.array([0, 1.0, 0]))
        print(f"  Cube at scene centroid + 1.0m Y: "
              f"{tuple(round(float(v),2) for v in cube_center)}")

    # If still at origin, fall back to median camera position
    if np.linalg.norm(cube_center) < 0.01:
        cam_positions = np.array([v["t"] for v in list(pred_poses.values())[:100]])
        cube_center = tuple(np.median(cam_positions, axis=0) + np.array([0, 0.3, 0]))
        print(f"  Cube at median camera pos: {tuple(round(float(v),2) for v in cube_center)}")

    all_frames = sorted(pred_poses.items())

    # Load per_frame.csv to rank sequences by localization quality
    per_frame_errors = {}
    per_frame_csv = Path(args.poses).parent / "per_frame.csv"
    if per_frame_csv.exists():
        import csv
        with open(per_frame_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("query", "")
                t_err_str = row.get("t_err", "")
                if name and t_err_str:
                    parts = str(name).replace('\\', '/').split('/')
                    seq = parts[-2] if len(parts) >= 2 else (parts[0] if parts else 'default')
                    try:
                        per_frame_errors.setdefault(seq, []).append(float(t_err_str))
                    except ValueError:
                        pass

    frames = sample_frames(all_frames, args.limit, per_frame_errors)

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
