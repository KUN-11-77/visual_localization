import argparse
import yaml
import json
import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from extensions import build_retrieval, build_detector, build_matcher
from scripts.evaluate import compute_recall, quaternion_angular_error
from scripts.timing import timed, _timing_log, dump_timing
from scripts.nvm_model import NVMModel
from scripts.colmap_localization import COLMAPLocalizationModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--overrides", nargs="*", default=[],
                   help="Key=value overrides, e.g. retrieval.top_k=20")
    p.add_argument("--limit_queries", type=int, default=0,
                   help="Limit number of query images (0=all)")
    p.add_argument("--spatial_radius", type=float, default=4.0,
                   help="Max pixel distance for 2D->3D lookup")
    p.add_argument("--build_sfm", action="store_true",
                   help="Build SP+SG SfM model via pycolmap triangulation")
    p.add_argument("--num_covis", type=int, default=20,
                   help="Number of covisible pairs per DB image for SfM")
    return p.parse_args()


def load_dataset(cfg):
    """Load dataset based on dataset type."""
    if cfg["dataset"]["name"] == "7scenes":
        return _load_7scenes(cfg)
    else:
        return _load_cambridge(cfg)


def _load_cambridge(cfg):
    """Load Cambridge dataset with NVM model for 2D-3D lookup."""
    dataset_cfg = cfg["dataset"]
    root = Path(dataset_cfg["root"])

    # Load query and database image lists
    query_path = root / dataset_cfg["query_list"]
    db_path = root / dataset_cfg["db_list"]

    # Skip header lines (first 2 lines in Cambridge format)
    query_images = []
    with open(query_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ImageFile") or line.startswith("Visual") or not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                query_images.append(parts[0])

    db_images = []
    with open(db_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ImageFile") or line.startswith("Visual") or not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                db_images.append(parts[0])

    # Load ground truth poses for queries
    gt_poses = {}
    for line in open(query_path):
        line = line.strip()
        if line.startswith("ImageFile") or line.startswith("Visual") or not line:
            continue
        parts = line.split()
        name = parts[0]
        tx, ty, tz = map(float, parts[1:4])
        # Cambridge format: X Y Z W P Q R = tx ty tz qw qx qy qz
        qw, qx, qy, qz = map(float, parts[4:8])
        gt_poses[name] = (np.array([tx, ty, tz]), np.array([qw, qx, qy, qz]))

    # Load 3D model for 2D-3D lookup (prefer COLMAP binary over text, then NVM)
    colmap_model = None
    nvm_model = None

    # Search for COLMAP model in multiple locations
    scene = dataset_cfg.get("scene", "KingsCollege")
    colmap_candidates = [
        root / "colmap_model" / "CambridgeLandmarks_Colmap_Retriangulated_1024px"
        / scene / "model_train",
        root / "colmap_reconstruction",
        root / "colmap_model",
    ]
    colmap_path = None
    for cand in colmap_candidates:
        if cand.exists() and ((cand / "images.bin").exists() or (cand / "images.txt").exists()):
            colmap_path = cand
            break

    if colmap_path is not None:
        print(f"  Loading COLMAP model from: {colmap_path}")
        colmap_model = COLMAPLocalizationModel(str(colmap_path))
    elif "nvm_model" in dataset_cfg:
        nvm_path = root / dataset_cfg["nvm_model"]
        if nvm_path.exists():
            nvm_model = NVMModel(str(nvm_path))

    # Build camera intrinsics
    K = _build_camera_matrix(dataset_cfg)

    return query_images, db_images, gt_poses, root, dataset_cfg, colmap_model, nvm_model, K, None


def _load_7scenes(cfg):
    """Load 7-Scenes dataset."""
    dataset_cfg = cfg["dataset"]
    root = Path(dataset_cfg["root"])
    scene = dataset_cfg["scene"]

    scene_root = root / scene

    # Load train/test split files
    train_seqs = []
    test_seqs = []
    with open(scene_root / "TrainSplit.txt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sequence"):
                train_seqs.append(line)
    with open(scene_root / "TestSplit.txt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sequence"):
                test_seqs.append(line)

    def _find_sequence_dir(seq_name):
        """Resolve sequence directory. Handles 'sequence1' -> 'seq-01' mapping."""
        d = scene_root / seq_name
        if d.exists():
            return d
        # Try 'seq-XX' format: sequence1 -> seq-01
        if seq_name.startswith("sequence"):
            num = seq_name[len("sequence"):]
            alt = scene_root / f"seq-{int(num):02d}"
            if alt.exists():
                return alt
        return None

    def _find_color_files(seq_dir):
        """Find all .color.png files, handling both flat and nested structures."""
        files = sorted(seq_dir.glob("frame-*.color.png"))
        if files:
            return files, seq_dir
        # Nested: seq-01/seq-01/frame-*.color.png
        for child in seq_dir.iterdir():
            if child.is_dir():
                nested = sorted(child.glob("frame-*.color.png"))
                if nested:
                    return nested, child
        return [], seq_dir

    # Collect database images from training sequences
    db_images = []
    db_pose_files = {}
    for seq_name in train_seqs:
        seq_dir = _find_sequence_dir(seq_name)
        if seq_dir is None:
            continue
        files, actual_dir = _find_color_files(seq_dir)
        for f in files:
            rel_path = str(f.relative_to(root))
            db_images.append(rel_path)
            db_pose_files[rel_path] = actual_dir / f"{f.name.replace('.color.png', '.pose.txt')}"

    # Collect query images from test sequences
    query_images = []
    gt_poses = {}
    for seq_name in test_seqs:
        seq_dir = _find_sequence_dir(seq_name)
        if seq_dir is None:
            continue
        files, actual_dir = _find_color_files(seq_dir)
        for f in files:
            rel_path = str(f.relative_to(root))
            query_images.append(rel_path)
            pose_file = actual_dir / f"{f.name.replace('.color.png', '.pose.txt')}"
            if pose_file.exists():
                pose = np.loadtxt(pose_file)
                R_c2w = pose[:3, :3]
                t_c2w = pose[:3, 3]
                # Store camera center in world + world-to-camera quaternion
                # (same convention as Cambridge GT)
                R_w2c = R_c2w.T
                q_w2c = _rotmat_to_quaternion(R_w2c)
                gt_poses[rel_path] = (t_c2w, q_w2c)

    K = _build_camera_matrix(dataset_cfg)
    return query_images, db_images, gt_poses, root, dataset_cfg, None, None, K, db_pose_files


def _build_camera_matrix(dataset_cfg):
    """Build camera intrinsic matrix."""
    if "camera_matrix" in dataset_cfg:
        K = np.array(dataset_cfg["camera_matrix"], dtype=np.float64)
    else:
        # Default: Cambridge 1920x1080 with focal ~1670
        fx = dataset_cfg.get("fx", 1670.0)
        fy = dataset_cfg.get("fy", 1670.0)
        cx = dataset_cfg.get("cx", 960.0)
        cy = dataset_cfg.get("cy", 540.0)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


@timed("retrieval")
def run_retrieval(query_desc, db_descs, db_names, top_k):
    sims = db_descs @ query_desc
    indices = np.argsort(sims)[::-1][:top_k]
    return [db_names[i] for i in indices]


@timed("matching")
def run_matching(query_data, db_data, matcher):
    return matcher.match(query_data, db_data)


@timed("pose_estimation")
def run_pnp(mkpts2d, mkpts3d, camera_matrix):
    import cv2
    if len(mkpts2d) < 4:
        return None, None, 0
    try:
        _, R_vec, t_vec, inliers = cv2.solvePnPRansac(
            mkpts3d.astype(np.float64),
            mkpts2d.astype(np.float64),
            camera_matrix,
            None,
            reprojectionError=12.0,
            confidence=0.9999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if inliers is None or len(inliers) < 8:
            return None, None, 0
        R, _ = cv2.Rodrigues(R_vec)
        t = t_vec.flatten()
        return R, t, len(inliers)
    except Exception:
        return None, None, 0


def _nvm_sift_encode(image, img_name, nvm_model, detector,
                     nvm_width=None, nvm_height=None):
    """Extract SIFT descriptors at NVM keypoint positions for direct 2D-3D mapping.

    Resizes the image to NVM reconstruction resolution if provided, so keypoint
    coordinates from the NVM model match pixel positions exactly.
    """
    import cv2
    h_orig, w_orig = image.shape[:2]

    if nvm_width is not None and nvm_height is not None:
        image = cv2.resize(image, (nvm_width, nvm_height))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Look up NVM keypoints for this image (try .png and .jpg)
    nvm_kps = None
    for name in (img_name,
                 img_name.replace('.png', '.jpg'),
                 img_name.replace('.jpg', '.png')):
        if name in nvm_model.image_keypoints:
            nvm_kps = nvm_model.image_keypoints[name]
            break

    if nvm_kps is None or len(nvm_kps) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 128), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )

    nvm_xy = nvm_kps[:, :2]
    nvm_pids = nvm_kps[:, 2].astype(np.int64)

    cv_kpts = [cv2.KeyPoint(x, y, size=31.0) for x, y in nvm_xy]
    descriptors = detector.detector.compute(gray, cv_kpts)[1]

    if descriptors is None or len(descriptors) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 128), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )

    valid_mask = np.array([pid >= 0 and pid in nvm_model.point3D_xyz
                           for pid in nvm_pids])
    keypoints = nvm_xy[valid_mask].astype(np.float32)
    descriptors = descriptors[valid_mask].astype(np.float32)
    p3d_ids = nvm_pids[valid_mask]

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descriptors = descriptors / norms

    return keypoints, descriptors, p3d_ids


def _colmap_sift_encode(image, img_name, colmap_model, detector):
    """Extract SIFT descriptors at COLMAP keypoint positions for direct 2D-3D mapping.

    Resizes the image to COLMAP reconstruction resolution (1024x576) so keypoint
    coordinates from the model match pixel positions exactly.
    """
    import cv2
    cw, ch = colmap_model.image_width, colmap_model.image_height
    if cw is None or ch is None:
        cw, ch = 1024, 576

    image_resized = cv2.resize(image, (cw, ch))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Look up COLMAP keypoints for this image
    colmap_kps, colmap_pids = colmap_model.get_keypoints_and_3d_ids(img_name)
    if len(colmap_kps) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 128), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )

    cv_kpts = [cv2.KeyPoint(x, y, size=31.0) for x, y in colmap_kps]
    descriptors = detector.detector.compute(gray, cv_kpts)[1]

    if descriptors is None or len(descriptors) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 128), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )

    # Only keep keypoints with valid 3D points known to the model
    valid_mask = np.array([pid >= 0 and pid in colmap_model.points3D
                           for pid in colmap_pids])
    keypoints = colmap_kps[valid_mask].astype(np.float32)
    descriptors = descriptors[valid_mask].astype(np.float32)
    p3d_ids = colmap_pids[valid_mask]

    # L2-normalize
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    descriptors = descriptors / norms

    return keypoints, descriptors, p3d_ids


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for kv in args.overrides:
        key, val = kv.split("=", 1)
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = yaml.safe_load(val)

    print(f"Initializing: {cfg['name']}")
    print(f"  {cfg['description']}")

    retrieval = build_retrieval(cfg["retrieval"])
    detector = build_detector(cfg["detector"])
    matcher = build_matcher(cfg["matcher"])

    result = load_dataset(cfg)
    query_images, db_images, gt_poses, root, dataset_cfg, colmap_model, nvm_model, K, db_pose_files = result

    n_query = len(query_images)
    if args.limit_queries > 0:
        query_images = query_images[:args.limit_queries]
        n_query = len(query_images)

    print(f"Dataset: {cfg['dataset']['name']} / {cfg['dataset']['scene']}")
    print(f"  DB images: {len(db_images)}, Query images: {n_query}")
    print(f"  COLMAP model: {colmap_model is not None}")
    print(f"  NVM model: {nvm_model is not None}")
    print(f"  Camera K:\n{K}")

    # Determine 2D-3D strategy
    use_colmap_sift = (colmap_model is not None)
    use_nvm_sift = (nvm_model is not None and colmap_model is None and
                    cfg["detector"]["method"] == "SIFTDetector")
    use_7scenes_depth = (db_pose_files is not None)
    depth_p3d_xyz = {}  # point3D_id -> xyz (world) for 7scenes depth-based 3D
    depth_next_pid = [0]  # mutable counter for unique point3D IDs

    # NVM reconstruction resolution (may be available from config or colmap_reconstruction)
    nvm_width = dataset_cfg.get("nvm_width", None)
    nvm_height = dataset_cfg.get("nvm_height", None)

    # If using COLMAP, work at reconstruction resolution
    if use_colmap_sift:
        colmap_w = colmap_model.image_width or 1024
        colmap_h = colmap_model.image_height or 576
        # Scale K to COLMAP resolution
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]
        # Assume K was built for original image size; compute original size
        orig_w = int(orig_cx * 2)  # 960*2 = 1920
        orig_h = int(orig_cy * 2)  # 540*2 = 1080
        sx = colmap_w / orig_w
        sy = colmap_h / orig_h
        K_colmap = K.copy()
        K_colmap[0, 0] *= sx
        K_colmap[0, 2] *= sx
        K_colmap[1, 1] *= sy
        K_colmap[1, 2] *= sy
        K = K_colmap
        print(f"  COLMAP resolution: {colmap_w}x{colmap_h}, K:\n{K}")

    if use_nvm_sift and nvm_width is not None and nvm_height is not None:
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]
        orig_w = int(orig_cx * 2)
        orig_h = int(orig_cy * 2)
        sx = nvm_width / orig_w
        sy = nvm_height / orig_h
        K_nvm = K.copy()
        K_nvm[0, 0] *= sx
        K_nvm[0, 2] *= sx
        K_nvm[1, 1] *= sy
        K_nvm[1, 2] *= sy
        K = K_nvm
        print(f"  NVM resolution: {nvm_width}x{nvm_height}, K:\n{K}")

    # Encode all database images
    print("Encoding database images...")
    if use_colmap_sift:
        print("  Using detector + COLMAP spatial 2D-3D lookup")
    elif use_nvm_sift:
        print("  Using NVM keypoints + SIFT descriptors (direct 2D-3D mapping)")
    elif use_7scenes_depth:
        print("  Using depth-based 2D-3D mapping (7Scenes RGB-D)")
    db_descs = []
    db_features = []
    db_img_paths = []
    # Pre-load COLMAP keypoints for spatial lookup (per image)
    colmap_kp_index = None
    if use_colmap_sift:
        colmap_kp_index = {}
        for img_name in db_images:
            ckps, cpids = colmap_model.get_keypoints_and_3d_ids(img_name)
            if len(ckps) > 0:
                colmap_kp_index[img_name] = (ckps, cpids)

    for img_name in tqdm(db_images, desc="  DB encode"):
        img_path = root / img_name
        image = _load_image(img_path)
        desc = retrieval.encode(image)

        if use_colmap_sift:
            # Resize to COLMAP resolution and detect SIFT normally
            image_resized = cv2.resize(image,
                                       (colmap_model.image_width, colmap_model.image_height))
            kpts, feats, scores = detector.detect(image_resized)
            db_descs.append(desc)
            db_features.append({
                "keypoints": kpts,
                "descriptors": feats,
                "scores": scores,
                "image_size": (colmap_model.image_width, colmap_model.image_height),
                "name": img_name,
            })
        elif use_nvm_sift:
            kpts, sift_descs, p3d_ids = _nvm_sift_encode(
                image, img_name, nvm_model, detector,
                nvm_width=nvm_width, nvm_height=nvm_height,
            )
            h, w = image.shape[:2]
            db_descs.append(desc)
            db_features.append({
                "keypoints": kpts,
                "descriptors": sift_descs,
                "scores": np.ones(len(kpts), dtype=np.float32),
                "image_size": (w, h),
                "name": img_name,
                "point3D_ids": p3d_ids,
            })
        elif use_7scenes_depth:
            kpts, feats, scores = detector.detect(image)
            h, w = image.shape[:2]
            # Load depth map
            depth_path = img_path.parent / f"{img_path.stem.replace('.color', '')}.depth.png"
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                pose = np.loadtxt(db_pose_files[img_name])
                R_c2w = pose[:3, :3]
                t_c2w = pose[:3, 3]
                valid, pts3d_world = _depth_unproject(kpts, depth_img, K, R_c2w, t_c2w)
                # Assign unique point3D IDs
                n_valid = int(np.sum(valid))
                p3d_ids = np.full(len(kpts), -1, dtype=np.int64)
                if n_valid > 0:
                    start_id = depth_next_pid[0]
                    p3d_ids_for_valid = np.arange(start_id, start_id + n_valid, dtype=np.int64)
                    p3d_ids[valid] = p3d_ids_for_valid
                    for i, pid in enumerate(p3d_ids_for_valid):
                        depth_p3d_xyz[int(pid)] = pts3d_world[i]
                    depth_next_pid[0] = start_id + n_valid
            else:
                p3d_ids = np.full(len(kpts), -1, dtype=np.int64)
            db_descs.append(desc)
            db_features.append({
                "keypoints": kpts,
                "descriptors": feats,
                "scores": scores,
                "image_size": (w, h),
                "name": img_name,
                "point3D_ids": p3d_ids,
            })
        else:
            kpts, feats, scores = detector.detect(image)
            h, w = image.shape[:2]
            db_descs.append(desc)
            db_features.append({
                "keypoints": kpts,
                "descriptors": feats,
                "scores": scores,
                "image_size": (w, h),
                "name": img_name,
            })
        db_img_paths.append(img_path)

    db_descs = np.stack(db_descs)
    db_names = db_images

    # Build SuperPoint+SuperGlue SfM model via pycolmap triangulation
    sp_p3d_xyz = None
    if args.build_sfm and colmap_model is not None and len(db_features) > 0:
        from scripts.build_sfm import build_superpoint_sfm

        sfm_output_dir = Path(cfg["output"]["results_dir"]) / "sfm_model"
        db_image_root = str(root)
        print(f"\n=== Building SP+SG SfM model (pycolmap triangulation) ===")
        lookup, reconstruction, sfm_dir = build_superpoint_sfm(
            colmap_model, db_features, db_images, db_image_root,
            sfm_output_dir, num_covis=args.num_covis, matcher=matcher,
        )

        # Build point3D_id -> xyz lookup from triangulated model
        sp_p3d_xyz = {}
        for pid, p3d in reconstruction.points3D.items():
            sp_p3d_xyz[pid] = p3d.xyz

        # Attach point3D_ids to each db_features entry
        for idx, img_name in enumerate(db_images):
            kp_count = len(db_features[idx]["keypoints"])
            p3d_ids = np.full(kp_count, -1, dtype=np.int64)
            for kp_idx in range(kp_count):
                pid = lookup.get((img_name, kp_idx))
                if pid is not None:
                    p3d_ids[kp_idx] = int(pid)
            db_features[idx]["point3D_ids"] = p3d_ids

        print(f"=== SfM model built: {len(sp_p3d_xyz)} 3D points ===\n")

    # Pre-build COLMAP 3D point lookup dict (for spatial lookup fallback)
    colmap_p3d_xyz = None
    if use_colmap_sift and colmap_model is not None:
        colmap_p3d_xyz = {pid: v['xyz'] for pid, v in colmap_model.points3D.items()}

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_poses = {}
    timing_records = []

    top_k = cfg["retrieval"]["top_k"]

    print(f"Localizing {n_query} queries (top_k={top_k})...")
    per_frame = []
    for qimg_name in tqdm(query_images, desc="  Localizing"):
        qimg_path = root / qimg_name
        query_image = _load_image(qimg_path)

        # 1. Retrieve
        query_desc = retrieval.encode(query_image)
        retrieved = run_retrieval(query_desc, db_descs, db_names, top_k)

        # 2. Detect query keypoints (resize for COLMAP/NVM if needed)
        if use_colmap_sift:
            query_image_resized = cv2.resize(query_image,
                                             (colmap_model.image_width, colmap_model.image_height))
            q_kpts, q_descs, q_scores = detector.detect(query_image_resized)
            q_img_w, q_img_h = colmap_model.image_width, colmap_model.image_height
        elif use_nvm_sift and nvm_width is not None and nvm_height is not None:
            query_image_resized = cv2.resize(query_image, (nvm_width, nvm_height))
            q_kpts, q_descs, q_scores = detector.detect(query_image_resized)
            q_img_w, q_img_h = nvm_width, nvm_height
        else:
            q_kpts, q_descs, q_scores = detector.detect(query_image)
            q_img_w, q_img_h = query_image.shape[1], query_image.shape[0]
        query_data = {
            "keypoints": q_kpts,
            "descriptors": q_descs,
            "scores": q_scores,
            "image_size": (q_img_w, q_img_h),
        }

        all_mkpts2d = []
        all_mkpts3d = []

        # 3. Match against each retrieved database image
        for rimg_name in retrieved:
            if rimg_name not in db_names:
                continue
            ridx = db_names.index(rimg_name)
            db_data = db_features[ridx]
            q_idx, db_idx, conf = matcher.match(query_data, db_data)

            if len(q_idx) < 4:
                continue

            db_kpts = db_data["keypoints"]

            # 2D-3D lookup
            if "point3D_ids" in db_data:
                # Direct mapping: matched DB keypoint index -> known 3D point ID
                # (NVM SIFT or triangulated SP+SG model)
                p3d_ids = db_data["point3D_ids"]
                if sp_p3d_xyz is not None:
                    p3d_xyz = sp_p3d_xyz
                elif use_nvm_sift and nvm_model is not None:
                    p3d_xyz = nvm_model.point3D_xyz
                elif use_7scenes_depth and len(depth_p3d_xyz) > 0:
                    p3d_xyz = depth_p3d_xyz
                else:
                    p3d_xyz = {}
                for qi, di in zip(q_idx, db_idx):
                    if di < len(p3d_ids):
                        pid = p3d_ids[di]
                        if pid >= 0 and pid in p3d_xyz:
                            all_mkpts2d.append(q_kpts[qi])
                            all_mkpts3d.append(p3d_xyz[pid])
            elif use_colmap_sift and colmap_kp_index is not None:
                # Spatial proximity: matched DB keypoint → nearest COLMAP keypoint → 3D point
                matched_db_kpts = db_kpts[db_idx]
                ckps, cpids = colmap_kp_index.get(rimg_name, (None, None))
                if ckps is not None and len(ckps) > 0:
                    radius = max(args.spatial_radius, 8.0)
                    dists = np.sqrt(
                        np.sum((ckps[np.newaxis, :, :] - matched_db_kpts[:, np.newaxis, :]) ** 2, axis=2)
                    )
                    best_j = np.argmin(dists, axis=1)
                    best_dist = np.min(dists, axis=1)
                    for i, (qi, di) in enumerate(zip(q_idx, db_idx)):
                        if best_dist[i] < radius:
                            pid = cpids[best_j[i]]
                            if pid >= 0 and pid in colmap_p3d_xyz:
                                all_mkpts2d.append(q_kpts[qi])
                                all_mkpts3d.append(colmap_p3d_xyz[pid])
            elif nvm_model is not None:
                # Fallback: spatial proximity to NVM keypoints
                matched_db_kpts = db_kpts[db_idx]
                pts3d, valid_3d = nvm_model.lookup_3d(
                    rimg_name, matched_db_kpts, radius=max(args.spatial_radius, 15.0)
                )
                for i, (qi, di) in enumerate(zip(q_idx, db_idx)):
                    if valid_3d[i]:
                        all_mkpts2d.append(q_kpts[qi])
                        all_mkpts3d.append(pts3d[i])
            else:
                # No 3D model: pseudo-3D (planar)
                for qi, di in zip(q_idx, db_idx):
                    all_mkpts2d.append(q_kpts[qi])
                    x, y = db_kpts[di]
                    all_mkpts3d.append(np.array([x, y, 0.0], dtype=np.float64))

        n_corr = len(all_mkpts2d)
        all_mkpts2d = np.array(all_mkpts2d) if all_mkpts2d else np.zeros((0, 2))
        all_mkpts3d = np.array(all_mkpts3d) if all_mkpts3d else np.zeros((0, 3))

        if len(all_mkpts2d) >= 4:
            R, t, n_inliers = run_pnp(all_mkpts2d, all_mkpts3d, K)
        else:
            R, t, n_inliers = None, None, 0

        # Compute errors if localized
        t_err = None
        r_err = None
        localized = False
        if R is not None and t is not None:
            # PnP returns world-to-camera: X_cam = R @ X_world + t
            # GT format: camera center C in world, world-to-camera quaternion q_w2c
            # Convert: camera center C = -R.T @ t, quaternion stays q_w2c = quat(R)
            R_c2w = R.T
            t_c2w = -R_c2w @ t
            q_pred = _rotmat_to_quaternion(R)  # world-to-camera, matches GT convention
            pred_poses[qimg_name] = (t_c2w, q_pred)
            if qimg_name in gt_poses:
                t_gt, q_gt = gt_poses[qimg_name]
                t_err = float(np.linalg.norm(t_c2w - t_gt))
                r_err = float(quaternion_angular_error(q_pred, q_gt))
                localized = t_err < 0.25 and r_err < 2.0
        # Per-frame record
        frame_info = {
            "query": qimg_name,
            "n_query_kpts": len(q_kpts),
            "n_correspondences": n_corr,
            "n_inliers": int(n_inliers or 0),
            "retrieved_top1": retrieved[0] if len(retrieved) > 0 else "",
            "retrieved_top5": ";".join(retrieved[:5]) if len(retrieved) > 0 else "",
            "t_err": t_err if t_err is not None else "",
            "r_err": r_err if r_err is not None else "",
            "localized_0.25m_2deg": localized,
        }
        per_frame.append(frame_info)

    # Evaluate
    recall = compute_recall(pred_poses, gt_poses)

    print("\n=== Results ===")
    for k in sorted(recall.keys()):
        v = recall[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save timing
    timing_path = results_dir / "timing.json"
    dump_timing(str(timing_path))
    print(f"\nTiming saved to {timing_path}")

    # Save per-frame log
    frames_path = results_dir / "per_frame.csv"
    with open(frames_path, "w", newline="") as f:
        fieldnames = [
            "query", "n_query_kpts", "n_correspondences", "n_inliers",
            "retrieved_top1", "retrieved_top5", "t_err", "r_err",
            "localized_0.25m_2deg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_frame:
            writer.writerow(row)
    print(f"Per-frame log saved to {frames_path}")

    # Save per-experiment results
    csv_path = results_dir / "results.csv"
    recall_items = sorted([k for k in recall.keys() if k.startswith("(")])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "dataset", "scene"] + recall_items + ["n_query", "n_localized"])
        writer.writerow([
            cfg["name"], cfg["dataset"]["name"], cfg["dataset"]["scene"],
            *[recall[k] for k in recall_items],
            recall.get("n_query", 0),
            recall.get("n_localized", 0),
        ])

    # Append to summary
    summary_path = Path("outputs/results/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["experiment", "dataset", "scene"] + recall_items + ["n_query", "n_localized"]
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if summary_path.stat().st_size == 0:
            writer.writerow(header)
        writer.writerow([
            cfg["name"], cfg["dataset"]["name"], cfg["dataset"]["scene"],
            *[recall[k] for k in recall_items],
            recall.get("n_query", 0),
            recall.get("n_localized", 0),
        ])

    print(f"Results saved to {csv_path}")
    print(f"Summary appended to {summary_path}")


def _load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def _depth_unproject(kpts, depth_img, K, R_c2w, t_c2w):
    """Unproject keypoints to 3D world coordinates using depth map (7Scenes).

    Args:
        kpts: (N, 2) float32 array of (x, y) pixel coordinates
        depth_img: (H, W) uint16 depth map in millimeters
        K: (3, 3) camera intrinsic matrix
        R_c2w: (3, 3) camera-to-world rotation
        t_c2w: (3,) camera-to-world translation

    Returns:
        valid: (N,) bool mask indicating which keypoints have valid depth
        pts3d_world: (M, 3) float32 world coordinates for valid keypoints
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.round(kpts[:, 0]).astype(np.int32)
    v = np.round(kpts[:, 1]).astype(np.int32)

    h, w = depth_img.shape
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    depth_mm = depth_img[v, u].astype(np.float32)
    valid = (depth_mm > 0) & (depth_mm < 65500)

    if not np.any(valid):
        return valid, np.zeros((0, 3), dtype=np.float32)

    z = depth_mm[valid] / 1000.0
    x = (u[valid].astype(np.float32) - cx) * z / fx
    y = (v[valid].astype(np.float32) - cy) * z / fy

    X_cam = np.stack([x, y, z], axis=-1)
    X_world = (R_c2w @ X_cam.T).T + t_c2w
    return valid, X_world.astype(np.float32)


def _rotmat_to_quaternion(R):
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


if __name__ == "__main__":
    main()
