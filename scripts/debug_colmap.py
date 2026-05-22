"""Debug: trace single query through COLMAP pipeline."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extensions import build_retrieval, build_detector, build_matcher
from scripts.colmap_localization import COLMAPLocalizationModel
from scripts.evaluate import quaternion_angular_error
import cv2
import yaml

def main():
    with open("configs/baseline_a.yaml") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["dataset"]["root"])

    # Load query list
    query_path = root / cfg["dataset"]["query_list"]
    db_path = root / cfg["dataset"]["db_list"]

    query_images = []
    gt_poses = {}
    with open(query_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("ImageFile") or line.startswith("Visual") or not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                name = parts[0]
                query_images.append(name)
                tx, ty, tz = map(float, parts[1:4])
                qw, qx, qy, qz = map(float, parts[4:8])
                gt_poses[name] = (np.array([tx, ty, tz]), np.array([qw, qx, qy, qz]))

    db_images = []
    with open(db_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("ImageFile") or line.startswith("Visual") or not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                db_images.append(parts[0])

    print(f"Queries: {len(query_images)}, DB: {len(db_images)}")

    # Load COLMAP
    colmap_path = root / "colmap_model" / "CambridgeLandmarks_Colmap_Retriangulated_1024px" / "KingsCollege" / "model_train"
    colmap = COLMAPLocalizationModel(str(colmap_path))

    # Build modules
    retrieval = build_retrieval(cfg["retrieval"])
    detector = build_detector(cfg["detector"])
    matcher = build_matcher(cfg["matcher"])

    # Camera at COLMAP resolution
    fx = cfg["dataset"]["fx"]
    fy = cfg["dataset"]["fy"]
    cx = cfg["dataset"]["cx"]
    cy = cfg["dataset"]["cy"]
    K_orig = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    sx = colmap.image_width / (cx * 2)
    sy = colmap.image_height / (cy * 2)
    K = K_orig.copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    print(f"K:\n{K}")

    # Pick first query
    qname = query_images[0]
    print(f"\n=== Query: {qname} ===")
    gt_t, gt_q = gt_poses[qname]

    # Encode query for retrieval
    qimg = cv2.imread(str(root / qname))
    print(f"Query image size: {qimg.shape}")
    q_desc = retrieval.encode(qimg)

    # Encode DB images for retrieval (sample first 100 for speed)
    n_db = min(100, len(db_images))
    db_descs_list = []
    for dbname in db_images[:n_db]:
        dbimg = cv2.imread(str(root / dbname))
        db_descs_list.append(retrieval.encode(dbimg))
    db_descs = np.stack(db_descs_list)

    sims = db_descs @ q_desc
    top_k = 5
    top_idx = np.argsort(sims)[::-1][:top_k]
    print(f"\nTop-{top_k} retrieved:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {db_images[idx]} (sim={sims[idx]:.4f})")

    # Resize query for COLMAP resolution
    cw, ch = colmap.image_width, colmap.image_height
    qimg_rs = cv2.resize(qimg, (cw, ch))
    q_kpts, q_descs, q_scores = detector.detect(qimg_rs)
    print(f"\nQuery keypoints (at {cw}x{ch}): {len(q_kpts)}")
    if len(q_kpts) > 0:
        print(f"  First kpt: ({q_kpts[0][0]:.1f}, {q_kpts[0][1]:.1f})")
        print(f"  Descriptor shape: {q_descs.shape}")

    # Encode top DB image with COLMAP SIFT
    rimg_name = db_images[top_idx[0]]
    print(f"\n--- Top DB image: {rimg_name} ---")

    # Check COLMAP has this image
    img_data = colmap.get_image(rimg_name)
    if img_data is None:
        print("NOT FOUND in COLMAP!")
        # Try with different extension
        for ext_img in [rimg_name, rimg_name.replace('.png', '.jpg'), rimg_name.replace('.jpg', '.png')]:
            d = colmap.get_image(ext_img)
            print(f"  Try '{ext_img}': {'FOUND' if d else 'NOT FOUND'}")
    else:
        print(f"COLMAP has {len(img_data['keypoints'])} keypoints")

    # Full COLMAP SIFT encode for this image
    dbimg = cv2.imread(str(root / rimg_name))
    dbimg_rs = cv2.resize(dbimg, (cw, ch))
    gray = cv2.cvtColor(dbimg_rs, cv2.COLOR_BGR2GRAY)

    colmap_kps, colmap_pids = colmap.get_keypoints_and_3d_ids(rimg_name)
    print(f"COLMAP keypoints: {len(colmap_kps)}")

    if len(colmap_kps) > 0:
        cv_kpts = [cv2.KeyPoint(x, y, size=31.0) for x, y in colmap_kps]
        db_descs_sift = detector.detector.compute(gray, cv_kpts)[1]

        valid = np.array([pid >= 0 and pid in colmap.points3D for pid in colmap_pids])
        print(f"Valid 3D refs: {valid.sum()} / {len(colmap_pids)}")

        if db_descs_sift is not None:
            db_kpts = colmap_kps[valid].astype(np.float32)
            db_descs_sift = db_descs_sift[valid].astype(np.float32)
            db_pids = colmap_pids[valid]
            norms = np.linalg.norm(db_descs_sift, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            db_descs_sift = db_descs_sift / norms
            print(f"DB SIFT descriptors after filtering: {db_descs_sift.shape}")
        else:
            print("SIFT.compute returned None!")
            db_kpts = np.zeros((0, 2), dtype=np.float32)
            db_descs_sift = np.zeros((0, 128), dtype=np.float32)
            db_pids = np.zeros(0, dtype=np.int64)
    else:
        db_kpts = np.zeros((0, 2), dtype=np.float32)
        db_descs_sift = np.zeros((0, 128), dtype=np.float32)
        db_pids = np.zeros(0, dtype=np.int64)

    # Match
    print("\n--- Matching ---")
    query_data = {
        "keypoints": q_kpts,
        "descriptors": q_descs,
        "scores": q_scores,
        "image_size": (cw, ch),
    }
    db_data = {
        "keypoints": db_kpts,
        "descriptors": db_descs_sift,
        "scores": np.ones(len(db_kpts), dtype=np.float32),
        "image_size": (cw, ch),
    }

    q_idx, db_idx, conf = matcher.match(query_data, db_data)
    print(f"Matches: {len(q_idx)}")
    if len(q_idx) > 0:
        print(f"  Confidence: [{conf.min():.4f}, {conf.max():.4f}]")

    # 2D-3D lookup
    print("\n--- 2D-3D Lookup ---")
    colmap_p3d = {pid: v['xyz'] for pid, v in colmap.points3D.items()}
    mkpts2d = []
    mkpts3d = []
    if len(q_idx) >= 4:
        for qi, di in zip(q_idx, db_idx):
            if di < len(db_pids):
                pid = db_pids[di]
                if pid >= 0 and pid in colmap_p3d:
                    mkpts2d.append(q_kpts[qi])
                    mkpts3d.append(colmap_p3d[pid])
    mkpts2d = np.array(mkpts2d) if mkpts2d else np.zeros((0, 2))
    mkpts3d = np.array(mkpts3d) if mkpts3d else np.zeros((0, 3))
    print(f"2D-3D correspondences: {len(mkpts2d)}")

    # PnP
    print("\n--- PnP ---")
    if len(mkpts2d) >= 4:
        try:
            _, R_vec, t_vec, inliers = cv2.solvePnPRansac(
                mkpts3d.astype(np.float64),
                mkpts2d.astype(np.float64),
                K, None,
                reprojectionError=12.0,
                minInliersCount=12,
                confidence=0.9999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if inliers is not None and len(inliers) >= 12:
                R, _ = cv2.Rodrigues(R_vec)
                t = t_vec.flatten()
                print(f"Inliers: {len(inliers)}")
                print(f"Pred t: {t}")
                print(f"GT t: {gt_t}")
                t_err = np.linalg.norm(t - gt_t)
                print(f"Translation error: {t_err:.3f}m")

                trace = np.trace(R)
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    qw_pred = 0.25 / s
                    qx_pred = (R[2, 1] - R[1, 2]) * s
                    qy_pred = (R[0, 2] - R[2, 0]) * s
                    qz_pred = (R[1, 0] - R[0, 1]) * s
                q_pred = np.array([qw_pred, qx_pred, qy_pred, qz_pred])
                q_pred = q_pred / np.linalg.norm(q_pred)
                r_err = quaternion_angular_error(q_pred, gt_q)
                print(f"Rotation error: {r_err:.3f} deg")
            else:
                print(f"Inliers too few: {len(inliers) if inliers is not None else 'None'}")
        except Exception as e:
            print(f"PnP failed: {e}")
    else:
        print("Not enough 2D-3D correspondences")

    print(f"\n=== DIAGNOSIS ===")
    print(f"Query kpts: {len(q_kpts)}")
    print(f"DB kpts (valid COLMAP): {len(db_kpts)}")
    print(f"Matches: {len(q_idx)}")
    print(f"2D-3D matches: {len(mkpts2d)}")

if __name__ == "__main__":
    main()
