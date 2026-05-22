"""Debug script: trace a single query through the full pipeline."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extensions import build_retrieval, build_detector, build_matcher
from scripts.nvm_model import NVMModel
from scripts.evaluate import compute_recall, quaternion_angular_error
import cv2
import yaml

def main():
    with open("configs/baseline_a.yaml") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["dataset"]["root"])
    print(f"Root: {root}")

    # Load datasets
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

    # Build modules
    retrieval = build_retrieval(cfg["retrieval"])
    detector = build_detector(cfg["detector"])
    matcher = build_matcher(cfg["matcher"])

    # Load NVM
    nvm_path = root / cfg["dataset"]["nvm_model"]
    nvm = NVMModel(str(nvm_path))

    # Camera
    fx = cfg["dataset"]["fx"]
    fy = cfg["dataset"]["fy"]
    cx = cfg["dataset"]["cx"]
    cy = cfg["dataset"]["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"K:\n{K}")

    # Pick first query
    qname = query_images[0]
    print(f"\n=== Query: {qname} ===")
    print(f"GT pose: t={gt_poses[qname][0]}, q={gt_poses[qname][1]}")

    qimg = cv2.imread(str(root / qname))
    print(f"Query image size: {qimg.shape}")

    # Step 1: Encode query
    q_desc = retrieval.encode(qimg)
    print(f"Query descriptor shape: {q_desc.shape}, norm: {np.linalg.norm(q_desc):.4f}")

    # Step 2: Encode first few DB images and retrieve
    print("\n--- Retrieval ---")
    n_db_sample = min(50, len(db_images))
    db_descs = []
    db_names_sample = []
    for dbname in db_images[:n_db_sample]:
        dbimg = cv2.imread(str(root / dbname))
        db_descs.append(retrieval.encode(dbimg))
        db_names_sample.append(dbname)
    db_descs = np.stack(db_descs)

    sims = db_descs @ q_desc
    top_k = 5
    top_idx = np.argsort(sims)[::-1][:top_k]
    print(f"Top-{top_k} retrieved:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {db_names_sample[idx]} (sim={sims[idx]:.4f})")

    # Step 3: Detect query keypoints
    print("\n--- Query Keypoints ---")
    q_kpts, q_descs, q_scores = detector.detect(qimg)
    h, w = qimg.shape[:2]
    print(f"Query keypoints: {len(q_kpts)}, descriptors: {q_descs.shape}")

    # Step 4: Match with top retrieved DB image
    print("\n--- Matching ---")
    rimg_name = db_names_sample[top_idx[0]]
    ridx = db_images.index(rimg_name) if rimg_name in db_images else -1
    print(f"Top DB image: {rimg_name} (index in full DB: {ridx})")

    # Check if DB image exists in NVM
    for ext in [rimg_name, rimg_name.replace(".png", ".jpg"), rimg_name.replace(".jpg", ".png")]:
        has = nvm.has_image(ext)
        print(f"  NVM has '{ext}': {has}")

    dbimg = cv2.imread(str(root / rimg_name))
    db_kpts, db_descs, db_scores = detector.detect(dbimg)
    print(f"DB keypoints: {len(db_kpts)}, descriptors: {db_descs.shape}")

    query_data = {
        "keypoints": q_kpts,
        "descriptors": q_descs,
        "scores": q_scores,
        "image_size": (w, h),
    }
    db_data = {
        "keypoints": db_kpts,
        "descriptors": db_descs,
        "scores": db_scores,
        "image_size": (dbimg.shape[1], dbimg.shape[0]),
    }

    q_idx, db_idx, conf = matcher.match(query_data, db_data)
    print(f"Matches: {len(q_idx)}")
    if len(q_idx) > 0:
        print(f"  Confidence range: [{conf.min():.4f}, {conf.max():.4f}]")
        print(f"  First 5 query indices: {q_idx[:5]}")
        print(f"  First 5 DB indices: {db_idx[:5]}")

    # Step 5: 2D-3D lookup
    print("\n--- 2D-3D Lookup ---")
    if len(q_idx) >= 4 and ridx >= 0:
        matched_db_kpts = db_kpts[db_idx]
        print(f"Matched DB keypoints shape: {matched_db_kpts.shape}")
        print(f"Sample matched DB kpts: {matched_db_kpts[:3]}")

        pts3d, valid_3d = nvm.lookup_3d(rimg_name, matched_db_kpts, radius=4.0)
        n_valid = valid_3d.sum()
        print(f"Valid 3D points: {n_valid} / {len(valid_3d)}")

        if n_valid > 0:
            print(f"Sample 3D points: {pts3d[valid_3d][:3]}")
    else:
        print(f"Skipped: too few matches ({len(q_idx)}) or DB image not found")
        pts3d = np.zeros((0, 3))
        valid_3d = np.zeros(0, dtype=bool)

    # Step 6: PnP
    print("\n--- PnP ---")
    all_mkpts2d = []
    all_mkpts3d = []
    if len(q_idx) >= 4:
        for qi, di, vi in zip(q_idx[valid_3d], db_idx[valid_3d], valid_3d[valid_3d]):
            if vi:
                all_mkpts2d.append(q_kpts[qi])
                all_mkpts3d.append(pts3d[valid_3d][np.where(valid_3d)[0]][
                    list(valid_3d[valid_3d]).index(True)
                ])
        # Simpler: use valid mask directly
        all_mkpts2d = q_kpts[q_idx[valid_3d]]
        all_mkpts3d = pts3d[valid_3d]

    print(f"2D-3D correspondences: {len(all_mkpts2d)}")

    if len(all_mkpts2d) >= 4:
        try:
            _, R_vec, t_vec, inliers = cv2.solvePnPRansac(
                all_mkpts3d.astype(np.float64),
                all_mkpts2d.astype(np.float64),
                K, None,
                reprojectionError=12.0,
                minInliersCount=8,
                confidence=0.9999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if inliers is not None:
                print(f"PnP success! Inliers: {len(inliers)}")
                R, _ = cv2.Rodrigues(R_vec)
                t = t_vec.flatten()
                print(f"Pred t: {t}, GT t: {gt_poses[qname][0]}")
                t_err = np.linalg.norm(t - gt_poses[qname][0])
                print(f"Translation error: {t_err:.3f}m")

                # Convert R to quaternion and compute angular error
                trace = np.trace(R)
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    qw = 0.25 / s
                    qx = (R[2, 1] - R[1, 2]) * s
                    qy = (R[0, 2] - R[2, 0]) * s
                    qz = (R[1, 0] - R[0, 1]) * s
                q_pred = np.array([qw, qx, qy, qz])
                q_pred = q_pred / np.linalg.norm(q_pred)
                r_err = quaternion_angular_error(q_pred, gt_poses[qname][1])
                print(f"Rotation error: {r_err:.3f} deg")
            else:
                print("PnP returned no inliers")
        except Exception as e:
            print(f"PnP exception: {e}")
    else:
        print("Not enough 2D-3D correspondences for PnP")

    # Summary
    print("\n=== DIAGNOSIS ===")
    print(f"Matches: {len(q_idx)} (need >= 4)")
    print(f"3D points found: {valid_3d.sum() if len(q_idx) >= 4 else 0}")
    print(f"Cause of failure: ", end="")
    if len(q_idx) < 4:
        print("INSUFFICIENT MATCHES")
    elif (valid_3d.sum() if len(q_idx) >= 4 else 0) < 4:
        print("INSUFFICIENT 2D-3D CORRESPONDENCES (NVM lookup failing)")
    else:
        print("CHECK PNP / CAMERA PARAMETERS")


if __name__ == "__main__":
    main()
