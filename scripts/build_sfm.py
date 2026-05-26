"""
Build SuperPoint+SuperGlue SfM model from a reference COLMAP (SIFT) model.

Pipeline:
  1. Generate covisibility pairs from the reference SIFT model
  2. Match DB image pairs with the configured matcher (e.g. SuperGlue)
  3. Create COLMAP database, import features + verified matches
  4. Triangulate new 3D points using known camera poses (pycolmap)
  5. Return (image_name, kp_idx) -> point3D_id lookup dict
"""

import os
import sys
import time
import h5py
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import pycolmap


def generate_covisibility_pairs(colmap_model, db_images, output_path,
                                 num_matched=20):
    """Generate DB image pairs from COLMAP covisibility graph.

    Args:
        colmap_model: ColmapLocalization instance
        db_images: list of image names in COLMAP model
        output_path: path to write pair file
        num_matched: max covisible images per image

    Returns:
        number of unique pairs
    """
    print(f"Generating covisibility pairs (num_matched={num_matched})...")

    # Build reverse index: point3D_id -> set of image names that observe it
    print("  Building point3D->images index...")
    pt3d_to_images = defaultdict(set)
    for img_name in tqdm(db_images):
        img_data = colmap_model.get_image(img_name)
        if img_data is None:
            continue
        pids = img_data.get("point3D_ids", [])
        for pid in pids:
            if pid >= 0:
                pt3d_to_images[pid].add(img_name)

    # For each image, find covisible images via shared 3D points
    print("  Finding covisible pairs...")
    pairs = []
    for img_name in tqdm(db_images):
        img_data = colmap_model.get_image(img_name)
        if img_data is None:
            continue
        pids = img_data.get("point3D_ids", [])
        valid_pids = [pid for pid in pids if pid >= 0]
        if len(valid_pids) == 0:
            continue

        covis = defaultdict(int)
        for pid in valid_pids:
            for covis_name in pt3d_to_images.get(pid, set()):
                if covis_name != img_name:
                    covis[covis_name] += 1

        if len(covis) == 0:
            continue

        covis_names = np.array(list(covis.keys()))
        covis_nums = np.array([covis[n] for n in covis_names])
        top_k = min(num_matched, len(covis_names))
        if top_k == len(covis_names):
            top_idx = np.argsort(-covis_nums)
        else:
            ind_top = np.argpartition(covis_nums, -top_k)[-top_k:]
            top_idx = ind_top[np.argsort(-covis_nums[ind_top])]

        for idx in top_idx:
            pairs.append((img_name, covis_names[idx]))

    # Deduplicate (A,B) vs (B,A)
    seen = set()
    unique_pairs = []
    for a, b in pairs:
        key = tuple(sorted([a, b]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((a, b))

    with open(output_path, "w") as f:
        f.write("\n".join(f"{a} {b}" for a, b in unique_pairs))

    print(f"  Generated {len(unique_pairs)} unique pairs from {len(pairs)} raw")
    return len(unique_pairs)


def save_features_h5(db_features, db_images, output_path):
    """Save DB features to HLoc-compatible HDF5 format.

    Args:
        db_features: list of dicts with "keypoints", "descriptors", "scores"
        db_images: list of image names (same order)
        output_path: path to output .h5 file
    """
    print(f"Writing features to {output_path}...")
    with h5py.File(str(output_path), "w", libver="latest") as f:
        for img_name, feat in tqdm(zip(db_images, db_features), total=len(db_images)):
            grp = f.create_group(img_name)
            kpts = np.array(feat["keypoints"])
            descs = np.array(feat["descriptors"])
            scores = np.array(feat.get("scores", np.ones(len(kpts), dtype=np.float32)))

            # HLoc format: keypoints (N,2), descriptors (D,N), scores (N,)
            grp.create_dataset("keypoints", data=kpts.astype(np.float64))
            grp.create_dataset("descriptors", data=descs.T.astype(np.float32))
            grp.create_dataset("scores", data=scores.astype(np.float32))
    print(f"  Saved features for {len(db_images)} images")


def match_db_pairs(pairs_path, db_features, db_images, matcher, output_path):
    """Match DB image pairs using the configured matcher.

    Args:
        pairs_path: path to pair file (image0 image1 per line)
        db_features: list of dicts with keypoints, descriptors, scores, image_size
        db_images: list of image names
        matcher: matcher instance (e.g. SuperGlueMatcher)
        output_path: path to output .h5 file
    """
    with open(pairs_path, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    print(f"Matching {len(pairs)} DB pairs...")
    name_to_idx = {name: i for i, name in enumerate(db_images)}

    with h5py.File(str(output_path), "w", libver="latest") as f:
        for name0, name1 in tqdm(pairs):
            pair_key = f"{name0}/{name1}"

            idx0 = name_to_idx[name0]
            idx1 = name_to_idx[name1]
            q_idx, db_idx, conf = matcher.match(db_features[idx0], db_features[idx1])

            n_matches = len(q_idx)
            matches0 = -np.ones(len(db_features[idx0]["keypoints"]), dtype=np.int64)
            matching_scores0 = np.zeros(len(db_features[idx0]["keypoints"]), dtype=np.float32)
            matches0[q_idx] = db_idx
            matching_scores0[q_idx] = conf

            grp = f.create_group(pair_key)
            grp.create_dataset("matches0", data=matches0)
            grp.create_dataset("matching_scores0", data=matching_scores0)

            grp.attrs["n_matches"] = n_matches
            grp.attrs["n_kps0"] = len(db_features[idx0]["keypoints"])
            grp.attrs["n_kps1"] = len(db_features[idx1]["keypoints"])

    print(f"  Saved {len(pairs)} pair match results")


def build_superpoint_sfm(colmap_model, db_features, db_images, db_image_root,
                          output_dir, pairs_path=None, num_covis=20,
                          matcher=None):
    """Build new SuperPoint-based SfM model via pycolmap triangulation.

    Args:
        colmap_model: ColmapLocalization instance (reference SIFT model)
        db_features: list of dicts with keypoints, descriptors, scores, image_size
        db_images: list of image names
        db_image_root: path to directory containing DB images
        output_dir: path for output files
        pairs_path: optional existing pair file (generated if None)
        num_covis: number of covisible pairs per image
        matcher: matcher instance for DB pair matching

    Returns:
        dict: {(image_name, kp_idx): point3D_id} lookup
        pycolmap.Reconstruction: the triangulated model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # 1. Generate covisibility pairs
    if pairs_path is None:
        pairs_path = output_dir / "db_covis_pairs.txt"
    if not os.path.exists(pairs_path):
        generate_covisibility_pairs(colmap_model, db_images, pairs_path,
                                     num_matched=num_covis)

    # 2. Save features to HLoc h5 format
    features_path = output_dir / "features_superpoint.h5"
    if not os.path.exists(features_path):
        save_features_h5(db_features, db_images, features_path)

    # 3. Match DB pairs
    matches_path = output_dir / "matches_superglue.h5"
    if matcher is not None and not os.path.exists(matches_path):
        match_db_pairs(pairs_path, db_features, db_images, matcher, matches_path)

    # 4. Create COLMAP database from reference model
    db_path = output_dir / "database.db"
    print(f"Creating COLMAP database at {db_path}...")
    reference = pycolmap.Reconstruction(str(colmap_model.model_path))

    image_ids = {}
    if db_path.exists():
        db_path.unlink()

    with pycolmap.Database.open(db_path) as db:
        # Write cameras
        for camera_id, camera in reference.cameras.items():
            db.write_camera(camera, use_camera_id=True)

        # Write minimal rigs (required by frames; accessing reference.rigs segfaults)
        for frame_id, frame in reference.frames.items():
            rig = pycolmap.Rig()
            rig.rig_id = frame.rig_id
            data_ids = frame.data_ids_by_sensor(pycolmap.SensorType.CAMERA)
            if data_ids:
                rig.add_ref_sensor(data_ids[0].sensor_id)
            db.write_rig(rig, use_rig_id=True)

        # Write frames (now that rigs exist)
        for frame_id, frame in reference.frames.items():
            db.write_frame(frame, use_frame_id=True)

        # Write images (now that frames exist)
        for image_id, image in reference.images.items():
            db.write_image(image, use_image_id=True)
            image_ids[image.name] = image_id

    # 5. Import features into database
    print("Importing features into database...")
    with pycolmap.Database.open(db_path) as db:
        for img_name, img_id in tqdm(image_ids.items()):
            if img_name not in db_images:
                continue
            idx = db_images.index(img_name)
            kpts = np.array(db_features[idx]["keypoints"]).astype(np.float64)
            kpts += 0.5  # COLMAP convention: center of pixel
            db.write_keypoints(img_id, kpts)

    # 6. Import matches and do geometric verification with known poses
    print("Importing matches and geometric verification...")

    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "hloc"))
    from hloc.utils.geometry import compute_epipolar_errors

    with open(pairs_path, "r") as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    name_to_idx = {name: i for i, name in enumerate(db_images)}
    inlier_ratios = []
    matched_pairs = set()

    with h5py.File(str(matches_path), "r") as hf, \
         pycolmap.Database.open(db_path) as db:
        for name0, name1 in tqdm(pairs):
            if name0 not in image_ids or name1 not in image_ids:
                continue
            id0, id1 = image_ids[name0], image_ids[name1]
            pair_key = (id0, id1) if id0 < id1 else (id1, id0)
            if pair_key in matched_pairs:
                continue
            matched_pairs.add(pair_key)

            hf_key = f"{name0}/{name1}"
            if hf_key not in hf:
                continue
            matches0 = hf[hf_key]["matches0"].__array__()
            valid_kp = np.where(matches0 > -1)[0]
            matches = np.stack([valid_kp, matches0[valid_kp]], -1)  # (N, 2)
            if len(matches) == 0:
                db.write_two_view_geometry(id0, id1, pycolmap.TwoViewGeometry())
                continue

            # Epipolar verification using known poses
            image0 = reference.images[id0]
            image1 = reference.images[id1]
            cam0 = reference.cameras[image0.camera_id]
            cam1 = reference.cameras[image1.camera_id]

            idx0 = name_to_idx.get(name0)
            idx1 = name_to_idx.get(name1)
            if idx0 is None or idx1 is None:
                continue
            kps0 = np.array(db_features[idx0]["keypoints"])
            kps1 = np.array(db_features[idx1]["keypoints"])

            # Normalize keypoints to camera coordinates
            if len(kps0) > 0:
                kps0_n = cam0.cam_from_img(kps0 + 0.5)
            else:
                kps0_n = np.zeros((0, 2))
            if len(kps1) > 0:
                kps1_n = cam1.cam_from_img(kps1 + 0.5)
            else:
                kps1_n = np.zeros((0, 2))

            try:
                cam1_from_cam0 = image1.cam_from_world() * image0.cam_from_world().inverse()
                errors0, errors1 = compute_epipolar_errors(
                    cam1_from_cam0.matrix(),
                    kps0_n[matches[:, 0]],
                    kps1_n[matches[:, 1]],
                )
                max_error = 4.0
                valid = np.logical_and(
                    errors0 <= cam0.cam_from_img_threshold(max_error),
                    errors1 <= cam1.cam_from_img_threshold(max_error),
                )
                inlier_ratios.append(np.mean(valid))

                tv = pycolmap.TwoViewGeometry(
                    inlier_matches=matches[valid],
                )
                db.write_two_view_geometry(id0, id1, tv)
            except Exception:
                db.write_two_view_geometry(id0, id1, pycolmap.TwoViewGeometry())
                inlier_ratios.append(0.0)

    if inlier_ratios:
        print(f"  mean/median inlier ratio: {np.mean(inlier_ratios)*100:.1f}% / "
              f"{np.median(inlier_ratios)*100:.1f}%")

    # 7. Triangulate points
    sfm_dir = output_dir / "sfm_superpoint_superglue"
    sfm_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running triangulation → {sfm_dir}...")
    reconstruction = pycolmap.triangulate_points(
        reference, db_path, str(db_image_root), str(sfm_dir),
        options={"ba_refine_focal_length": False, "ba_refine_extra_params": False}
    )
    print(f"Triangulation done.\n{reconstruction.summary()}")

    # 8. Build (image_name, kp_idx) -> point3D_id lookup
    print("Building point3D lookup...")
    valid_pids = set(reconstruction.points3D.keys())
    lookup = {}
    for image_id, image in reconstruction.images.items():
        img_name = image.name
        p2d = image.points2D
        if p2d is not None:
            for kp_idx, p in enumerate(p2d):
                pid = int(p.point3D_id)
                if pid in valid_pids:
                    lookup[(img_name, kp_idx)] = pid
    print(f"  Built lookup with {len(lookup)} entries")
    print(f"  Total time: {time.time() - t_start:.1f}s")

    return lookup, reconstruction, sfm_dir


if __name__ == "__main__":
    print("build_sfm.py - Use via run_pipeline.py with --build_sfm flag")
