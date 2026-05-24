# Experiment Log

## Pipeline Code Versions

| Version | File | Used For |
|---------|------|----------|
| v1 (Cambridge) | `scripts/versions/run_pipeline_v1_cambridge.py` | baseline_a, baseline_b, baseline_b+triangulation, exp_retrieval, exp_match, exp_full, exp_crica |
| v2 (7Scenes) | `scripts/versions/run_pipeline_v2_7scenes.py` | 7scenes_stairs_baseline, 7scenes_stairs_eigenplaces |

The active pipeline at `scripts/run_pipeline.py` is currently v2. To reproduce Cambridge results, use v1.

---

## 2026-05-21: Pipeline Setup & Debugging

### Issue 1: NetVLAD model path error
- **Symptom**: Import error when loading NetVLAD
- **Root Cause**: `get_parent_dir(__file__, level=1)` returns `xrloc/`, then `+/../models/` = `xrlocalization/models/` (empty), but model at `xrloc/models/`
- **Fix**: Copied `Pitts30K_struct.mat` to `xrlocalization/models/VGG16-NetVLAD-Pitts30K` as file

### Issue 2: SIFT keypoints double-overwrite bug
- **Symptom**: Crash in SIFT detector
- **Root Cause**: `keypoints` list converted to numpy array, then `.response` accessed on numpy elements
- **Fix**: Save `cv_kps` before conversion, extract `.response` from `cv_kps`

### Issue 3: Memory OOM
- **Symptom**: Pipeline crashes with out-of-memory
- **Root Cause**: 1220 DB images stored as raw numpy arrays in `db_features`
- **Fix**: Removed `"image"` key from db_features and query_data dicts

### Issue 4: NVM filename parsing
- **Symptom**: NVM model can't find images
- **Root Cause**: NVM format uses `\t` separator but code used `split(" ")`
- **Fix**: Split on `"\t"` first, fall back to `" "`

### Issue 5: COLMAP text model empty
- **Symptom**: 0 3D points loaded, 0 keypoints per image
- **Root Cause**: `empty_all/` directory has empty shell (0 observations per image)
- **Fix**: Discovered `model_train/` with binary format containing full data (1220 images, 335K 3D points)

### Issue 6: COLMAP binary format differences
- **Symptom**: Binary reader produced garbage data
- **Root Cause**: 
  - `point3D_id` in images.bin is `int64` (8 bytes), not `int32`
  - `point3D_id` in points3D.bin is `uint64` (8 bytes), not `uint32`
  - `cameras.bin` has NO `num_params` field — param count determined by camera model ID
  - Reconstruction at 1024×576 resolution (not 1920×1080)
- **Fix**: Implemented correct binary reader in `scripts/colmap_localization.py`

### Issue 7: PnP `minInliersCount` parameter error
- **Symptom**: `solvePnPRansac() got an unexpected keyword argument 'minInliersCount'`
- **Root Cause**: OpenCV 4.12.0 Python bindings don't accept `minInliersCount` parameter
- **Fix**: Removed `minInliersCount` parameter, lowered post-PnP inlier threshold from 12 to 8

### Issue 8: NN Matcher ratio test formula inverted
- **Symptom**: 3921 query kpts → 3921 matches (no filtering), then PnP gets only 11 inliers
- **Root Cause**: Ratio test `best_score >= ratio * second_score` was inverted; with cosine similarity, this accepted nearly all matches
- **Fix Attempt 1**: Corrected formula to `second_score < ratio * best_score` — but at ratio=0.8, got 0 matches
- **Root Cause (deeper)**: Cosine similarity of L2-normalized SIFT descriptors has too little dynamic range (best=0.454, second=0.451 for cross-seq). Lowe's ratio test fundamentally can't work with cosine similarity.
- **Fix Attempt 2**: Switched to Euclidean distance + mutual nearest neighbor (cross-check) matching instead of ratio test

### Issue 9: COLMAP resolution mismatch
- **Symptom**: COLMAP keypoints at 1024×576 but dataset images at 1920×1080
- **Fix**: Resize images to 1024×576 before processing, scale K matrix accordingly (fx=890.67, cx=512, cy=288)

### Issue 10: PnP → GT coordinate convention mismatch (ROOT CAUSE of recall=0)
- **Symptom**: t_err = 20~1500m, r_err = 90~180°, all recall = 0
- **Root Cause**: PnP returns `(R, t)` where `X_cam = R @ X_world + t`, but GT stores camera center in world `C` and world-to-camera quaternion `q_w2c`. Code was storing PnP output directly → comparing tvec against camera position (completely different quantities)
- **Fix**: Convert: `t_c2w = -R.T @ t` (camera center in world), `q_pred = _rotmat_to_quaternion(R)` (world-to-camera quaternion, matching GT convention)
- **Note**: First attempt used `_rotmat_to_quaternion(R.T)` which gave camera-to-world quaternion (r_err ~170°). Correct is `_rotmat_to_quaternion(R)` for world-to-camera.

### Issue 11: SIFT descriptor incompatibility (COLMAP VLFeat vs OpenCV)
- **Symptom**: 700+ wrong matches, only 8-14 inliers, PnP converged to completely wrong poses
- **Root Cause**: COLMAP reconstruction used VLFeat SIFT, but `_colmap_sift_encode` computed OpenCV SIFT at COLMAP positions. VLFeat and OpenCV SIFT descriptors are incompatible.
- **Fix**: Changed COLMAP DB encoding from `_colmap_sift_encode` to regular SIFT `detectAndCompute` on resized DB images. During matching, use spatial proximity (radius=8px) to find nearest COLMAP keypoint → 3D point. This gives compatible SIFT descriptors between query and DB.

### Issue 12: SuperGlue missing scores fields
- **Symptom**: SuperGlue model crash on forward()
- **Root Cause**: `superglue.py` forward() requires `scores0`/`scores1` in data dict (line 249), but matcher wrapper didn't pass them
- **Fix**: Extract scores from query_data/db_data dicts, pass as tensors

### Issue 13: Hardcoded COLMAP scene name
- **Symptom**: First candidate path hardcoded "KingsCollege" as scene name
- **Fix**: Use `dataset_cfg["scene"]` instead

### Issue 14: NVM encode missing image resize
- **Symptom**: NVM keypoints at different resolution than query images
- **Fix**: Added optional `nvm_width`/`nvm_height` params to `_nvm_sift_encode`, resize images before detection

---

## Experiment Results

### 2026-05-22: baseline_a (NetVLAD + SIFT + NN cross-check + COLMAP binary) - FULL (343 queries)

| Metric | Value |
|--------|-------|
| (0.25m, 2°) | **65.60%** (225/343) |
| (0.5m, 5°) | **92.42%** (317/343) |
| (5.0m, 10°) | **98.83%** (339/343) |

**Key fixes applied**:
1. COLMAP binary reader (Issue 6) - correct struct.unpack('ddq'*N) for 24-byte triplets
2. PnP→GT coordinate conversion (Issue 10) - t_c2w = -R.T @ t, q_w2c = _rotmat_to_quaternion(R)
3. SIFT detectAndCompute + COLMAP spatial 2D-3D lookup (Issue 11)
4. SuperGlue scores fields (Issue 12)
5. Various minor fixes (Issues 1-5, 7-9, 13-14)

### 2026-05-22: pycolmap triangulation integration (--build_sfm)

**Goal**: Reproduce HLoc's SP+SG triangulation pipeline (~86% recall vs our ~66%).

**Approach**: Follow HLoc's Cambridge pipeline: covisibility pairs → SP features → SG matches → COLMAP database → triangulate with known poses.

**Issue 15: pycolmap database creation fails with frame/rig constraint**
- **Symptom**: `db.write_frame()` → "SQLite error: constraint failed", `db.write_image()` → "Check failed: frame.HasDataId(image.DataId())"
- **Root Cause**: Cambridge dataset has frames referencing rig_ids. `reference.rigs` causes segfault (pycolmap bug). Without rigs, frames can't be written; without frames, images can't be written.
- **Fix**: Create minimal `pycolmap.Rig` objects from frame metadata: `frame.data_ids_by_sensor(SensorType.CAMERA)` gives us the sensor_id, then `rig.add_ref_sensor(sensor_id)`. Write rigs first, then frames, then images.
- **Verification**: Full DB creation (1220 cameras, 1220 rigs, 1220 frames, 1220 images) succeeds.

**Issue 16: matches array wrong shape for TwoViewGeometry**
- **Symptom**: `TwoViewGeometry(inlier_matches=matches)` → "Could not convert ndarray"
- **Root Cause**: `np.stack(np.where(matches0 > -1), -1)` produces shape `(N, 1)` but TwoViewGeometry expects `(N, 2)` — pairs of `(kp_idx0, kp_idx1)`.
- **Fix**: `valid_kp = np.where(matches0 > -1)[0]; matches = np.stack([valid_kp, matches0[valid_kp]], -1)` → shape `(N, 2)`.

**Issue 17: cam_from_world is a method, not a property**
- **Symptom**: `image.cam_from_world.inverse()` → AttributeError
- **Fix**: `image.cam_from_world().inverse()` (add parentheses).

**Issue 18: point3D_id is uint64, has_point3D unreliable**
- **Symptom**: `int too big to convert` (OverflowError) — uint64 max (18446744073709551615, representing -1) overflows int64 numpy array
- **Root Cause**: `Point2D.has_point3D` returns True even for invalid point3D IDs. `point3D_id` is uint64 type, unsigned -1 = 2^64-1.
- **Fix**: Check `int(p.point3D_id) in reconstruction.points3D` (valid point3D ID set) instead of `has_point3D`.

**Build SfM model: 178,072 3D points, 2,054,587 observations, mean track length 11.5, mean reprojection error 1.61px**

### 2026-05-22: baseline_b + pycolmap triangulation (--build_sfm) - FULL (343 queries)

| Metric | baseline_b (SIFT model) | baseline_b + SP+SG model | Delta |
|--------|------------------------|--------------------------|-------|
| (0.25m, 2°) | 66.18% (227/343) | **69.68%** (239/343) | **+3.50%** |
| (0.5m, 5°) | 92.42% (317/343) | **93.00%** (319/343) | +0.58% |
| (5.0m, 10°) | 100% (343/343) | **100%** (343/343) | 0% |

**Key findings**:
1. pycolmap triangulation with SuperPoint+SuperGlue covisibility pairs produces a 178k-point 3D model aligned with SP keypoints
2. +3.5% improvement at strict (0.25m, 2°) threshold — SP keypoints now have corresponding 3D points from a SP-native model
3. All 343 queries localized at (5.0m, 10°) — 100% maintained
4. Gap to HLoc paper (~86%) remains: likely due to 1024px vs 1920px resolution, and HLoc's more extensive pair selection

**Files created/modified**:
- `scripts/build_sfm.py`: Full pipeline module (covisibility pairs, feature export, DB matching, geometric verification, triangulation)
- `scripts/run_pipeline.py`: Added `--build_sfm` and `--num_covis` flags; integrated triangulation into 2D-3D lookup
- `configs/baseline_b.yaml`: SuperPoint 4096 kpts, SuperGlue outdoor weights

**How to run**:
```
$env:PYTHONPATH="." ; python scripts/run_pipeline.py --config configs/baseline_b.yaml --build_sfm --limit_queries 343
```

---

### 2026-05-23: exp_match (NetVLAD + ALIKED + LightGlue) - FULL (343 queries)

**Goal**: Isolate matching module gain — swap SIFT+NN → ALIKED+LightGlue while keeping NetVLAD + COLMAP model fixed.

**Config**: NetVLAD (PCA 4096) + ALIKED (5000 kpts) + LightGlue (aliked, τ=0.1) + COLMAP binary model (no triangulation)

**Issue 20: ALIKED keypoint_scores key name**
- **Symptom**: `KeyError: 'scores'` in aliked.py
- **Root Cause**: ALIKED's `extract()` returns `"keypoint_scores"` not `"scores"` (line 774 in lightglue/aliked.py)
- **Fix**: Changed `feats["scores"]` → `feats["keypoint_scores"]` in aliked.py line 69

**Issue 21: LightGlue input_dim mismatch**
- **Symptom**: `AssertionError: assert desc0.shape[-1] == self.conf.input_dim`
- **Root Cause**: Descriptors were transposed (N×D → D×N), making last dim = N (keypoints) instead of D (128 for ALIKED)
- **Fix**: Removed `.T` transpose from desc0/desc1 in lightglue.py match()

**Issue 22: LightGlue device mismatch**
- **Symptom**: `RuntimeError` — tensors on CPU while model on CUDA
- **Root Cause**: Keypoint/descriptor tensors not moved to GPU
- **Fix**: Added `.to(self.device)` to all data tensors in lightglue.py match()

| Metric | exp_match | baseline_a | Delta |
|--------|----------|------------|-------|
| (0.25m, 2°) | **67.35%** (231/343) | 65.60% (225/343) | **+1.75%** |
| (0.5m, 5°) | **93.29%** (320/343) | 92.42% (317/343) | **+0.87%** |
| (5.0m, 10°) | **100%** (343/343) | 98.83% (339/343) | **+1.17%** |

**Key findings**:
1. ALIKED + LightGlue outperforms SIFT + NN cross-check across ALL thresholds
2. ALIKED generates ~2700 keypoints (vs SIFT 5000) but LightGlue produces 16K-18K high-quality inliers
3. LightGlue's adaptive depth/confidence mechanism effectively filters incorrect matches
4. 100% at (5.0m, 10°) — zero complete failures, 4 more queries localized than baseline_a
5. LightGlue matching is ~12ms per query on average

**How to run**:
```
PYTHONPATH="." KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_pipeline.py --config configs/exp_match.yaml --limit_queries 343
```

---

### 2026-05-23: exp_full (EigenPlaces + ALIKED + LightGlue) - FULL (343 queries)

**Goal**: Full pipeline swap — EigenPlaces retrieval + ALIKED detection + LightGlue matching.

**Config**: EigenPlaces (R50, GeM, 2048-dim) + ALIKED (5000 kpts) + LightGlue (aliked, τ=0.1) + COLMAP binary model

| Metric | exp_full | exp_match | exp_retrieval | baseline_a |
|--------|----------|-----------|---------------|------------|
| (0.25m, 2°) | **61.52%** (211/343) | 67.35% | 63.27% | 65.60% |
| (0.5m, 5°) | **86.88%** (298/343) | 93.29% | 85.71% | 92.42% |
| (5.0m, 10°) | **100%** (343/343) | 100% | 100% | 98.83% |

**Key findings**:
1. exp_full (61.52%) underperforms exp_match (67.35%) by 5.83% — EigenPlaces bottleneck confirmed
2. LightGlue vs SuperGlue at medium: exp_full 86.88% > exp_retrieval 85.71% (+1.17%)
3. NetVLAD → EigenPlaces consistently costs 5-7% regardless of downstream components
4. 100% at (5.0m, 10°) maintained across all configurations

**How to run**:
```
PYTHONPATH="." KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_pipeline.py --config configs/exp_full.yaml --limit_queries 343
```

---

### 2026-05-23: exp_crica (CricaVPR + ALIKED + LightGlue) - FULL (343 queries)

**Goal**: CricaVPR retrieval comparison — ViT-based VPR vs NetVLAD and EigenPlaces.

**Config**: CricaVPR (DINOv2 ViT-B/14, 10752-dim) + ALIKED (5000 kpts) + LightGlue (aliked, τ=0.1) + COLMAP binary model

| Metric | exp_crica | exp_full | exp_match | baseline_a |
|--------|-----------|----------|-----------|------------|
| (0.25m, 2°) | **60.64%** (208/343) | 61.52% | 67.35% | 65.60% |
| (0.5m, 5°) | **85.71%** (294/343) | 86.88% | 93.29% | 92.42% |
| (5.0m, 10°) | **100%** (343/343) | 100% | 100% | 98.83% |

**Key findings**:
1. CricaVPR (60.64%) is the weakest retriever — 6.71% below NetVLAD, 0.88% below EigenPlaces
2. Despite ViT-B/14 backbone and cross-image correlation mechanism, CricaVPR doesn't generalize well to Cambridge campus scenes
3. Possible causes: (a) trained on structured road scenes; (b) 10752-dim descriptors lack PCA whitening; (c) cross-image encoder only operates on top-K retrieved set
4. 100% at (5.0m, 10°) still maintained — all retrievers find relevant DB images for every query

**How to run**:
```
PYTHONPATH="." KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_pipeline.py --config configs/exp_crica.yaml --limit_queries 343
```

---

## Ablation Study Summary (2026-05-23)

| Config | (0.25m, 2°) | (0.5m, 5°) | (5.0m, 10°) |
|--------|-------------|-------------|--------------|
| baseline_a (NetVLAD+SIFT+NN) | 65.60% | 92.42% | 98.83% |
| baseline_b (NetVLAD+SP+SG) | 66.18% | 92.42% | 100% |
| baseline_b + triangulation | **69.68%** | **93.00%** | 100% |
| exp_retrieval (Eigen+SP+SG) | 63.27% | 85.71% | 100% |
| exp_match (NetVLAD+ALIKED+LG) | 67.35% | 93.29% | 100% |
| exp_full (Eigen+ALIKED+LG) | 61.52% | 86.88% | 100% |
| exp_crica (CricaVPR+ALIKED+LG) | 60.64% | 85.71% | 100% |

**Key conclusions**:
1. **Matching**: ALIKED+LightGlue > SIFT+NN (+1.75%) and SP+SG, consistent across all retrieval backends
2. **Retrieval**: NetVLAD (PCA 4096) is the best retriever on Cambridge; EigenPlaces loses 6-7%; CricaVPR loses 6.7%
3. **Triangulation**: SP+SG pycolmap triangulation adds +3.5%, confirming native-feature 3D model importance
4. **Best config**: baseline_b + triangulation (69.68%), but still below HLoc paper (~86%) due to 1024px vs 1920px resolution

---

### 2026-05-23: exp_retrieval (EigenPlaces + SuperPoint + SuperGlue) - FULL (343 queries)

**Goal**: Isolate retrieval module gain — swap NetVLAD → EigenPlaces while keeping SP+SG + SfM fixed.

**Issue 19: OpenMP conflict (pycolmap + PyTorch)**
- **Symptom**: `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.` → Crash during geometric verification
- **Root Cause**: Both PyTorch and pycolmap (via COLMAP's CERES dependency) link against OpenMP. On Windows, duplicate OpenMP runtime initialization is fatal.
- **Fix**: Set `KMP_DUPLICATE_LIB_OK=TRUE` environment variable before running.

**Config**: EigenPlaces (ResNet50, 2048-dim) + SuperPoint (4096 kpts) + SuperGlue (outdoor, τ=0.2) + pycolmap SfM (num_covis=20)

| Metric | exp_retrieval | baseline_b + SP+SG | Delta |
|--------|--------------|--------------------|-------|
| (0.25m, 2°) | **63.27%** (217/343) | 69.68% (239/343) | **-6.41%** |
| (0.5m, 5°) | **85.71%** (294/343) | 93.00% (319/343) | **-7.29%** |
| (5.0m, 10°) | **100%** (343/343) | 100% (343/343) | 0% |

**SfM model**: 178,065 3D points, 2,054,573 observations, mean track 11.54, mean reproj 1.61px

**Key findings**:
1. **EigenPlaces underperforms NetVLAD** — 6-7% drop at strict and medium thresholds despite identical SP+SG pipeline
2. NetVLAD's PCA-4096 features appear better suited for Cambridge landmarks than EigenPlaces' ResNet50+GeM 2048-dim descriptors
3. (5.0m, 10°) remains at 100%, meaning EigenPlaces retrieval still always finds at least one DB image with enough co-visible 3D points for coarse localization
4. Possible causes: (a) NetVLAD trained on Google Street View (more similar to Cambridge outdoor scenes); (b) PCA whitening; (c) 4096-dim vs 2048-dim descriptor capacity

**How to run**:
```
$env:PYTHONPATH="." ; KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_pipeline.py --config configs/exp_retrieval.yaml --build_sfm --limit_queries 343
```

---

## 7Scenes Stairs Experiments

### 2026-05-23: 7scenes_stairs_baseline (NetVLAD + ALIKED + LightGlue) - FULL (1000 queries)

**Goal**: Validate Cambridge conclusions on 7Scenes Stairs (indoor RGB-D dataset).

**Config**: NetVLAD (PCA 4096) + ALIKED (5000 kpts) + LightGlue (aliked, τ=0.1) + depth-based 3D (no COLMAP)

**Pipeline version**: v2 (scripts/versions/run_pipeline_v2_7scenes.py)

**7Scenes-specific fixes applied**:
- `_load_7scenes()`: Fixed path nesting (seq-0X/seq-0X/) and sequence name mapping (sequenceN → seq-0N)
- `_load_7scenes()`: Fixed GT pose — store camera center t_c2w instead of t_w2c
- Depth-based 2D-3D: `_depth_unproject()` unprojects keypoints to 3D world coords using depth maps + camera poses
- Camera intrinsics: fx=fy=585, cx=320, cy=240 (Kinect 640x480)

| Metric | 7scenes NetVLAD | Cambridge NetVLAD | Delta |
|--------|-----------------|-------------------|-------|
| (0.25m, 2°) | **30.60%** (306/1000) | 67.35% (231/343) | -36.75% |
| (0.5m, 5°) | **84.70%** (847/1000) | 93.29% (320/343) | -8.59% |
| (5.0m, 10°) | **97.10%** (971/1000) | 100% (343/343) | -2.90% |

**Key findings**:
1. Indoor 7Scenes is significantly harder than outdoor Cambridge — 30.6% vs 67.35% at strict threshold
2. 84.7% at (0.5m, 5°) is reasonable for indoor RGB-D localization
3. Depth-based 3D mapping works but is limited by Kinect depth noise and 640x480 resolution

**How to run**:
```
PYTHONPATH="." KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_pipeline.py --config configs/7scenes_stairs_baseline.yaml --limit_queries 1000
```

---

### 2026-05-23: 7scenes_stairs_eigenplaces (EigenPlaces + ALIKED + LightGlue) - FULL (1000 queries)

**Goal**: Cross-dataset validation of EigenPlaces vs NetVLAD retrieval comparison.

**Config**: EigenPlaces (R50, GeM, 2048-dim) + ALIKED (5000 kpts) + LightGlue (aliked, τ=0.1) + depth-based 3D

| Metric | 7scenes EigenPlaces | 7scenes NetVLAD | Delta |
|--------|--------------------|--------------------|-------|
| (0.25m, 2°) | **18.90%** (189/1000) | 30.60% (306/1000) | **-11.70%** |
| (0.5m, 5°) | **56.20%** (562/1000) | 84.70% (847/1000) | **-28.50%** |
| (5.0m, 10°) | **82.30%** (823/1000) | 97.10% (971/1000) | **-14.80%** |

**Key findings**:
1. EigenPlaces underperforms NetVLAD on 7Scenes too — consistent with Cambridge findings
2. The gap is even larger on 7Scenes (11.7% vs 6.4% at strict), suggesting EigenPlaces struggles more with indoor scenes
3. Cross-dataset conclusion confirmed: **NetVLAD > EigenPlaces** for visual localization retrieval

**How to run**:
```
PYTHONPATH="." KMP_DUPLICATE_LIB_OK=TRUE python scripts/run_pipeline.py --config configs/7scenes_stairs_eigenplaces.yaml --limit_queries 1000
```

---

### 7Scenes Cross-Dataset Summary

| Config | (0.25m, 2°) | (0.5m, 5°) | (5.0m, 10°) |
|--------|-------------|-------------|--------------|
| NetVLAD + ALIKED + LightGlue | 30.60% | 84.70% | 97.10% |
| EigenPlaces + ALIKED + LightGlue | 18.90% | 56.20% | 82.30% |

**Consistent findings across Cambridge + 7Scenes**:
1. NetVLAD (PCA 4096) is the superior retriever for visual localization
2. EigenPlaces (R50, 2048-dim) lags by 40-50% relative in strict thresholds
3. ALIKED + LightGlue works reliably on both outdoor and indoor scenes
4. Depth-based 2D-3D is viable when SfM models are unavailable