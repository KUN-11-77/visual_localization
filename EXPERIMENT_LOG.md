# Experiment Log

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

### 2026-05-22: baseline_a (NetVLAD + SIFT + NN cross-check + COLMAP binary) - 50 queries

| Metric | Value |
|--------|-------|
| (0.25m, 2°) | 14.29% (49/50 localized) |
| (0.5m, 5°) | 14.58% (50/50 localized) |
| (5.0m, 10°) | 14.58% (50/50 localized) |

**Note**: Recall computed against all 343 queries but only 50 tested. Actual per-query success rate: 49/50 = 98% at (0.25m, 2°).

**Error distribution** (50 queries):
- Median t_err: ~0.07m
- Median r_err: ~0.10°
- Best: t_err=0.009m, r_err=0.025°
- Only failure: frame00025 (t_err=0.29m barely over 0.25m threshold, r_err=0.25° well within)

**Key fixes applied**:
1. COLMAP binary reader (Issue 6) - correct struct.unpack('ddq'*N) for 24-byte triplets
2. PnP→GT coordinate conversion (Issue 10) - t_c2w = -R.T @ t, q_w2c = _rotmat_to_quaternion(R)
3. SIFT detectAndCompute + COLMAP spatial 2D-3D lookup (Issue 11)
4. SuperGlue scores fields (Issue 12)
5. Various minor fixes (Issues 1-5, 7-9, 13-14)