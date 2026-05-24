# 7Scenes Experiments Code Snapshot (v2)

This directory contains the EXACT code used for 7Scenes stairs experiments.

## Pipeline Version
- `run_pipeline.py` — Extended pipeline with:
  - Fixed `_load_7scenes()` (path nesting + GT pose format)
  - Depth-based 2D-3D mapping via `_depth_unproject()`
  - `db_pose_files` support for RGB-D datasets

## Experiment → Code Mapping

| Experiment | Config | Detector | Matcher | Retrieval |
|------------|--------|----------|---------|-----------|
| 7scenes_baseline | configs/7scenes_stairs_baseline.yaml | aliked.py | lightglue.py | netvlad.py |
| 7scenes_eigenplaces | configs/7scenes_stairs_eigenplaces.yaml | aliked.py | lightglue.py | eigenplaces.py |

## Key Changes from v1
1. `_load_7scenes()`: Handles nested `seq-0X/seq-0X/` structure and `sequenceN` → `seq-0N` mapping
2. `_load_7scenes()`: Fixed GT pose to store camera center (t_c2w) instead of t_w2c
3. `main()`: Added `use_7scenes_depth` branch — loads depth maps, unprojects keypoints to 3D
4. `_depth_unproject()`: New function for RGB-D → 3D world coordinate conversion

## Quick Verification
```bash
PYTHONPATH="." KMP_DUPLICATE_LIB_OK=TRUE python scripts/versions/v2_7scenes/run_pipeline.py --config configs/7scenes_stairs_baseline.yaml --limit_queries 5
```
