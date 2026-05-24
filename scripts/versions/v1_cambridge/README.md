# Cambridge Experiments Code Snapshot (v1)

This directory contains the EXACT code used for all Cambridge KingsCollege experiments.

## Pipeline Version
- `run_pipeline.py` — Cambridge pipeline (no 7Scenes support)
- Git commit: a3c99db + unstaged modifications

## Experiment → Code Mapping

| Experiment | Config | Detector | Matcher | Retrieval |
|------------|--------|----------|---------|-----------|
| baseline_a | configs/baseline_a.yaml | sift.py | nn.py | netvlad.py |
| baseline_b | configs/baseline_b.yaml | superpoint.py | superglue.py | netvlad.py |
| baseline_b+tri | configs/baseline_b.yaml + --build_sfm | superpoint.py | superglue.py | netvlad.py |
| exp_retrieval | configs/exp_retrieval.yaml | superpoint.py | superglue.py | eigenplaces.py |
| exp_match | configs/exp_match.yaml | aliked.py | lightglue.py | netvlad.py |
| exp_full | configs/exp_full.yaml | aliked.py | lightglue.py | eigenplaces.py |
| exp_crica | configs/exp_crica.yaml | aliked.py | lightglue.py | cricavpr.py |

## Key Bug Fixes Applied
1. `aliked.py` line 69: `feats["scores"]` → `feats["keypoint_scores"]`
2. `lightglue.py`: Removed `.T` transpose; Added `.to(self.device)` to all tensors
3. `superglue.py`: Pass `scores0`/`scores1` to SuperGlue forward()

## Quick Verification
```bash
PYTHONPATH="." python scripts/versions/v1_cambridge/run_pipeline.py --config configs/baseline_a.yaml --limit_queries 5
```
