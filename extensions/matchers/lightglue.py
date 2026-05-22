from extensions.matchers.base import BaseMatcher
from extensions import register_matcher
import numpy as np


@register_matcher("LightGlueMatcher")
class LightGlueMatcher(BaseMatcher):
    """
    Wrapper for LightGlue (ICCV 2023).
    LightGlue is an efficient feature matcher with adaptive depth.
    Supports superpoint, aliked, disk, and sift feature types.
    """

    DEFAULT_CONFIG = {
        "features": "superpoint",      # "superpoint" | "aliked" | "disk" | "sift"
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "filter_threshold": 0.1,
        "device": "cuda",
    }

    def _load_model(self):
        import sys
        import torch
        from pathlib import Path

        try:
            from lightglue import LightGlue
        except ImportError:
            local_path = Path(__file__).resolve().parents[2] / "third_party" / "LightGlue" / "LightGlue-main"
            if local_path.exists():
                sys.path.insert(0, str(local_path))
                from lightglue import LightGlue
            else:
                raise ImportError(
                    "lightglue not installed. Install with:\n"
                    "  pip install lightglue\n"
                    "OR clone the repo:\n"
                    "  git clone https://github.com/cvg/LightGlue.git third_party/LightGlue"
                )

        self.model = LightGlue(
            features=self.config.get("features", "superpoint"),
            depth_confidence=self.config.get("depth_confidence", 0.95),
            width_confidence=self.config.get("width_confidence", 0.99),
            filter_threshold=self.config.get("filter_threshold", 0.1),
        )

        device = self.config.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model.eval().to(self.device)

    def match(self, query_data, db_data):
        import torch

        kpts0 = query_data["keypoints"]
        kpts1 = db_data["keypoints"]
        desc0 = query_data["descriptors"].T
        desc1 = db_data["descriptors"].T

        if len(kpts0) < 2 or len(kpts1) < 2:
            return np.array([]), np.array([]), np.array([])

        data = {
            "image0": {
                "keypoints": torch.from_numpy(kpts0[np.newaxis]).float(),
                "descriptors": torch.from_numpy(desc0[np.newaxis]).float(),
                "image_size": torch.tensor([query_data.get("image_size", (0, 0))]),
            },
            "image1": {
                "keypoints": torch.from_numpy(kpts1[np.newaxis]).float(),
                "descriptors": torch.from_numpy(desc1[np.newaxis]).float(),
                "image_size": torch.tensor([db_data.get("image_size", (0, 0))]),
            },
        }

        with torch.no_grad():
            pred = self.model(data)
            matches = pred["matches0"][0].cpu().numpy()
            scores = pred["matching_scores0"][0].cpu().numpy()

        valid = matches > -1
        q_idx = np.where(valid)[0]
        db_idx = matches[valid]
        conf = scores[valid]
        return q_idx, db_idx, conf
