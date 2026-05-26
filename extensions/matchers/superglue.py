from extensions.matchers.base import BaseMatcher
from extensions import register_matcher
import numpy as np


@register_matcher("SuperGlueMatcher")
class SuperGlueMatcher(BaseMatcher):
    """
    Wrapper for SuperGlue from third_party/SuperGlue.
    """

    DEFAULT_CONFIG = {
        "weights": "indoor",
        "confidence_threshold": 0.2,
    }

    def _load_model(self):
        import sys
        import torch
        from pathlib import Path
        sp_dir = str(Path(__file__).resolve().parents[2] / "vendor" / "superglue")
        sys.path.insert(0, sp_dir)
        from superglue import SuperGlue
        config = {
            "descriptor": "superpoint",
            "weights": self.config.get("weights", "indoor"),
            "matching": {"threshold": self.config.get("confidence_threshold", 0.2)},
        }
        self.model = SuperGlue(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)

    def match(self, query_data, db_data):
        import torch
        import numpy as np

        kpts0 = query_data["keypoints"]
        kpts1 = db_data["keypoints"]
        desc0 = query_data["descriptors"].T
        desc1 = db_data["descriptors"].T

        if len(kpts0) < 2 or len(kpts1) < 2:
            return np.array([]), np.array([]), np.array([])

        scores0 = query_data.get("scores", np.ones(len(kpts0), dtype=np.float32))
        scores1 = db_data.get("scores", np.ones(len(kpts1), dtype=np.float32))

        img_size0 = query_data.get("image_size", (1, 1))
        img_size1 = db_data.get("image_size", (1, 1))
        w0, h0 = img_size0 if isinstance(img_size0, tuple) else (img_size0, img_size0)
        w1, h1 = img_size1 if isinstance(img_size1, tuple) else (img_size1, img_size1)
        data = {
            "keypoints0": torch.from_numpy(kpts0[np.newaxis]).float().to(self.device),
            "keypoints1": torch.from_numpy(kpts1[np.newaxis]).float().to(self.device),
            "descriptors0": torch.from_numpy(desc0[np.newaxis]).float().to(self.device),
            "descriptors1": torch.from_numpy(desc1[np.newaxis]).float().to(self.device),
            "scores0": torch.from_numpy(scores0[np.newaxis]).float().to(self.device),
            "scores1": torch.from_numpy(scores1[np.newaxis]).float().to(self.device),
            "image0": torch.zeros(1, 3, h0, w0, device=self.device),
            "image1": torch.zeros(1, 3, h1, w1, device=self.device),
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