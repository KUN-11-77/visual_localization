from extensions.detectors.base import BaseDetector
from extensions import register_detector
import numpy as np


@register_detector("ALIKEDDetector")
class ALIKEDDetector(BaseDetector):
    """
    Wrapper for ALIKED (A Lightweight Keypoint and Descriptor)
    from the LightGlue package.
    """

    DEFAULT_CONFIG = {
        "max_keypoints": 5000,
        "detection_threshold": 0.2,
        "device": "cuda",
    }

    def _load_model(self):
        import sys
        import torch
        from pathlib import Path

        # Try lightglue from pip, then from local clone
        try:
            from lightglue import ALIKED
        except ImportError:
            local_path = Path(__file__).resolve().parents[2] / "third_party" / "LightGlue" / "LightGlue-main"
            if local_path.exists():
                sys.path.insert(0, str(local_path))
                from lightglue import ALIKED
            else:
                raise ImportError(
                    "lightglue not installed. Install with:\n"
                    "  pip install lightglue\n"
                    "OR clone the repo:\n"
                    "  git clone https://github.com/cvg/LightGlue.git third_party/LightGlue"
                )

        self.device = self.config.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.model = ALIKED(
            max_num_keypoints=self.config.get("max_keypoints", 5000),
            detection_threshold=self.config.get("detection_threshold", 0.2),
        ).eval()
        self.model = self.model.to(self.device)

    def detect(self, image: np.ndarray):
        import torch

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        image_t = (
            torch.from_numpy(image.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            / 255.0
        )
        image_t = image_t.to(self.device)

        with torch.no_grad():
            feats = self.model.extract(image_t)

        keypoints = feats["keypoints"][0].cpu().numpy().astype(np.float32)
        descriptors = feats["descriptors"][0].cpu().numpy().astype(np.float32)
        scores = feats["scores"][0].cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        descriptors = descriptors / norms
        scores = np.clip(scores, 0.0, 1.0)

        return keypoints, descriptors, scores
