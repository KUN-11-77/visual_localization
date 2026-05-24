from extensions.detectors.base import BaseDetector
from extensions import register_detector
import numpy as np


@register_detector("SuperPointDetector")
class SuperPointDetector(BaseDetector):
    """
    Wrapper for SuperPoint from SuperGluePretrainedNetwork.
    """

    DEFAULT_CONFIG = {
        "max_keypoints": 2048,
        "keypoint_threshold": 0.005,
        "nms_radius": 4,
        "remove_borders": 4,
    }

    def _load_model(self):
        import sys
        import torch
        from pathlib import Path

        sp_dir = str(Path(__file__).resolve().parents[2] / "third_party" / "SuperGluePretrainedNetwork")
        if sp_dir not in sys.path:
            sys.path.insert(0, sp_dir)
        from models.superpoint import SuperPoint

        self.config_sp = {
            "max_keypoints": self.config.get("max_keypoints", 2048),
            "keypoint_threshold": self.config.get("keypoint_threshold", 0.005),
            "nms_radius": self.config.get("nms_radius", 4),
            "remove_borders": self.config.get("remove_borders", 4),
        }
        self.model = SuperPoint(self.config_sp)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)

    def detect(self, image: np.ndarray):
        import torch
        import cv2

        # SuperPoint expects single-channel grayscale input
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        image_t = (
            torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        )
        image_t = image_t.to(self.device)

        with torch.no_grad():
            data = self.model({"image": image_t})

        keypoints = data["keypoints"][0].cpu().numpy()
        descriptors = data["descriptors"][0].cpu().numpy().T  # (D,N) -> (N,D)
        scores = data["scores"][0].cpu().numpy()

        return keypoints, descriptors, scores
