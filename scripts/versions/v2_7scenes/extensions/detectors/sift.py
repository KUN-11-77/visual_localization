from extensions.detectors.base import BaseDetector
from extensions import register_detector
import numpy as np


@register_detector("SIFTDetector")
class SIFTDetector(BaseDetector):
    """
    Wrapper for OpenCV SIFT detector.
    """

    DEFAULT_CONFIG = {
        "max_keypoints": 5000,
        "contrast_threshold": 0.04,
        "edge_threshold": 10,
        "sigma": 1.6,
    }

    def _load_model(self):
        import cv2
        self.detector = cv2.SIFT_create(
            nfeatures=self.config.get("max_keypoints", 5000),
            contrastThreshold=self.config.get("contrast_threshold", 0.04),
            edgeThreshold=self.config.get("edge_threshold", 10),
            sigma=self.config.get("sigma", 1.6),
        )

    def detect(self, image: np.ndarray):
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 128), dtype=np.float32),
                np.ones(0, dtype=np.float32),
            )

        cv_kps = keypoints
        keypoints = np.array([kp.pt for kp in cv_kps], dtype=np.float32)
        scores = np.array([kp.response for kp in cv_kps], dtype=np.float32)

        if len(keypoints) > self.config.get("max_keypoints", 5000):
            top_k = self.config.get("max_keypoints", 5000)
            indices = np.argsort(scores)[::-1][:top_k]
            keypoints = keypoints[indices]
            descriptors = descriptors[indices]
            scores = scores[indices]

        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        descriptors = descriptors / norms

        return keypoints, descriptors.astype(np.float32), scores