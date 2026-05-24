from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Any


class BaseDetector(ABC):
    """
    Contract:
      - detect() returns keypoints (N,2) and descriptors (N,D)
      - scores optional but recommended for NMS
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            image: HxWx3 uint8 RGB
        Returns:
            keypoints:   (N, 2) float32  [x, y] pixel coords
            descriptors: (N, D) float32  L2-normalized
            scores:      (N,)   float32  confidence in [0,1]
        """
        ...

    def to_hloc_format(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        scores: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Dict:
        """
        Converts output to HLoc h5 format.
        image_shape: (H, W)
        """
        return {
            "keypoints": keypoints,  # (N,2)
            "descriptors": descriptors.T,  # (D,N) — HLoc convention
            "scores": scores,  # (N,)
            "image_size": np.array(image_shape[::-1]),  # (W, H)
        }