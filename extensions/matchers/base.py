from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Any


class BaseMatcher(ABC):
    """
    Contract:
      - match() returns indices into query and db keypoint arrays
      - confidence optional
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        ...

    @abstractmethod
    def match(
        self,
        query_data: Dict,
        db_data: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            query_data: {"keypoints": (N,2), "descriptors": (D,N), "image": HxWx3}
            db_data:    {"keypoints": (M,2), "descriptors": (D,M), "image": HxWx3}
        Returns:
            query_idx: (K,) int32   indices into query keypoints
            db_idx:    (K,) int32   indices into db keypoints
            conf:      (K,) float32 match confidence in [0,1]
        """
        ...