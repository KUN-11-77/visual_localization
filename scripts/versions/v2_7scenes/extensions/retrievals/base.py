from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class BaseRetrieval(ABC):
    """
    Contract:
      - __init__ receives config dict
      - encode() returns L2-normalized descriptor of shape (D,)
      - retrieve() returns ranked list of db image names
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load pretrained weights. Must be idempotent."""
        ...

    @abstractmethod
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HxWx3 uint8 RGB
        Returns:
            descriptor: (D,) float32, L2-normalized
        """
        ...

    def retrieve(
        self,
        query_desc: np.ndarray,
        db_descs: np.ndarray,
        db_names: List[str],
        top_k: int = 10,
    ) -> List[str]:
        """
        Default implementation: cosine similarity ranking.
        Override if the method has its own ranking logic.

        Args:
            query_desc: (D,) float32
            db_descs:   (N, D) float32
            db_names:   list of N image names
            top_k:      number of candidates to return
        Returns:
            ranked list of image names, length <= top_k
        """
        sims = db_descs @ query_desc  # (N,)
        indices = np.argsort(sims)[::-1][:top_k]
        return [db_names[i] for i in indices]