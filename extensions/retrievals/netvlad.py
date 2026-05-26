from extensions.retrievals.base import BaseRetrieval
from extensions import register_retrieval
import numpy as np


@register_retrieval("NetVLADRetrieval")
class NetVLADRetrieval(BaseRetrieval):
    """
    Wrapper for NetVLAD from XRLocalization.
    """

    DEFAULT_CONFIG = {
        "model_name": "VGG16-NetVLAD-Pitts30K",
        "whiten": True,
    }

    def _load_model(self):
        import sys
        from pathlib import Path
        vendor_path = str(Path(__file__).resolve().parents[2] / "vendor" / "netvlad")
        sys.path.insert(0, vendor_path)
        from netvlad import NetVLAD as _NetVLAD
        self.model = _NetVLAD({"model_name": self.config.get("model_name", "VGG16-NetVLAD-Pitts30K"),
                                "whiten": self.config.get("whiten", True)})
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(self.device)

    def encode(self, image: np.ndarray) -> np.ndarray:
        import torch
        import numpy as np
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        image_t = image_t.to(self.device)
        with torch.no_grad():
            data = self.model(image_t)
        desc = data["global_descriptor"][0].cpu().numpy()
        desc = desc / np.linalg.norm(desc)
        return desc