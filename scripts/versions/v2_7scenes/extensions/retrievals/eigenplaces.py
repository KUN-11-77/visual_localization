from extensions.retrievals.base import BaseRetrieval
from extensions import register_retrieval
import numpy as np


def _build_eigenplaces(backbone: str, fc_output_dim: int):
    """Build GeoLocalizationNet without torch.hub (no CosPlace download)."""
    import torch
    import torchvision
    from torch import nn

    CHANNELS_NUM_IN_LAST_CONV = {
        "ResNet18": 512, "ResNet50": 2048, "ResNet101": 2048,
        "ResNet152": 2048, "VGG16": 512,
    }
    assert backbone in CHANNELS_NUM_IN_LAST_CONV

    # Get torchvision backbone
    tv_model = getattr(torchvision.models, backbone.lower())()

    if backbone.startswith("ResNet"):
        layers = list(tv_model.children())[:-2]
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone]
    elif backbone == "VGG16":
        layers = list(tv_model.features.children())[:-2]
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone]

    encoder = nn.Sequential(*layers)

    # GeM pooling + FC layer
    from eigenplaces_model.layers import Flatten, L2Norm, GeM

    class GeoLocalizationNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = encoder
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm(),
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.aggregation(x)
            return x

    return GeoLocalizationNet()


@register_retrieval("EigenPlacesRetrieval")
class EigenPlacesRetrieval(BaseRetrieval):
    """
    Wrapper for EigenPlaces: deep visual place recognition with
    ResNet-50 + GeM pooling backbone, producing L2-normalized
    global descriptors.

    Reference: https://github.com/gmberton/EigenPlaces
    """

    DEFAULT_CONFIG = {
        "backbone": "ResNet50",
        "fc_output_dim": 2048,
        "device": "cuda",
        "input_size": 320,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    def _load_model(self):
        import torch
        import sys
        from pathlib import Path

        device = self.config.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        backbone = self.config.get("backbone", "ResNet50")
        fc_output_dim = self.config.get("fc_output_dim", 2048)

        base_path = Path(__file__).resolve().parents[2] / "third_party" / "eigenplaces"
        repo_path = base_path / "EigenPlaces-main"
        if not repo_path.exists():
            raise RuntimeError(
                "EigenPlaces repo not found. Clone it with:\n"
                "  git clone https://github.com/gmberton/EigenPlaces.git "
                "third_party/eigenplaces\n"
                "Or extract the ZIP to third_party/eigenplaces/EigenPlaces-main/"
            )
        sys.path.insert(0, str(repo_path))

        # Direct model construction — avoid CosPlace torch.hub download.
        # The EigenPlaces checkpoint already contains all backbone weights.
        self.model = _build_eigenplaces(backbone, fc_output_dim)

        # Load pretrained weights
        weights_path = base_path / "weights"
        pth_files = list(weights_path.glob("*.pth")) if weights_path.exists() else []
        if pth_files:
            state = torch.load(pth_files[0], map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        else:
            raise RuntimeError(
                "EigenPlaces weights not found. Download the pretrained weights:\n"
                "  https://github.com/gmberton/EigenPlaces/releases\n"
                f"  and place the .pth file in {weights_path}/"
            )

        self.model.eval().to(self.device)

    def encode(self, image: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms.functional as TF

        # Handle grayscale input
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Numpy HxWxC uint8 -> tensor CxHxW float [0, 1]
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Resize to model input size
        input_size = self.config.get("input_size", 320)
        image_t = TF.resize(image_t, [input_size, input_size], antialias=True)

        # Normalize with ImageNet statistics
        mean = self.config.get("mean", [0.485, 0.456, 0.406])
        std = self.config.get("std", [0.229, 0.224, 0.225])
        image_t = TF.normalize(image_t, mean=mean, std=std)

        # Add batch dimension
        image_t = image_t.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            desc = self.model(image_t)

        # Handle different output shapes: (1, D), (D,), (1, D, 1) etc.
        if isinstance(desc, torch.Tensor):
            desc = desc.squeeze()
        desc = desc.cpu().numpy().astype(np.float32)

        # L2-normalize
        desc = desc / (np.linalg.norm(desc) + 1e-12)

        return desc
