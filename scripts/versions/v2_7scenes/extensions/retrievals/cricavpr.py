from extensions.retrievals.base import BaseRetrieval
from extensions import register_retrieval
import numpy as np


@register_retrieval("CricaVPRRetrieval")
class CricaVPRRetrieval(BaseRetrieval):
    """
    Wrapper for CricaVPR: Cross-Image Correlation-Aware Representation
    for Visual Place Recognition.

    Uses a DINOv2 ViT-B/14 backbone with a cross-image transformer encoder
    to produce L2-normalized global descriptors (dimension 10752).

    Reference: https://github.com/Lu-Feng/CricaVPR  (CVPR 2024)

    Required weight files (place in weights/):
      - weights/CricaVPR.pth         (trained CricaVPR model weights)
      - weights/dinov2_vitb14_pretrain.pth  (DINOv2 backbone checkpoint)
    """

    DEFAULT_CONFIG = {
        "backbone": "dinov2_vitb14",
        "weights_path": "weights/CricaVPR.pth",
        "dinov2_weights": "weights/dinov2_vitb14_pretrain.pth",
        "device": "cuda",
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    def _load_model(self):
        import torch
        import sys
        from pathlib import Path

        # Resolve absolute paths for weights (relative to project root)
        project_root = Path(__file__).resolve().parents[2]

        weights_path = Path(self.config.get("weights_path", self.DEFAULT_CONFIG["weights_path"]))
        dinov2_weights = Path(self.config.get("dinov2_weights", self.DEFAULT_CONFIG["dinov2_weights"]))
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        if not dinov2_weights.is_absolute():
            dinov2_weights = project_root / dinov2_weights

        # Validate weight files exist
        missing = []
        if not weights_path.exists():
            missing.append(f"  - CricaVPR weights: {weights_path}\n"
                          f"    Download from: https://github.com/Lu-Feng/CricaVPR/releases/download/v1.0/CricaVPR.pth")
        if not dinov2_weights.exists():
            missing.append(f"  - DINOv2 backbone weights: {dinov2_weights}\n"
                          f"    Download from: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")
        if missing:
            raise FileNotFoundError(
                "CricaVPRRetrieval: missing weight file(s). Please download the following files and place them in the weights/ directory:\n\n"
                + "\n".join(missing) + "\n\n"
                "Expected location: " + str(project_root / "weights") + "\n"
            )

        # Patch torch.load to default weights_only=False for PyTorch >= 2.6.
        # The upstream CricaVPR code (network.py get_backbone) calls torch.load
        # without weights_only=False, which breaks on PyTorch 2.6+.
        _orig_torch_load = torch.load
        def _patched_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_torch_load(*a, **kw)
        torch.load = _patched_load

        try:
            # Add third_party/cricavpr to sys.path so network.py and backbone/ can be imported
            cricavpr_path = project_root / "third_party" / "cricavpr"
            if not cricavpr_path.exists():
                raise RuntimeError(
                    "CricaVPR repo not found at third_party/cricavpr/.\n"
                    "Clone it with:\n"
                    "  cd third_party && git clone https://github.com/Lu-Feng/CricaVPR.git cricavpr"
                )
            sys.path.insert(0, str(cricavpr_path))

            from network import CricaVPRNet

            # Force standard attention path: xformers memory_efficient_attention
            # does not support float32 on RTX 50-series (compute capability 12.x).
            import backbone.dinov2.attention as _attn_mod
            _attn_mod.XFORMERS_AVAILABLE = False

            # Determine device
            device = self.config.get("device", self.DEFAULT_CONFIG["device"])
            if device == "cuda" and not torch.cuda.is_available():
                print("CricaVPRRetrieval: CUDA requested but not available, falling back to CPU.")
                device = "cpu"
            self.device = device

            # Build the model: DINOv2 backbone with pretrained foundation weights
            self.model = CricaVPRNet(
                pretrained_foundation=True,
                foundation_model_path=str(dinov2_weights),
            )

            # Load CricaVPR trained weights.
            # The checkpoint stores state under "model_state_dict" with DataParallel "module." prefix.
            checkpoint = torch.load(str(weights_path), map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Strip "module." prefix from DataParallel wrapping
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval().to(self.device)
        finally:
            # Restore original torch.load
            torch.load = _orig_torch_load

    def encode(self, image: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms.functional as TF

        # Handle grayscale input
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # Numpy HxWxC uint8 -> tensor CxHxW float [0, 1]
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Resize to model input size
        input_size = self.config.get("input_size", self.DEFAULT_CONFIG["input_size"])
        image_t = TF.resize(image_t, [input_size, input_size], antialias=True)

        # Normalize with ImageNet statistics
        mean = self.config.get("mean", self.DEFAULT_CONFIG["mean"])
        std = self.config.get("std", self.DEFAULT_CONFIG["std"])
        image_t = TF.normalize(image_t, mean=mean, std=std)

        # Add batch dimension
        image_t = image_t.unsqueeze(0).to(self.device)

        # Forward pass – model already L2-normalizes internally,
        # but we re-normalize for safety
        with torch.no_grad():
            desc = self.model(image_t)

        # Handle different output shapes: (1, D), (D,), (1, D, 1) etc.
        if isinstance(desc, torch.Tensor):
            desc = desc.squeeze()
        desc = desc.cpu().numpy().astype(np.float32)

        # L2-normalize
        desc = desc / (np.linalg.norm(desc) + 1e-12)

        return desc
