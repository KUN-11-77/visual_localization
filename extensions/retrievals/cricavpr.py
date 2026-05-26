"""
CricaVPR — 跨图像相关性感知的视觉地点识别 (CVPR 2024)

核心创新：在检索时对 Top-K 候选图像进行交叉注意力编码，
使描述符之间相互感知，增强细粒度区分能力。
骨干网络：DINOv2 ViT-B/14 + 空间金字塔池化 + 跨图像 Transformer 编码器，
输出 10752 维 (14×768) L2 归一化描述符。

参考: Fan et al., "CricaVPR: Cross-Image Correlation-Aware
      Representation Learning for Visual Place Recognition", CVPR 2024
"""

from extensions.retrievals.base import BaseRetrieval
from extensions import register_retrieval
import numpy as np


@register_retrieval("CricaVPRRetrieval")
class CricaVPRRetrieval(BaseRetrieval):
    """
    XRLocalization 框架适配器 — 将 CricaVPR 封装为标准检索器接口。

    使用 DINOv2 ViT-B/14 骨干网络 + 跨图像自注意力编码器。
    需要两个权重文件: CricaVPR.pth (训练权重) + dinov2_vitb14_pretrain.pth (骨干基础权重)
    """

    DEFAULT_CONFIG = {
        "backbone": "dinov2_vitb14",
        "weights_path": "weights/CricaVPR.pth",
        "dinov2_weights": "weights/dinov2_vitb14_pretrain.pth",
        "device": "cuda",
        "input_size": 224,               # ViT 标准输入尺寸
        "mean": [0.485, 0.456, 0.406],   # ImageNet 均值
        "std": [0.229, 0.224, 0.225],    # ImageNet 标准差
    }

    def _load_model(self):
        """
        加载 CricaVPR 模型。

        处理三个兼容性要点：
        1. pytorch >= 2.6 的 torch.load 默认 weights_only=True，需 patch
        2. RTX 50 系列 (compute capability 12.x) 不支持 xformers float32
        3. DataParallel 包装的权重 state_dict 需去除 'module.' 前缀
        """
        import torch
        import sys
        from pathlib import Path

        # === 解析权重路径 ===
        project_root = Path(__file__).resolve().parents[2]

        weights_path = Path(self.config.get(
            "weights_path", self.DEFAULT_CONFIG["weights_path"]))
        dinov2_weights = Path(self.config.get(
            "dinov2_weights", self.DEFAULT_CONFIG["dinov2_weights"]))
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        if not dinov2_weights.is_absolute():
            dinov2_weights = project_root / dinov2_weights

        # 验证权重文件存在
        missing = []
        if not weights_path.exists():
            missing.append(
                f"  - CricaVPR weights: {weights_path}\n"
                f"    Download from: https://github.com/Lu-Feng/CricaVPR/"
                f"releases/download/v1.0/CricaVPR.pth")
        if not dinov2_weights.exists():
            missing.append(
                f"  - DINOv2 backbone weights: {dinov2_weights}\n"
                f"    Download from: https://dl.fbaipublicfiles.com/dinov2/"
                f"dinov2_vitb14/dinov2_vitb14_pretrain.pth")
        if missing:
            raise FileNotFoundError(
                "CricaVPRRetrieval: missing weight file(s).\n\n"
                + "\n".join(missing) + "\n\n"
                "Expected location: " + str(project_root / "weights") + "\n"
            )

        # === Patch torch.load: PyTorch >= 2.6 默认 weights_only=True ===
        # CricaVPR 网络加载 DINOv2 backbone 时调用 torch.load 不带
        # weights_only=False，需全局 patch
        _orig_torch_load = torch.load
        def _patched_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_torch_load(*a, **kw)
        torch.load = _patched_load

        try:
            # === 添加 CricaVPR 源码路径 ===
            cricavpr_path = project_root / "vendor" / "cricavpr"
            sys.path.insert(0, str(cricavpr_path))

            from network import CricaVPRNet

            # === 禁用 xformers: RTX 50 系 (sm_120) 不支持 float32 ===
            import backbone.dinov2.attention as _attn_mod
            _attn_mod.XFORMERS_AVAILABLE = False

            # === 设备选择 ===
            device = self.config.get("device", self.DEFAULT_CONFIG["device"])
            if device == "cuda" and not torch.cuda.is_available():
                print("CricaVPRRetrieval: CUDA requested but not available, "
                      "falling back to CPU.")
                device = "cpu"
            self.device = device

            # === 构建模型 ===
            self.model = CricaVPRNet(
                pretrained_foundation=True,
                foundation_model_path=str(dinov2_weights),
            )

            # === 加载 CricaVPR 训练权重 ===
            checkpoint = torch.load(str(weights_path), map_location="cpu")
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # 去除 DataParallel 的 'module.' 前缀
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", ""): v
                    for k, v in state_dict.items()
                }

            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval().to(self.device)

        finally:
            # 恢复原始 torch.load（避免影响其他模块）
            torch.load = _orig_torch_load

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        将输入图像编码为 L2 归一化全局描述符。

        与 EigenPlaces 使用相同的预处理流水线：
        uint8 → float32 [0,1] → resize → ImageNet 标准化 → 前向推理

        Args:
            image: uint8 NumPy 数组, (H, W) 灰度或 (H, W, 3) 彩色

        Returns:
            float32 (10752,) — L2 归一化跨图像感知的描述符
        """
        import torch
        import torchvision.transforms.functional as TF

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        input_size = self.config.get("input_size",
                                     self.DEFAULT_CONFIG["input_size"])
        image_t = TF.resize(image_t, [input_size, input_size], antialias=True)

        mean = self.config.get("mean", self.DEFAULT_CONFIG["mean"])
        std = self.config.get("std", self.DEFAULT_CONFIG["std"])
        image_t = TF.normalize(image_t, mean=mean, std=std)

        image_t = image_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            desc = self.model(image_t)

        if isinstance(desc, torch.Tensor):
            desc = desc.squeeze()
        desc = desc.cpu().numpy().astype(np.float32)
        desc = desc / (np.linalg.norm(desc) + 1e-12)

        return desc
