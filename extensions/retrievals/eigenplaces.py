"""
EigenPlaces — 基于特征分解的视觉地点识别 (ICCV 2023)

使用 ResNet50 骨干网络 + GeM (Generalized-Mean) 池化层，
在大规模地理标注数据上训练，生成 2048 维 L2 归一化全局描述符。
通过场景特征分解学习通用视觉场所表示。

参考: Berton et al., "EigenPlaces: Training Viewpoint Robust Models
      for Visual Place Recognition", ICCV 2023
"""

from extensions.retrievals.base import BaseRetrieval
from extensions import register_retrieval
import numpy as np


def _build_eigenplaces(backbone: str, fc_output_dim: int):
    """
    直接构建 GeoLocalizationNet — 不依赖 torch.hub 下载 CosPlace。

    使用 torchvision 的预训练骨干网络，在其上添加 L2Norm + GeM + FC 层。
    由于 EigenPlaces 发布的 checkpoint 已包含所有权重，无需额外下载。

    Args:
        backbone: ResNet50|ResNet101|ResNet152|VGG16
        fc_output_dim: 最终描述子维度 (EigenPlaces: 2048)

    Returns:
        GeoLocalizationNet 模型实例（未加载权重）
    """
    import torch
    import torchvision
    from torch import nn

    # 各骨干网络最后一个卷积层的输出通道数
    CHANNELS_NUM_IN_LAST_CONV = {
        "ResNet18": 512, "ResNet50": 2048, "ResNet101": 2048,
        "ResNet152": 2048, "VGG16": 512,
    }
    assert backbone in CHANNELS_NUM_IN_LAST_CONV

    # 获取 torchvision 预训练骨干，去掉最后的池化和 FC 层
    tv_model = getattr(torchvision.models, backbone.lower())()

    if backbone.startswith("ResNet"):
        layers = list(tv_model.children())[:-2]  # 去掉 avgpool 和 fc
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone]
    elif backbone == "VGG16":
        layers = list(tv_model.features.children())[:-2]
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone]

    encoder = nn.Sequential(*layers)

    # 导入 EigenPlaces 的聚合层
    from layers import Flatten, L2Norm, GeM

    class GeoLocalizationNet(nn.Module):
        """
        EigenPlaces 的特征聚合网络：
        L2Norm → GeM Pooling → Flatten → FC → L2Norm
        """
        def __init__(self):
            super().__init__()
            self.backbone = encoder
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),                              # Generalized-Mean 池化
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm(),                           # 输出 L2 归一化
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.aggregation(x)
            return x

    return GeoLocalizationNet()


@register_retrieval("EigenPlacesRetrieval")
class EigenPlacesRetrieval(BaseRetrieval):
    """
    XRLocalization 框架适配器 — 将 EigenPlaces 封装为标准检索器接口。

    继承 BaseRetrieval，实现 _load_model() 和 encode() 两个方法，
    使 EigenPlaces 可以无缝接入分层定位流水线的图像检索阶段。
    """

    DEFAULT_CONFIG = {
        "backbone": "ResNet50",          # 骨干网络
        "fc_output_dim": 2048,           # 描述子维度
        "device": "cuda",
        "input_size": 320,               # 模型输入分辨率
        "mean": [0.485, 0.456, 0.406],   # ImageNet 均值
        "std": [0.229, 0.224, 0.225],    # ImageNet 标准差
    }

    def _load_model(self):
        """
        加载 EigenPlaces 预训练模型和权重。

        从 third_party/eigenplaces/weights/*.pth 加载预训练权重。
        """
        import torch
        import sys
        from pathlib import Path

        # 设备自动降级
        device = self.config.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        backbone = self.config.get("backbone", "ResNet50")
        fc_output_dim = self.config.get("fc_output_dim", 2048)

        # 添加 EigenPlaces 源码路径到 sys.path
        base_path = Path(__file__).resolve().parents[2] / "vendor" / "eigenplaces"
        sys.path.insert(0, str(base_path))

        # 直接构建模型 — 避免 CosPlace torch.hub 下载
        self.model = _build_eigenplaces(backbone, fc_output_dim)

        # 加载预训练权重
        weights_path = base_path / "weights"
        pth_files = (list(weights_path.glob("*.pth"))
                     if weights_path.exists() else [])
        if pth_files:
            state = torch.load(pth_files[0], map_location="cpu",
                               weights_only=True)
            self.model.load_state_dict(state)
        else:
            raise RuntimeError(
                "EigenPlaces weights not found. Download the pretrained "
                "weights from:\n"
                "  https://github.com/gmberton/EigenPlaces/releases\n"
                f"  and place the .pth file in {weights_path}/"
            )

        self.model.eval().to(self.device)

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        将输入图像编码为 L2 归一化全局描述符。

        Args:
            image: uint8 NumPy 数组, (H, W) 灰度或 (H, W, 3) 彩色

        Returns:
            float32 (fc_output_dim,) — L2 归一化全局描述符向量
        """
        import torch
        import torchvision.transforms.functional as TF

        # 灰度 → 三通道
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # uint8 H×W×C → float32 C×H×W [0, 1]
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 缩放到模型输入尺寸
        input_size = self.config.get("input_size", 320)
        image_t = TF.resize(image_t, [input_size, input_size], antialias=True)

        # ImageNet 标准化
        mean = self.config.get("mean", [0.485, 0.456, 0.406])
        std = self.config.get("std", [0.229, 0.224, 0.225])
        image_t = TF.normalize(image_t, mean=mean, std=std)

        # 添加 batch 维度 → (1, C, H, W)
        image_t = image_t.unsqueeze(0).to(self.device)

        # GPU 前向推理
        with torch.no_grad():
            desc = self.model(image_t)

        # 将输出转为 1D NumPy 数组并 L2 归一化
        if isinstance(desc, torch.Tensor):
            desc = desc.squeeze()
        desc = desc.cpu().numpy().astype(np.float32)
        desc = desc / (np.linalg.norm(desc) + 1e-12)

        return desc
