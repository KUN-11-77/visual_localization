"""
ALIKED (A-LIKE Keypoint Detector) — 深度学习关键点检测器

基于可变形卷积 (Deformable Convolution) 和特征金字塔结构，能够在不同
尺度下检测鲁棒的关键点。相比 SIFT 对光照变化和视角变化更鲁棒；
相比 SuperPoint，关键点分布更均匀，在纹理贫乏区域表现更佳。

参考: Wang et al., "ALIKED: A Lighter Keypoint and Descriptor
      Extraction Network via Deformable Transformation", T-IM 2023
"""

from extensions.detectors.base import BaseDetector
from extensions import register_detector
import numpy as np


@register_detector("ALIKEDDetector")
class ALIKEDDetector(BaseDetector):
    """
    XRLocalization 框架适配器 — 将 ALIKED 封装为标准检测器接口。

    继承 BaseDetector，实现 _load_model() 和 detect() 两个方法，
    使 ALIKED 可以无缝接入分层定位流水线。
    """

    # 默认配置 — 可通过 YAML 配置文件覆盖
    DEFAULT_CONFIG = {
        "max_keypoints": 5000,          # 每张图像最大关键点数
        "detection_threshold": 0.2,     # 关键点得分阈值（降低可获取更多点）
        "device": "cuda",               # 推理设备
    }

    def _load_model(self):
        """
        加载 ALIKED 预训练模型。
        优先从 pip 安装的 lightglue 包导入，失败则从本地仓库导入。
        """
        import sys
        import torch
        from pathlib import Path

        try:
            from lightglue import ALIKED
        except ImportError:
            # pip 安装失败 → 回退到本地 LightGlue 源码
            local_path = (Path(__file__).resolve().parents[2]
                          / "vendor" / "lightglue")
            if local_path.exists():
                sys.path.insert(0, str(local_path))
                from lightglue import ALIKED
            else:
                raise ImportError(
                    "lightglue not installed. Install with:\n"
                    "  pip install lightglue\n"
                )

        # 设备自动降级: CUDA 不可用时回退 CPU
        self.device = self.config.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # 构建模型并切换到推理模式
        self.model = ALIKED(
            max_num_keypoints=self.config.get("max_keypoints", 5000),
            detection_threshold=self.config.get("detection_threshold", 0.2),
        ).eval()
        self.model = self.model.to(self.device)

    def detect(self, image: np.ndarray):
        """
        对输入图像提取关键点和描述符。

        Args:
            image: uint8 NumPy 数组, (H, W) 灰度或 (H, W, 3) 彩色

        Returns:
            keypoints:   float32 (N, 2) — (x, y) 像素坐标
            descriptors: float32 (N, D) — L2 归一化局部描述符
            scores:      float32 (N,)  — 关键点得分 [0, 1]
        """
        import torch

        # 灰度 → 三通道伪彩
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        # uint8 (H,W,C) → float32 (1,C,H,W) 归一化到 [0, 1]
        image_t = (
            torch.from_numpy(image.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            / 255.0
        )
        image_t = image_t.to(self.device)

        # GPU 端前向推理（无梯度计算）
        with torch.no_grad():
            feats = self.model.extract(image_t)

        # 将 GPU 张量转为 CPU NumPy 数组
        keypoints = feats["keypoints"][0].cpu().numpy().astype(np.float32)
        descriptors = feats["descriptors"][0].cpu().numpy().astype(np.float32)
        scores = feats["keypoint_scores"][0].cpu().numpy().astype(np.float32)

        # L2 归一化描述符（保证余弦相似度计算等价于内积）
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)  # 防除零
        descriptors = descriptors / norms
        scores = np.clip(scores, 0.0, 1.0)

        return keypoints, descriptors, scores
