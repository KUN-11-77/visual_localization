"""
LightGlue — 轻量级深度特征匹配器 (ICCV 2023)

核心创新：自适应深度 (Adaptive Depth) 和早期退出 (Early Exiting)。
在简单匹配对上仅使用 3--5 层注意力，困难匹配对使用全部 9 层，
从而实现 3--4× 的推理加速，同时保持与 SuperGlue 相当的匹配质量。

参考: Lindenberger et al., "LightGlue: Local Feature Matching
      at Light Speed", ICCV 2023
"""

from extensions.matchers.base import BaseMatcher
from extensions import register_matcher
import numpy as np


@register_matcher("LightGlueMatcher")
class LightGlueMatcher(BaseMatcher):
    """
    XRLocalization 框架适配器 — 将 LightGlue 封装为标准匹配器接口。

    支持多种特征类型: superpoint, aliked, disk, sift。
    通过自适应深度机制在匹配质量和速度之间取得平衡。
    """

    # 默认配置 — 可通过 YAML 配置文件覆盖
    DEFAULT_CONFIG = {
        "features": "superpoint",       # 特征类型: superpoint|aliked|disk|sift
        "depth_confidence": 0.95,       # 自适应深度置信度（越高越早退出）
        "width_confidence": 0.99,       # 双向匹配阈值
        "filter_threshold": 0.1,        # 匹配置信度过滤阈值
        "device": "cuda",               # 推理设备
    }

    def _load_model(self):
        """
        加载 LightGlue 预训练模型。
        优先从 pip 安装的 lightglue 包导入，失败则从本地仓库导入。
        """
        import sys
        import torch
        from pathlib import Path

        try:
            from lightglue import LightGlue
        except ImportError:
            # pip 安装失败 → 回退到本地 LightGlue 源码
            local_path = (Path(__file__).resolve().parents[2]
                          / "vendor" / "lightglue")
            if local_path.exists():
                sys.path.insert(0, str(local_path))
                from lightglue import LightGlue
            else:
                raise ImportError(
                    "lightglue not installed. Install with:\n"
                    "  pip install lightglue\n"
                )

        self.model = LightGlue(
            features=self.config.get("features", "superpoint"),
            depth_confidence=self.config.get("depth_confidence", 0.95),
            width_confidence=self.config.get("width_confidence", 0.99),
            filter_threshold=self.config.get("filter_threshold", 0.1),
        )

        # 设备自动降级
        device = self.config.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model.eval().to(self.device)

    def match(self, query_data, db_data):
        """
        在 Query 和 DB 图像关键点之间建立匹配对应关系。

        Args:
            query_data: dict, 包含 keypoints (N1,2), descriptors (N1,D),
                        image_size (2,) — Query 图像特征
            db_data:    dict, 同上结构 — Database 图像特征

        Returns:
            q_idx:  int64 (M,) — Query 侧匹配点索引
            db_idx: int64 (M,) — DB 侧匹配点索引
            conf:   float32 (M,) — 匹配置信度得分 (0..1)
        """
        import torch

        kpts0 = query_data["keypoints"]
        kpts1 = db_data["keypoints"]
        desc0 = query_data["descriptors"]
        desc1 = db_data["descriptors"]

        # 关键点过少无法匹配（LightGlue 最小需要 2 个点）
        if len(kpts0) < 2 or len(kpts1) < 2:
            return np.array([]), np.array([]), np.array([])

        # 构建 LightGlue 输入字典（batch 维度=1）
        data = {
            "image0": {
                "keypoints": (torch.from_numpy(kpts0[np.newaxis])
                              .float().to(self.device)),
                "descriptors": (torch.from_numpy(desc0[np.newaxis])
                                .float().to(self.device)),
                "image_size": torch.tensor(
                    [query_data.get("image_size", (0, 0))]
                ).to(self.device),
            },
            "image1": {
                "keypoints": (torch.from_numpy(kpts1[np.newaxis])
                              .float().to(self.device)),
                "descriptors": (torch.from_numpy(desc1[np.newaxis])
                                .float().to(self.device)),
                "image_size": torch.tensor(
                    [db_data.get("image_size", (0, 0))]
                ).to(self.device),
            },
        }

        # GPU 端前向推理（自适应深度在此自动决策）
        with torch.no_grad():
            pred = self.model(data)
            matches = pred["matches0"][0].cpu().numpy()       # (N1,)
            scores = pred["matching_scores0"][0].cpu().numpy() # (N1,)

        # matches[i] = j 表示 Query 第 i 个点匹配到 DB 第 j 个点
        # matches[i] = -1 表示第 i 个点未匹配 → 过滤
        valid = matches > -1
        q_idx = np.where(valid)[0]
        db_idx = matches[valid]
        conf = scores[valid]
        return q_idx, db_idx, conf
