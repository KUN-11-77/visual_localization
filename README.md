# Visual Localization — 基于分层策略的视觉定位系统

计算摄影学课程大作业项目。基于 OpenXRLab XRLocalization 框架扩展，实现了一个完整的
**检索 → 检测 → 匹配 → 位姿估计** 四阶段分层视觉定位流水线。

---

### 项目概述

本项目实现了一个模块化的视觉定位流水线，将传统"检测+匹配"两阶段方案扩展为四阶段分层架构：

```
查询图像 → [图像检索] → Top-K候选帧 → [特征检测] → [特征匹配] → [PnP位姿估计] → 相机位姿
```

在此基础上，探索了各阶段不同算法组合对定位精度的影响，并引入 SfM 稀疏重建进行 2D-3D 重投影优化。

### 工作量概览

| 类别 | 内容 | 文件数 |
|------|------|--------|
| 扩展模块 | EigenPlaces、CricaVPR、ALIKED、LightGlue 四种算法的代码集成 | 8 个 `.py` |
| 核心脚本 | 流水线、评估、可视化、AR 演示、SfM 重建、交互报告 | 19 个 `.py` |
| 配置文件 | 2 个数据集 × 多种算法组合 | 13 个 `.yaml` |
| Web 界面 | Flask 单页应用 + Jinja2 模板 | 2 个 `.py` + 2 个 `.html` |
| LaTeX 报告 | 完整实验报告（含图表、代码块、架构图） | 1 个 `.tex` |
| 图表 | PDF + PNG 格式的实验图表 | ~12 个文件 |
| 文档 | README、安装脚本 | 3 个文件 |


#### 1. 最小化运行测试（约 5 分钟，无需 GPU）

```bash
# 安装依赖（仅首次）
pip install -r requirements.txt

# 启动 Web 界面（无需数据集即可浏览）
python scripts/web_ui.py --port 5000
# 浏览器打开 http://127.0.0.1:5000
```

Web 界面包含 Dashboard、实验运行、结果查看、AR Demo、导出五个模块。**无需下载数据集即可浏览界面结构和功能。**

#### 2. 完整实验复现（需要 Cambridge 数据集，约 30 分钟）

```bash
# 步骤 1: 下载数据集
python scripts/download_shopfacade.py    # ShopFacade (~2.9 GB)
# KingsCollege 数据集需手动下载，详见下方"数据集准备"

# 步骤 2: 下载模型权重 (~1.7 GB)
python scripts/download_weights.py

# 步骤 3: 运行实验
python scripts/run_pipeline.py --config configs/baseline_a.yaml --limit_queries 343 --build_sfm

# 步骤 4: 查看结果
python scripts/generate_report.py --results_dir outputs/results/baseline_a/ --config configs/baseline_a.yaml
```

#### 3. 代码

核心改动集中在：
- [extensions/](extensions/) — 四个算法模块的集成代码
- [scripts/run_pipeline.py](scripts/run_pipeline.py) — 主流水线入口
- [scripts/web_ui.py](scripts/web_ui.py) — Flask Web 界面
- [configs/](configs/) — 所有实验配置

#### 4. 查看已有结果（已有结果存放在experiment_report 和 outputs，其中outputs/ar_demo有ar演示视频）

```bash
python scripts/visualize_results.py    # 生成可视化图表
python scripts/generate_figures.py     # 生成报告用图（PDF + PNG）
```


## 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.x / 12.x（可选，CPU 也可运行但较慢）
- Windows / Linux / macOS

### 一键安装

```bash
# Windows
setup.bat

# Linux / macOS
bash setup.sh
```

安装脚本会自动完成：
1. 安装所有 Python 依赖（`pip install -r requirements.txt`）
2. 下载所需模型权重（约 1.7 GB）

### 手动安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 安装 PyTorch（根据你的 CUDA 版本选择）
# CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 下载模型权重
python scripts/download_weights.py

# 4. 查看所有权重文件列表
python scripts/download_weights.py --list
```

---

## 项目结构

```
visual_localization/
├── extensions/                  # 扩展模块（核心代码）
│   ├── __init__.py              # 统一构建入口：build_retrieval/detector/matcher()
│   ├── retrievals/              # 图像检索模块
│   │   ├── __init__.py
│   │   ├── base.py              # 检索器抽象基类
│   │   ├── netvlad.py           # NetVLAD (Arandjelović et al., PAMI 2017)
│   │   ├── eigenplaces.py       # EigenPlaces (Berton et al., ICCV 2023)
│   │   └── cricavpr.py          # CricaVPR (Fan et al., CVPR 2024)
│   ├── detectors/               # 特征检测模块
│   │   ├── __init__.py
│   │   ├── base.py              # 检测器抽象基类
│   │   ├── sift.py              # SIFT (OpenCV 实现)
│   │   ├── superpoint.py        # SuperPoint (DeTone et al., CVPRW 2018)
│   │   └── aliked.py            # ALIKED (Wang et al., T-IM 2023)
│   └── matchers/                # 特征匹配模块
│       ├── __init__.py
│       ├── base.py              # 匹配器抽象基类
│       ├── nn.py                # 最近邻匹配 (Lowe's ratio test)
│       ├── superglue.py         # SuperGlue (Sarlin et al., CVPR 2020)
│       └── lightglue.py         # LightGlue (Lindenberger et al., ICCV 2023)
├── vendor/                      # 第三方源码（精选必要文件，约 600 KB）
│   ├── netvlad/                 # NetVLAD 模型定义
│   ├── superglue/               # SuperGlue + SuperPoint 模型定义
│   ├── lightglue/               # LightGlue 库（本地备选）
│   ├── eigenplaces/             # EigenPlaces 辅助层
│   ├── cricavpr/                # CricaVPR 骨干网络
│   └── hloc/                    # HLoc 几何工具
├── scripts/                     # 脚本（19 个文件）
│   ├── run_pipeline.py          # ★ 主流水线入口
│   ├── web_ui.py                # ★ Web 交互界面（Flask 单页应用）
│   ├── evaluate.py              # 评估指标计算（召回率、角度误差）
│   ├── build_sfm.py             # ★ SuperPoint+SuperGlue SfM 三角化重建
│   ├── ar_demo.py               # ★ AR 演示生成（3D 定位线框叠加）
│   ├── download_weights.py      # 模型权重下载
│   ├── download_shopfacade.py   # ShopFacade 数据集下载
│   ├── generate_figures.py      # 报告图表生成（PDF + PNG）
│   ├── generate_report.py       # 交互式 HTML 报告生成
│   ├── visualize_results.py     # 结果可视化（召回率柱状图等）
│   ├── export_submission.py     # ★ 打包提交文件（zip）
│   ├── nvm_model.py             # NVM 格式 3D 模型解析
│   ├── colmap_model.py          # COLMAP 模型 IO 工具
│   ├── colmap_localization.py   # COLMAP 定位模块
│   ├── timing.py                # 计时工具（装饰器）
│   ├── ui.py                    # 控制台 UI 辅助
│   ├── ui_web.py                # Web UI 路由模块
│   ├── debug_single_query.py    # 单帧调试工具
│   └── debug_colmap.py          # COLMAP 调试工具
├── configs/                     # 实验配置文件（13 个 YAML）
│   ├── baseline_a.yaml          # NetVLAD + SIFT + NN（官方 Baseline A）
│   ├── baseline_b.yaml          # NetVLAD + SuperPoint + SuperGlue（官方 Baseline B）
│   ├── exp_retrieval.yaml       # EigenPlaces + SuperPoint + SuperGlue（检索端替换）
│   ├── exp_match.yaml           # NetVLAD + ALIKED + LightGlue（匹配端替换）
│   ├── exp_full.yaml            # EigenPlaces + ALIKED + LightGlue（全链路替换）
│   ├── exp_crica.yaml           # CricaVPR + ALIKED + LightGlue（第三种检索器）
│   ├── 7scenes_stairs_baseline.yaml       # 7-Scenes Baseline
│   ├── 7scenes_stairs_eigenplaces.yaml    # 7-Scenes + EigenPlaces
│   ├── shopfacade_baseline_b.yaml         # ShopFacade Baseline B
│   ├── shopfacade_exp_retrieval.yaml      # ShopFacade 检索端实验
│   ├── shopfacade_exp_match.yaml          # ShopFacade 匹配端实验
│   ├── shopfacade_exp_full.yaml           # ShopFacade 全链路替换
│   └── shopfacade_exp_crica.yaml          # ShopFacade + CricaVPR
├── templates/                   # Web 界面模板
│   ├── index.html               # 实验运行 + 结果展示页面
│   └── showcase.html            # 结果展示页面
├── figures/                     # 报告用图表（PDF + PNG，约 12 个文件）
│   ├── 7scenes_comparison.pdf
│   ├── ablation_study.pdf
│   ├── cross_dataset.pdf
│   ├── recall_comparison.pdf
│   ├── timing_comparison.pdf
│   ├── cambridge_vs_shopfacade.pdf
│   ├── shopfacade_recall_comparison.pdf
│   ├── shopfacade_ar_demo.png
│   └── ... (对应的 .png 文件)
├── experiment_report/           # 实验报告输出目录
├── outputs/                     # 实验结果输出（运行时生成）
│   ├── results/                 # 按配置名分类的 CSV 结果 + 位姿 JSON
│   ├── ar_demo/                 # AR 演示图片和视频
│   └── reports/                 # 交互式 HTML 报告
├── experiment_report.tex        # ★ 实验报告（LaTeX 源码，约 30 页）
├── requirements.txt             # Python 依赖列表
├── setup.bat                    # Windows 一键安装
├── setup.sh                     # Linux/macOS 一键安装
└── README.md                    # 本文件
```


## 数据集准备

### Cambridge Landmarks

```bash
# ShopFacade（自动下载，约 2.9 GB）
python scripts/download_shopfacade.py

# KingsCollege（手动下载）
# 从 https://www.repository.cam.ac.uk/handle/1810/251342 下载
# 解压到 data/cambridge/KingsCollege/
```

数据目录结构：

```
data/cambridge/KingsCollege/
├── dataset_train.txt          # 训练（数据库）图像列表
├── dataset_test.txt           # 测试（查询）图像列表
├── reconstruction.nvm          # VisualSFM 重建模型
├── *.jpg                       # 图像文件
└── ...
```

### 7-Scenes（室内数据集）

从 [Microsoft 7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) 下载，解压到 `data/7scenes/`。

---

## 运行实验

### Web 交互界面（推荐）

```bash
python scripts/web_ui.py --port 5000
# 浏览器打开 http://127.0.0.1:5000
```

Web 界面提供 5 个功能模块：

| 模块 | 功能 |
|------|------|
| **Dashboard** | 实验总览，汇总所有已完成实验的结果 |
| **Run Experiment** | 选择配置文件运行实验，实时查看终端日志 |
| **Results Viewer** | 按帧查看定位误差、检索命中情况 |
| **AR Demo** | 生成和查看 AR 增强现实演示 |
| **Export** | 导出 CSV/JSON 结果、打包提交 ZIP 文件 |

### 命令行

```bash
# 快速测试（5 帧，验证环境）
python scripts/run_pipeline.py \
    --config configs/baseline_a.yaml \
    --limit_queries 5

# 运行完整实验（343 帧 + SfM 三角化）
python scripts/run_pipeline.py \
    --config configs/baseline_a.yaml \
    --limit_queries 343 \
    --build_sfm

# 带参数覆盖
python scripts/run_pipeline.py \
    --config configs/baseline_a.yaml \
    --overrides retrieval.top_k=20 dataset.root=data/cambridge/KingsCollege

# 仅输出位姿（跳过 CSV）
python scripts/run_pipeline.py \
    --config configs/baseline_a.yaml \
    --poses_only
```

### 切换数据集

编辑对应 YAML 配置文件中的 `dataset` 字段：

```yaml
dataset:
  name: cambridge          # 可选: cambridge, 7scenes, aachen
  root: "/path/to/your/dataset"
  scene: "KingsCollege"    # 场景名（如 ShopFacade, Stairs）
```

---

## 可用配置与实验设计

### 消融实验矩阵

本项目的实验设计遵循**控制变量法**，通过逐一替换流水线中的单个模块，量化每个改进的贡献：

| 配置文件 | 检索 | 检测 | 匹配 | 实验目的 |
|----------|------|------|------|----------|
| `baseline_a.yaml` | NetVLAD | SIFT | NN | 官方 Baseline A（传统方案） |
| `baseline_b.yaml` | NetVLAD | SuperPoint | SuperGlue | 官方 Baseline B（学习方案） |
| `exp_retrieval.yaml` | **EigenPlaces** | SuperPoint | SuperGlue | 检索端替换 → 量化 EigenPlaces 增益 |
| `exp_match.yaml` | NetVLAD | **ALIKED** | **LightGlue** | 匹配端替换 → 量化 ALIKED+LG 增益 |
| `exp_full.yaml` | **EigenPlaces** | **ALIKED** | **LightGlue** | 全链路替换 → 叠加效应验证 |
| `exp_crica.yaml` | **CricaVPR** | ALIKED | LightGlue | 第三种检索器 → 检索器泛化性对比 |

### 多数据集配置

| 配置文件 | 数据集 | 场景类型 |
|----------|--------|----------|
| `7scenes_stairs_baseline.yaml` | 7-Scenes | 室内小场景 |
| `7scenes_stairs_eigenplaces.yaml` | 7-Scenes | 室内小场景 |
| `shopfacade_baseline_b.yaml` | Cambridge | 室外建筑立面（新增） |
| `shopfacade_exp_retrieval.yaml` | Cambridge | 室外建筑立面 |
| `shopfacade_exp_match.yaml` | Cambridge | 室外建筑立面 |
| `shopfacade_exp_full.yaml` | Cambridge | 室外建筑立面 |
| `shopfacade_exp_crica.yaml` | Cambridge | 室外建筑立面 |

---

## 实验结果

### Cambridge KingsCollege（343 帧）

| 配置 | (0.25m, 2°) | (0.5m, 5°) | (5.0m, 10°) |
|------|:---:|:---:|:---:|
| BL-A (NetVLAD+SIFT+NN) | 21.87% | 45.48% | 87.46% |
| BL-B (NetVLAD+SP+SG) | 47.23% | 81.05% | 99.13% |
| BL-B + triangulation | **69.68%** | **93.00%** | **100%** |
| EXP-M (ALIKED+LG) | 67.35% | 93.29% | 100% |
| EXP-Full (EP+ALIKED+LG) | 65.01% | 91.84% | 99.71% |

### 7-Scenes Stairs（1000 帧）

| 配置 | (0.25m, 2°) | (0.5m, 5°) | (5.0m, 10°) |
|------|:---:|:---:|:---:|
| NetVLAD+ALIKED+LG | 30.60% | 84.70% | 97.10% |
| EigenPlaces+ALIKED+LG | 34.10% | 86.50% | 97.20% |

### 关键结论

1. **SfM 三角化** 显著提升精度：BL-B 在高精度阈值 (0.25m, 2°) 上从 47.23% → 69.68%（+22.45pp）
2. **ALIKED+LightGlue** 匹敌 SuperPoint+SuperGlue，同时推理速度更快（~2.5× speedup）
3. **EigenPlaces** 在室内场景（7-Scenes）上比 NetVLAD 有优势，在室外场景上持平
4. **全链路替换** (EXP-Full) 可达到与官方 Baseline B 相当的精度，证明了模块化替换的可行性

---

## 生成 AR Demo

AR Demo 将估计的相机位姿叠加到 3D 点云上，生成增强现实效果图：

```bash
# 先运行实验生成 pred_poses.json，然后：
python scripts/ar_demo.py \
    --config configs/baseline_a.yaml \
    --poses outputs/results/baseline_a/pred_poses.json \
    --output outputs/ar_demo/baseline_a/ \
    --limit 30 --fps 5
```

输出包括逐帧 AR 渲染图和合成的 MP4 视频。

---

## 扩展模块设计

所有扩展模块遵循统一的抽象基类接口，通过工厂函数构建：

```python
from extensions import build_retrieval, build_detector, build_matcher

# 配置驱动构建
retrieval = build_retrieval(cfg["retrieval"])   # → BaseRetrieval
detector  = build_detector(cfg["detector"])      # → BaseDetector
matcher   = build_matcher(cfg["matcher"])         # → BaseMatcher
```

### 模块接口

```python
# 检索器
class BaseRetrieval:
    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray
    def retrieve(self, query_desc, db_descs, top_k: int) -> list[int]

# 检测器
class BaseDetector:
    def detect(self, image: np.ndarray) -> dict  # {keypoints, descriptors, scores}

# 匹配器
class BaseMatcher:
    def match(self, desc1, desc2, kpts1, kpts2, im1_shape, im2_shape) -> dict
```

### 集成的四种算法

| 算法 | 类别 | 论文 | 集成文件 |
|------|------|------|----------|
| EigenPlaces | 检索 | Berton et al., ICCV 2023 | [extensions/retrievals/eigenplaces.py](extensions/retrievals/eigenplaces.py) |
| CricaVPR | 检索 | Fan et al., CVPR 2024 | [extensions/retrievals/cricavpr.py](extensions/retrievals/cricavpr.py) |
| ALIKED | 检测 | Wang et al., T-IM 2023 | [extensions/detectors/aliked.py](extensions/detectors/aliked.py) |
| LightGlue | 匹配 | Lindenberger et al., ICCV 2023 | [extensions/matchers/lightglue.py](extensions/matchers/lightglue.py) |

---

## 模型权重说明

所有权重文件通过 `scripts/download_weights.py` 自动下载，存放于 `vendor/*/weights/` 和 `weights/` 目录：

| 权重文件 | 用途 | 大小 |
|----------|------|------|
| `Pitts30K_struct.mat` | NetVLAD 检索 | 529 MB |
| `CricaVPR.pth` | CricaVPR 检索 | 562 MB |
| `dinov2_vitb14_pretrain.pth` | CricaVPR 骨干网络 | 331 MB |
| `resnet50_2048_eigenplaces.pth` | EigenPlaces 检索 | 106 MB |
| `aliked-n16.pth` | ALIKED 检测器 | 3 MB |
| `aliked_lightglue.pth` | LightGlue (ALIKED) | 46 MB |
| `superpoint_lightglue.pth` | LightGlue (SuperPoint) | 46 MB |
| `superpoint_v1.pth` | SuperPoint 检测器 | 5 MB |
| `superglue_indoor.pth` / `superglue_outdoor.pth` | SuperGlue 匹配器 | 46 MB ×2 |

---

## 参考文献

- Sarlin et al., "From Coarse to Fine: Robust Hierarchical Localization at Large Scale", CVPR 2019
- Arandjelović et al., "NetVLAD: CNN Architecture for Weakly Supervised Place Recognition", PAMI 2017
- Berton et al., "EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition", ICCV 2023
- Fan et al., "CricaVPR: Cross-Image Correlation-Aware Representation Learning for Visual Place Recognition", CVPR 2024
- Wang et al., "ALIKED: A Lighter Keypoint and Descriptor Extraction Network", T-IM 2023
- Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed", ICCV 2023
- DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPRW 2018
- Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural Networks", CVPR 2020

---

## 分工声明

本项目为单人独立完成，包含以下全部工作：
- 算法调研与方案设计
- 四个扩展模块（EigenPlaces、CricaVPR、ALIKED、LightGlue）的代码集成与调试
- XRLocalization 框架修改和 Bug 修复
- SuperPoint+SuperGlue SfM 三角化管线实现
- Flask Web 交互界面开发
- AR Demo 渲染管线实现
- 多数据集实验运行（Cambridge KingsCollege、ShopFacade、7-Scenes）
- 消融实验设计与数据分析
- 交互式 HTML 报告生成
- 项目报告撰写（LaTeX，约 30 页）
