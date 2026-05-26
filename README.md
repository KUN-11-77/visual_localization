# Visual Localization — 基于分层策略的视觉定位系统

计算摄影学课程大作业项目。基于 OpenXRLab XRLocalization 框架扩展，实现了一个完整的
**检索 → 检测 → 匹配 → 位姿估计** 四阶段分层视觉定位流水线。

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

## 项目结构

```
├── extensions/              # 扩展模块（核心代码）
│   ├── retrievals/          # 图像检索：NetVLAD, EigenPlaces, CricaVPR
│   ├── detectors/           # 特征检测：SIFT, SuperPoint, ALIKED
│   └── matchers/            # 特征匹配：NN, SuperGlue, LightGlue
├── vendor/                  # 第三方源码（精选必要文件）
│   ├── netvlad/             # NetVLAD 模型定义
│   ├── superglue/           # SuperGlue + SuperPoint 模型定义
│   ├── lightglue/           # LightGlue 库（本地备选）
│   ├── eigenplaces/         # EigenPlaces 辅助层
│   ├── cricavpr/            # CricaVPR 骨干网络
│   └── hloc/                # HLoc 几何工具
├── scripts/                 # 脚本
│   ├── run_pipeline.py      # 主流水线入口
│   ├── web_ui.py            # Web 交互界面（Flask）
│   ├── ar_demo.py           # AR 演示生成
│   ├── download_weights.py  # 模型权重下载
│   ├── generate_figures.py  # 报告图表生成
│   ├── build_sfm.py         # COLMAP SfM 重建
│   └── evaluate.py          # 结果评估
├── configs/                 # 实验配置文件（YAML）
├── figures/                 # 报告用图表（PDF + PNG）
├── outputs/                 # 实验结果输出（运行时生成）
│   ├── results/             # CSV 结果 + 位姿
│   └── ar_demo/             # AR 演示图片和视频
├── experiment_report.tex    # 实验报告（LaTeX 源码）
├── requirements.txt         # Python 依赖
├── setup.bat / setup.sh     # 一键安装脚本
└── README.md
```

## 运行实验

### Web 交互界面（推荐）

```bash
python scripts/web_ui.py --port 5000
# 浏览器打开 http://127.0.0.1:5000
```

Web 界面提供 5 个功能模块：
- **Dashboard** — 实验总览
- **Run Experiment** — 选择配置文件运行实验，实时查看日志
- **Results Viewer** — 按帧查看定位误差
- **AR Demo** — 生成和查看 AR 演示
- **Export** — 导出结果、打包提交文件

### 命令行

```bash
# 运行单个实验（限制 5 帧快速测试）
python scripts/run_pipeline.py \
    --config configs/baseline_a.yaml \
    --limit_queries 5

# 运行完整实验（Cambridge KingsCollege）
python scripts/run_pipeline.py \
    --config configs/baseline_a.yaml \
    --limit_queries 343 \
    --build_sfm
```

### 切换数据集

编辑对应 YAML 配置文件中的 `dataset.root` 路径：

```yaml
dataset:
  name: cambridge          # 或 7scenes
  root: "/path/to/your/dataset"
  scene: "KingsCollege"    # 场景名
```

## 可用配置

| 配置文件 | 检索 | 检测 | 匹配 | 说明 |
|----------|------|------|------|------|
| `baseline_a.yaml` | NetVLAD | SIFT | NN | 官方 Baseline A |
| `baseline_b.yaml` | NetVLAD | SuperPoint | SuperGlue | 官方 Baseline B |
| `exp_retrieval.yaml` | EigenPlaces | SuperPoint | SuperGlue | 检索端替换 |
| `exp_match.yaml` | NetVLAD | ALIKED | LightGlue | 匹配端替换 |
| `exp_full.yaml` | EigenPlaces | ALIKED | LightGlue | 全链路替换 |
| `exp_crica.yaml` | CricaVPR | ALIKED | LightGlue | 第三种检索器 |
| `7scenes_stairs_baseline.yaml` | NetVLAD | ALIKED | LightGlue | 7-Scenes 室内 |
| `7scenes_stairs_eigenplaces.yaml` | EigenPlaces | ALIKED | LightGlue | 7-Scenes 室内 |

## 生成 AR Demo

```bash
# 先运行实验生成 pred_poses.json，然后：
python scripts/ar_demo.py \
    --config configs/baseline_a.yaml \
    --poses outputs/results/baseline_a/pred_poses.json \
    --output outputs/ar_demo/baseline_a/ \
    --limit 30 --fps 5
```

## 实验结果

Cambridge KingsCollege 最佳结果（343 帧）：

| 配置 | (0.25m, 2°) | (0.5m, 5°) | (5.0m, 10°) |
|------|-------------|-------------|-------------|
| BL-B + triangulation | **69.68%** | **93.00%** | 100% |
| EXP-M (ALIKED+LG) | 67.35% | 93.29% | 100% |

7-Scenes Stairs（1000 帧）：

| 配置 | (0.25m, 2°) | (0.5m, 5°) | (5.0m, 10°) |
|------|-------------|-------------|--------------|
| NetVLAD+ALIKED+LG | 30.60% | 84.70% | 97.10% |

## 模型权重说明

所有权重文件通过 `scripts/download_weights.py` 自动下载，包括：

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

## 常见问题

**Q: 运行时报 `OMP: Error #15`？**
```bash
# Windows 上 PyTorch 与 pycolmap 的 OpenMP 冲突，设置环境变量：
set KMP_DUPLICATE_LIB_OK=TRUE
```

**Q: 找不到模型权重？**
运行 `python scripts/download_weights.py` 下载所有权重文件。

**Q: 如何在没有 GPU 的电脑上运行？**
所有模块会自动降级到 CPU。修改 YAML 中 `device: cpu` 可强制使用 CPU。

**Q: lightglue 模块找不到？**
项目 `vendor/lightglue/` 目录包含了 LightGlue 的本地副本。如果需要 pip 版本：
```bash
pip install lightglue  # 注意：Windows 可能不可用
```

## 参考文献

- Sarlin et al., "From Coarse to Fine: Robust Hierarchical Localization at Large Scale", CVPR 2019
- Arandjelović et al., "NetVLAD: CNN Architecture for Weakly Supervised Place Recognition", PAMI 2017
- Berton et al., "EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition", ICCV 2023
- Fan et al., "CricaVPR: Cross-Image Correlation-Aware Representation Learning for Visual Place Recognition", CVPR 2024
- Wang et al., "ALIKED: A Lighter Keypoint and Descriptor Extraction Network", T-IM 2023
- Lindenberger et al., "LightGlue: Local Feature Matching at Light Speed", ICCV 2023
- DeTone et al., "SuperPoint: Self-Supervised Interest Point Detection and Description", CVPRW 2018
- Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural Networks", CVPR 2020

## 分工声明

本项目为单人独立完成，包含以下全部工作：
- 算法调研与方案设计
- 四个扩展模块（EigenPlaces、CricaVPR、ALIKED、LightGlue）的代码集成与调试
- XRLocalization 框架修改和 Bug 修复
- Flask Web 界面开发
- AR Demo 渲染管线实现
- 全部实验运行和数据分析
- 项目报告撰写
