# 视觉定位方法设计与实验方案

> 基于 XRLocalization 框架，在 HLoc 三阶段流水线（图像检索 → 特征提取 → 特征匹配）各模块中引入未在 XRLab/HLoc 中集成的新方法，与两条 baseline 进行系统对比与消融分析。

---

## 1. 问题定义与流水线概述

视觉定位（Visual Localization）的任务是：给定一张查询图像，恢复相机在世界坐标系中的 6DoF 位姿（平移向量 $\mathbf{t} \in \mathbb{R}^3$ + 旋转矩阵 $\mathbf{R} \in SO(3)$）。

HLoc（Hierarchical Localization）将这一过程分解为四个阶段：

```
查询图像 I_q
    │
    ▼
┌──────────────┐
│  1. 图像检索   │  全局描述子 → top-K 候选数据库图像
└──────────────┘
    │  {I_db^(1), ..., I_db^(K)}
    ▼
┌──────────────┐
│  2. 特征提取   │  局部特征关键点 + 描述子
└──────────────┘
    │  keypoints, descriptors
    ▼
┌──────────────┐
│  3. 特征匹配   │  查询 ↔ 候选 2D-2D 匹配 → 2D-3D 对应
└──────────────┘
    │  2D-3D correspondences
    ▼
┌──────────────┐
│  4. 位姿估计   │  PnP-RANSAC → (R, t)
└──────────────┘
```

其中，COLMAP 离线重建提供稀疏 3D 点云模型，使 2D-3D 对应可被索引。

---

## 2. Baseline 方法

| 模块 | Baseline A | Baseline B |
|---|---|---|
| 图像检索 | NetVLAD (PCA 4096-d) | NetVLAD (PCA 4096-d) |
| 特征提取 | SIFT (max 5000 kps) | SuperPoint (max 2048 kps) |
| 特征匹配 | Nearest Neighbor (ratio test 0.9) | SuperGlue |

两条 baseline 共享同一检索模块 NetVLAD，区别在于局部特征与匹配策略：Baseline A 采用经典手工特征 + 最近邻匹配，Baseline B 采用深度学习特征 + 学习匹配。通过对比，可以分别量化匹配模块和检索模块的贡献。

---

## 3. 新增方法选型与动机

选型遵循三个维度：**技术合理性**（方法本身是否在对应任务上有优势）、**工程可行性**（是否能接入 XRLocalization 框架而不引入过多工程复杂度）、**对比干净性**（是否与 baseline 构成清晰的消融对比）。核心原则是：**能分析清楚"为什么好 / 为什么不好"比盲目追求最高精度更重要**。

---

### 3.1 图像检索：EigenPlaces（ICCV 2023）

**为什么选 EigenPlaces 而不是 Patch-NetVLAD / TransVPR / SARE：**

视觉定位中的图像检索有两个目的：一是提供近似位姿估计，二是确定场景中哪些部分在查询图像中可见。因此检索模块必须关注**视角鲁棒性**，而不只是通用图像相似度。

- **Patch-NetVLAD** 虽在作业推荐列表中，但已实现在 HLoc/XRLab 体系中，直接排除。
- **TransVPR / SARE** 是 2021 年前后的早期方法，已被后续工作大幅超越，分析价值有限。
- **EigenPlaces**（ICCV 2023）的核心思路是通过将训练数据按地理聚类划分、在不同类簇上联合训练，迫使模型学习视角鲁棒的场景表征。在 MSLS、Pittsburgh、Tokyo247 等数据集上超越 MixVPR 和 NetVLAD，且 ResNet-50 骨干训练仅需 ~7 GB 显存，推理可直接使用预训练权重。未集成进 XRLab/HLoc，符合作业要求，工程接入成本低。

**模型架构：** ResNet-50 + GeM Pooling + FC 降维，输出 L2 归一化的全局描述子。

### 3.2 图像检索（加分）：CricaVPR（CVPR 2024）

**为什么选 CricaVPR 作为补充检索方案：**

- CricaVPR 以 DINOv2 为骨干，引入**跨图像相关感知表示**（cross-image correlation-aware representation），是当前 VPR 领域的主流趋势——将视觉基础模型（ViT）引入地点识别。
- 与 EigenPlaces 形成**"传统 CNN 骨干 vs. ViT 骨干"**的方法论对比，分析价值很高：在不同场景下两种范式的表现差异及其原因，本身就是很好的报告素材。
- 工作量受限时为可选项，优先保证 EigenPlaces 主线完成。

### 3.3 特征提取：ALIKED

**为什么选 ALIKED 而不是 R2D2 / D2Net：**

- **R2D2 / D2Net** 是 2019–2020 年的方法，已被后续工作超越；D2Net 的 dense 提取方式速度慢，不适合快速实验迭代。
- **ALIKED** 基于可变形卷积设计，在保持高关键点重复率的同时输出可靠局部描述子。
- **关键决策因素**：LightGlue 官方仓库已提供 ALIKED + LightGlue 的预训练权重，两者天然配套。选 ALIKED 不是单独升级特征提取，而是把"特征提取 + 特征匹配"作为一个**整体模块**升级——与 baseline 的"SuperPoint + SuperGlue"形成整体配对对比，不需要自己调超参。
- 未集成到 XRLab 或 HLoc，满足作业要求。

**模型输出：** 关键点坐标 (N, 2)、描述子 (N, 128)、得分 (N,)（得分表示关键点的可重复性置信度）。

### 3.4 特征匹配：LightGlue（ICCV 2023）— 核心亮点

**为什么选 LightGlue 而不是 LoFTR / COTR / ECO-TR：**

这是整个方案最核心的方法选择，理由最充分：

1. **与 baseline 形成完美的配对消融**：作业 baseline B 是 SuperPoint + SuperGlue，把 SuperGlue 换成 LightGlue 后，特征提取不变，**唯一变量就是匹配器**。对比干净，分析说服力强，这是其他任何匹配方法都无法提供的优势。

2. **速度优势可直接量化**：LightGlue 引入自适应深度退出机制——简单匹配对在浅层即可完成，困难匹配对才进入更深层。视觉重叠大或外观变化小时推理速度显著快于 SuperGlue。这直接体现在耗时分析（加分项）里，可以进行逐对匹配的退出层数统计。

3. **工程集成最简单**：官方仓库提供 SuperPoint、DISK、ALIKED、SIFT 四种特征的预训练权重，接口统一，几行代码即可替换匹配器。

相比之下：
- **LoFTR** 是 dense/semi-dense matcher，接口与 HLoc 的 sparse 流程不兼容，需要大量管线改造，风险高。
- **COTR / ECO-TR** 工程维护较少，复现难度高，容易卡在环境配置上。

**核心机制：** LightGlue 是对 SuperGlue 多个设计决策的系统性重新审视——包括注意力机制、位置编码、匹配层深度等——改进累积后使其在内存和计算上更高效、精度更高、训练更简单。其中最关键的是自适应退出（每层 predictor 判断匹配是否已充分，提前终止推理），配合多头注意力匹配。

**在 Aachen Day-Night、InLoc 等定位 benchmark 上精度超越 SuperGlue，未集成到 XRLab 或 HLoc，满足作业要求。**


---

## 4. 消融实验矩阵

实验设计遵循**逐模块隔离分析**原则：固定两个模块不变，单独替换一个模块，以量化每个新方法的独立增益。

| 实验编号 | 检索 | 特征提取 | 特征匹配 | 实验目的 |
|---|---|---|---|---|
| **BL-A** | NetVLAD | SIFT | NN | 官方对照—经典 pipeline |
| **BL-B** | NetVLAD | SuperPoint | SuperGlue | 官方对照—深度学习 pipeline |
| **EXP-R** | **EigenPlaces** | SuperPoint | SuperGlue | 隔离检索模块增益（仅替换检索） |
| **EXP-M** | NetVLAD | **ALIKED** | **LightGlue** | 隔离匹配模块增益（仅替换特征+匹配） |
| **EXP-Full** | **EigenPlaces** | **ALIKED** | **LightGlue** | 全新流水线综合效果 |
| **EXP-Crica** (加分) | **CricaVPR** | ALIKED | LightGlue | 更强检索方法的对比 |

**消融分析逻辑（逐模块隔离 + 协同效应检测）：**

核心原则是**每次只动一个模块**，这样每个实验的 Recall 变化可以直接归因：

```
Baseline B:  NetVLAD  + SuperPoint + SuperGlue   ← 参照
EXP-R:       EigenPlaces + SuperPoint + SuperGlue  ← 只换检索，验证检索增益
EXP-M:       NetVLAD  + ALIKED    + LightGlue    ← 只换提取+匹配，验证匹配增益
EXP-Full:    EigenPlaces + ALIKED  + LightGlue   ← 全换，看是否有协同效应
```

- `EXP-R` vs `BL-B` → EigenPlaces 替代 NetVLAD 的检索 Recall 增益（ΔR）
- `EXP-M` vs `BL-B` → ALIKED + LightGlue 替代 SuperPoint + SuperGlue 的匹配增益（ΔM）
- `EXP-Full` vs `BL-B` → 三个模块全部替换后的整体增益（ΔFull）
- **协同效应分析**：若 ΔFull > ΔR + ΔM，说明模块间存在正向协同（例如 EigenPlaces 检索出的更相关候选图像使得 LightGlue 的匹配优势得以更充分发挥）；若 ΔFull < ΔR + ΔM，则说明新增模块之间可能存在冗余或冲突——无论哪种情况，都可以在报告中做深入的原因分析，这正是作业要求的"思考与见解"。
- `EXP-Crica` vs `EXP-Full` → CNN 骨干 vs ViT 骨干的检索方法 head-to-head 对比

---

## 5. 数据集

### 5.1 数据集选择逻辑

**首选 7-Scenes 作为主要数据集：**

- **数据量小**：7 个室内场景，每个场景几千张图，下载快、COLMAP 重建快，适合在有限时间内完整跑通全流程。
- **难度适中**：室内场景包含纹理重复、运动模糊、光照变化等挑战，足够体现不同方法的性能差异，不会像 Aachen 那样因数据规模大而拖慢实验周期。
- **Cambridge 作为加分**：室外场景，与 7-Scenes 形成室内/室外分布对比，分析"方法在不同场景下为什么表现不同"时论据更充分。

---

### 5.2 主要数据集：7-Scenes

| 属性 | 说明 |
|---|---|
| 总大小 | ~20 GB（7 个独立 zip 文件） |
| 场景数 | 7 个室内场景（chess, fire, heads, office, pumpkin, redkitchen, stairs） |
| 图像格式 | PNG（640×480） |
| 位姿格式 | 4×4 相机到世界矩阵（`.pose.txt`） |
| 3D 模型 | KinectFusion 生成的 TSDF + 地面真值轨迹 |

**下载策略**：建议先下 `chess.zip`（~2.9 GB，场景最小、跑通最快）和 `office.zip`（~10.3 GB，场景复杂适合完整分析），等流水线调通后再按需补其他 5 个场景。

**COLMAP 预处理**：7-Scenes 自带 KinectFusion 点云和轨迹，但 HLoc 管线需要 COLMAP 格式的稀疏重建来做 2D-3D 索引。HLoc 仓库提供 7-Scenes 到 COLMAP 格式的转换脚本，提前装好 COLMAP 是避免卡壳的关键。

---

### 5.3 加分：Cambridge Landmarks

| 属性 | 说明 |
|---|---|
| 场景数 | 4 个室外场景，推荐 KingsCollege（343 查询）+ ShopFacade（103 查询） |
| 图像格式 | JPEG |
| 位姿格式 | `image_name qw qx qy qz tx ty tz` |

**下载方式**：dsacstar 仓库提供 `setup_cambridge.py` 脚本，可自动下载数据集并从稀疏 SfM 重建生成坐标文件，比手动去官网找链接更可靠。

---

### 5.4 加分：Aachen Day-Night

| 属性 | 说明 |
|---|---|
| 特点 | 日夜跨度极大，光照变化剧烈 |
| 意义 | LightGlue 自适应机制的强项场景——因其对困难匹配对能自动加深推理层数 |

---

## 5.5 环境与模型权重

**模型权重获取方式：**

| 方法 | 权重获取方式 | 备注 |
|---|---|---|
| EigenPlaces | pip 安装时自动拉取（`torch.hub`） | 无需手动下载 |
| ALIKED | LightGlue 仓库提供 / pip 安装 | 与 LightGlue 配套 |
| LightGlue | pip 安装 `lightglue` | 自动拉取预训练权重 |
| SuperPoint | 随 LightGlue 或 HLoc 安装 | 自动拉取 |
| NetVLAD | HLoc 内置 / pip 安装 | 自动拉取 |
| CricaVPR | **手动下载** `CricaVPR.pth`（GitHub Releases） | 需配合 DINOv2 骨干 |
| DINOv2 (CricaVPR 依赖) | **手动下载** `dinov2_vitb14_pretrain.pth`（Meta 官方直链） | `wget` 即可 |

大部分模型权重（EigenPlaces、LightGlue、SuperPoint、NetVLAD）通过 pip 安装或 `torch.hub` 自动拉取，不需要额外管理。真正需要手动下载的只有 CricaVPR 相关权重。

---

## 6. 评价指标

### 6.1 定位 Recall

报告三档阈值下的 Recall（命中率），这是视觉定位的标准评测指标：

| 阈值 | 含义 |
|---|---|
| (0.25m, 2°) | 高精度—可支持 AR 应用 |
| (0.5m, 5°) | 中等精度—典型定位需求 |
| (5m, 10°) | 粗精度—大致位置正确 |

**Recall 定义：** 对于每档阈值 $(T_t, T_r)$，若预测位姿 $(t_{pred}, R_{pred})$ 与真值 $(t_{gt}, R_{gt})$ 满足：

$$
\| t_{pred} - t_{gt} \|_2 < T_t \quad \text{且} \quad \angle(R_{pred}, R_{gt}) < T_r
$$

则判定为"定位成功"。Recall = 定位成功数 / 总查询数。

其中角度误差由单位四元数计算：

$$
\theta = 2 \arccos(|\langle q_{pred}, q_{gt} \rangle|) \quad (\text{单位: 度})
$$

### 6.2 检索 Recall@K

衡量检索模块的独立性能：对于每张查询图像，若 top-K 检索结果中至少有一张图像与查询的**真实最近邻**在空间上足够接近（平移 < 10m，旋转 < 30°），则判定为检索命中。

绘制 Recall@K 曲线（K = 1, 2, 5, 10, 20），用于对比 NetVLAD 与 EigenPlaces 的检索质量。

### 6.3 时间开销

每个查询按阶段分解耗时：

$$
T_{total} = T_{retrieval} + T_{detection} + T_{matching} + T_{pnp}
$$

报告各阶段均值 ± 标准差（ms），绘制耗时饼图。重点分析：
- 检索阶段：NetVLAD vs EigenPlaces 的推理速度差异
- 匹配阶段：SuperGlue vs LightGlue 的加速比（自适应退出的实际收益）

---

## 7. 可视化分析

### 7.1 检索 Recall@K 曲线

多方法对比曲线图，x 轴为 K 值，y 轴为 Recall。每个实验一条曲线，直观展示检索模块性能差异。重点分析 EigenPlaces 是否在小 K 值（K=1, 2）下相比 NetVLAD 有显著提升。

### 7.2 特征点检测可视化

选取典型查询图像，叠加关键点：
- 颜色编码得分（红=高重复性置信度，蓝=低）
- 对比 SIFT vs SuperPoint vs ALIKED 的关键点分布差异
- 分析：ALIKED 的关键点是否更集中在纹理丰富、结构显著的区域

### 7.3 特征匹配可视化

查询图像与候选图像 side-by-side 匹配连线：
- 绿色 = 高置信度匹配，红色 = 低置信度
- Inlier / Outlier 区分（经过 RANSAC 后）
- 对比 SuperGlue vs LightGlue 在困难场景下的匹配质量

### 7.4 耗时分析饼图

各阶段运行时间占比饼图，标注 ms 值和百分比。对比：
- BL-A vs BL-B 的时间分配差异
- EXP-Full 的时间分配—LightGlue 自适应退出带来的变化

### 7.5 定位成功/失败 Case 分析

选取定位成功和失败的典型查询：
- 分析失败原因（检索未命中？匹配不足？PnP 内点数不够？）
- 对比不同 pipeline 在同一查询上的表现差异
- 结合场景特点（纹理弱、光照极端、视角变化大等）给出解释

---

## 8. AR 演示

在重建场景中放置虚拟立方体，使用预测位姿将立方体投影渲染到查询图像上：

- 立方体边长为场景尺度的合理值（7-Scenes 室内场景取 ~0.3m）
- 半透明混合（alpha ≈ 0.6），可直观感受定位抖动（AR 物体在连续帧中的空间一致性）
- **抖动幅度直接反映定位精度**—高精度定位的查询帧中立方体与场景几何对齐良好

---

## 9. 预期分析要点

在报告中需重点关注以下问题：

1. **EigenPlaces 为什么比 NetVLAD 检索更准？** 从训练策略（地理聚类、交叉验证训练）和模型结构（GeM Pooling 的细粒度聚合能力）两个角度分析。

2. **LightGlue 相比 SuperGlue 在什么场景下优势最大？** 预测在简单重复纹理场景中自适应退出加速最明显；在极端视角/光照变化场景中深层次推理带来的精度优势最大。

3. **ALIKED + LightGlue 的组合是否优于 SuperPoint + SuperGlue？** 分析关键点质量（ALIKED 的可变形卷积 vs SuperPoint 的合成数据训练）和匹配策略（自适应退出 vs 固定层数）各自贡献。

4. **每个新增方法引入的代价是什么？** 包括模型参数量增加、推理时间变化、显存占用等。

5. **失败案例的共性是什么？** 通过聚合统计失败查询的特征，总结当前 pipeline 的能力边界。

---

## 参考资料

| 方法 | 论文 | 代码 |
|---|---|---|
| HLoc | Sarlin et al., CVPR 2019 | [hloc](https://github.com/cvg/Hierarchical-Localization) |
| NetVLAD | Arandjelović et al., TPAMI 2017 | [netvlad](https://www.di.ens.fr/willow/research/netvlad/) |
| EigenPlaces | Berton et al., ICCV 2023 | [EigenPlaces](https://github.com/gmberton/EigenPlaces) |
| ALIKED | Zhao et al., 2023 | [ALIKED](https://github.com/Shiaoming/ALIKED) |
| LightGlue | Lindenberger et al., ICCV 2023 | [LightGlue](https://github.com/cvg/LightGlue) |
| CricaVPR | Lu et al., CVPR 2024 | [CricaVPR](https://github.com/Lu-Feng/CricaVPR) |
| SuperPoint | DeTone et al., CVPRW 2018 | [SuperPoint](https://github.com/rpautrat/SuperPoint) |
| SuperGlue | Sarlin et al., CVPR 2020 | [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) |
| XRLocalization | OpenXRLab | [xrlocalization](https://github.com/openxrlab/xrlocalization) |

---

## 10. 实施进度追踪

### 环境与数据准备

- [ ] 克隆 XRLocalization，安装基础依赖
- [ ] 安装 COLMAP（7-Scenes → COLMAP 格式转换必需）
- [ ] 下载 7-Scenes `chess.zip`（~2.9 GB，快速验证用）
- [ ] 下载 7-Scenes `office.zip`（~10.3 GB，完整分析用）
- [ ] （加分）下载 Cambridge Landmarks（KingsCollege + ShopFacade）
- [ ] （加分）下载 7-Scenes 其余 5 个场景


### 模型与方法集成

- [ ] 集成 EigenPlaces 到 `extensions/retrievals/`
- [ ] 集成 ALIKED 到 `extensions/detectors/`
- [ ] 集成 LightGlue 到 `extensions/matchers/`
- [ ] （加分）集成 CricaVPR：手动下载 `CricaVPR.pth` + `dinov2_vitb14_pretrain.pth`

### Baseline 实验

- [ ] 编写 `configs/baseline_a.yaml`（NetVLAD + SIFT + NN）
- [ ] 编写 `configs/baseline_b.yaml`（NetVLAD + SuperPoint + SuperGlue）
- [ ] 跑通 BL-A → 验证 Recall 数值合理
- [ ] 跑通 BL-B → 验证 Recall > BL-A

### 消融实验

- [ ] EXP-R：EigenPlaces + SuperPoint + SuperGlue（检索增益）
- [ ] EXP-M：NetVLAD + ALIKED + LightGlue（匹配增益）
- [ ] EXP-Full：EigenPlaces + ALIKED + LightGlue（综合效果）
- [ ] （加分）EXP-Crica：CricaVPR + ALIKED + LightGlue（检索对比）

### 分析与可视化

- [ ] 汇总 `outputs/results/summary.csv`（所有实验的 Recall + 耗时）
- [ ] 检索 Recall@K 曲线图
- [ ] 特征点检测可视化
- [ ] 特征匹配可视化（inlier/outlier 连线）
- [ ] 各阶段耗时饼图
- [ ] 定位成功/失败 case 分析

### AR 演示

- [ ] 实现 `ar_demo/render_cube.py`
- [ ] 在 5 个以上样本查询上渲染立方体并输出结果

### 报告

- [ ] 撰写方法描述（选型动机、消融设计）
- [ ] 撰写定量分析（Recall 表、耗时表、协同效应分析）
- [ ] 撰写定性分析（可视化解读、case study、失败原因分析）
- [ ] （加分）7-Scenes vs Cambridge 跨数据集泛化分析
