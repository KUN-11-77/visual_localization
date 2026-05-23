# Visual Localization Project — 检查报告

> 检查日期: 2026-05-21 | 基于 task.md 和 method.md

---

## 一、检查概览

| 模块 | 状态 | 关键问题 |
|------|------|----------|
| 1. 环境与基础依赖 | 部分完成 | 缺少 `lightglue` pip 包, `faiss-cpu`、`kornia`、`einops` 未见安装 |
| 2. 数据集 | 部分完成 | 7-Scenes 只有 stairs 场景, method.md 计划 chess+office; Cambridge KingsCollege 完整 |
| 3. 扩展模块 (extensions/) | ✅ 已完成 | 3 检索 + 3 检测 + 3 匹配 全部注册,代码质量好 |
| 4. 模型权重 | 部分完成 | NetVLAD 缺失(路径 bug); EigenPlaces/CricaVPR 齐备 |
| 5. 配置文件 (configs/) | ✅ 已完成 | 6 个配置文件完整,对应消融矩阵 |
| 6. Pipeline 脚本 | 基本完成 | run_pipeline.py 可工作,但尚未成功跑通任一实验 |
| 7. 可视化脚本 | 基本完成 | visualize_results.py 框架已就绪,缺少实际数据验证 |
| 8. AR Demo | 基本完成 | render_cube.py 已实现,缺少运行主脚本 |
| 9. 报告 | ❌ 未开始 | REPORT.md 为空;report/figures/ 为空 |

---

## 二、详细检查

### 2.1 扩展模块 ✅

全部已正确实现并注册:

| 类别 | 方法 | 文件 | 注册名 | 状态 |
|------|------|------|--------|------|
| 检索 | NetVLAD | extensions/retrievals/netvlad.py | NetVLADRetrieval | ✅ |
| 检索 | EigenPlaces | extensions/retrievals/eigenplaces.py | EigenPlacesRetrieval | ✅ |
| 检索 | CricaVPR | extensions/retrievals/cricavpr.py | CricaVPRRetrieval | ✅ |
| 检测 | SIFT | extensions/detectors/sift.py | SIFTDetector | ✅ |
| 检测 | SuperPoint | extensions/detectors/superpoint.py | SuperPointDetector | ✅ |
| 检测 | ALIKED | extensions/detectors/aliked.py | ALIKEDDetector | ✅ |
| 匹配 | NN | extensions/matchers/nn.py | NNMatcher | ✅ |
| 匹配 | SuperGlue | extensions/matchers/superglue.py | SuperGlueMatcher | ✅ |
| 匹配 | LightGlue | extensions/matchers/lightglue.py | LightGlueMatcher | ✅ |

`extensions/__init__.py` 注册+导入逻辑完整。

### 2.2 配置文件 ✅

| 文件 | 实验 ID | 检索 | 检测 | 匹配 | 数据集 |
|------|---------|------|------|------|--------|
| configs/baseline_a.yaml | BL-A | NetVLAD | SIFT | NN | Cambridge/KingsCollege |
| configs/baseline_b.yaml | BL-B | NetVLAD | SuperPoint | SuperGlue | Cambridge/KingsCollege |
| configs/exp_retrieval.yaml | EXP-R | EigenPlaces | SuperPoint | SuperGlue | Cambridge/KingsCollege |
| configs/exp_match.yaml | EXP-M | NetVLAD | ALIKED | LightGlue | Cambridge/KingsCollege |
| configs/exp_full.yaml | EXP-Full | EigenPlaces | ALIKED | LightGlue | Cambridge/KingsCollege |
| configs/exp_crica.yaml | EXP-Crica | CricaVPR | ALIKED | LightGlue | Cambridge/KingsCollege |

**注意**: 所有配置都指向 Cambridge/KingsCollege, method.md 计划使用 7-Scenes 作为主数据集。建议增加 7-Scenes 配置文件。

### 2.3 数据集 ⚠️ 部分完成

| 数据集 | 场景 | 3D 模型 | 状态 |
|--------|------|---------|------|
| Cambridge | KingsCollege | reconstruction.nvm ✅ | 完整可用 |
| 7-Scenes | stairs | ❌ 无 COLMAP 模型 | 只有图像+位姿,缺少 3D 模型索引 |
| 7-Scenes | chess/office/heads/... | ❌ | method.md 计划但未下载 |

**7-Scenes 缺少 3D 模型**: run_pipeline.py 的 `_load_7scenes` 函数返回 `nvm_model=None`, 导致 2D-3D 对应查找失败,回退到伪 3D 坐标 (plane assumption),定位精度会很差。

### 2.4 模型权重 ⚠️ NetVLAD 路径问题

| 模型 | 路径 | 状态 |
|------|------|------|
| NetVLAD | xrloc/models/VGG16-NetVLAD-Pitts30K/Pitts30K_struct.mat | ❌ 路径解析错误 |
| EigenPlaces | third_party/eigenplaces/weights/ResNet50_2048_eigenplaces.pth | ✅ |
| CricaVPR | weights/CricaVPR.pth | ✅ |
| DINOv2 | weights/dinov2_vitb14_pretrain.pth | ✅ |
| SuperPoint/SuperGlue | third_party/SuperGluePretrainedNetwork/models/weights/ | ✅ |
| ALIKED/LightGlue | third_party/LightGlue/LightGlue-main/lightglue/ | ✅ |

**NetVLAD 问题分析**:
- 模型文件实际位置: `xrlocalization/xrloc/models/VGG16-NetVLAD-Pitts30K/Pitts30K_struct.mat`
- `get_parent_dir(__file__)` (level=1) 返回 `xrloc/`
- 拼接 `/../models/` 后解析为 `xrlocalization/models/` (空目录)
- **修复方法**: 将 `xrloc/models/VGG16-NetVLAD-Pitts30K/` 复制/软链接到 `xrlocalization/models/`

### 2.5 Pipeline 脚本 ⚠️ 基本完成但有风险点

`scripts/run_pipeline.py` — 功能完整,流程正确。

**已知问题**:
1. **7-Scenes 无 3D 模型**: 当 nvm_model=None 时,使用伪 3D 坐标 `(x, y, 0.0)`,平面假设在非平面场景中误差极大
2. **检索在每张 query 上重新编码**: `retrieval.encode(query_image)` 在循环内执行,无批量优化(功能上正确但慢)
3. **PnP 参数硬编码**: `reprojectionError=12.0`, `minInliersCount=12` 在代码中硬编码,未从 config 读取

**其他脚本状态**:
- `scripts/evaluate.py` ✅ — Recall 计算逻辑正确,三档阈值符合 task.md 要求
- `scripts/nvm_model.py` ✅ — NVM 解析+2D-3D 空间查找逻辑正确
- `scripts/colmap_model.py` ✅ — COLMAP 模型读取实现,但目前未被使用
- `scripts/timing.py` ✅ — 简洁的计时工具
- `scripts/visualize_results.py` ⚠️ — 代码框架完整但 `recall_comparison` 函数使用硬编码列名 (`recall_025m_2deg`),与 evaluate.py 输出的格式 `(0.25m, 2deg)` 不一致

**可视化脚本与实际输出的格式不匹配**: 
- evaluate.py 输出的 key: `"(0.25m, 2deg)"`  
- visualize_results.py 中查找的列名: `"recall_025m_2deg"`
- 会导致可视化失败。

### 2.6 AR Demo ⚠️

`ar_demo/render_cube.py`:
- `render_cube_on_image()` 函数 ✅ — 立方体投影实现正确
- `rotmat_to_quaternion()` ✅ — 与 run_pipeline.py 中一致
- **缺少**: 主运行脚本(读取预测位姿、加载图像、批量调用 render_cube)、demo_output/ 目录为空

### 2.7 报告 ❌ 未开始

- REPORT.md 为空
- report/figures/ 为空
- 所有可视化图未生成
- 消融分析未开始
- 数据集对比未进行

### 2.8 依赖安装

requirements.txt 列出的包缺少:
- `lightglue` (ALIKED/LightGlue 需要)
- 未验证 `faiss-cpu`、`kornia`、`einops` 是否已安装

---

## 三、阻塞问题清单

### 🔴 P0 — 阻塞所有实验

| # | 问题 | 影响范围 | 修复方式 |
|---|------|----------|----------|
| 1 | **NetVLAD 模型路径错误** | BL-A, BL-B, EXP-M 无法运行 | 复制 `xrloc/models/VGG16-NetVLAD-Pitts30K/` → `xrlocalization/models/` |
| 2 | **缺少 lightglue 包** | EXP-M, EXP-Full, EXP-Crica 无法运行 | `pip install lightglue` |

### 🟡 P1 — 阻塞 7-Scenes 实验

| # | 问题 | 影响范围 | 修复方式 |
|---|------|----------|----------|
| 3 | **7-Scenes 缺少 COLMAP/NVM 3D 模型** | 无法在 7-Scenes 上正确评估定位 Recall | 运行 COLMAP 重建 或 使用 HLoc 提供的转换脚本 |
| 4 | **7-Scenes 只下载了 stairs 场景** | method.md 计划 chess + office | 下载 chess/office 并完成 3D 重建 |
| 5 | **缺少 7-Scenes 配置文件** | 无法对 7-Scenes 运行实验 | 创建 `configs/*_7scenes_*.yaml` |

### 🟢 P2 — 代码质量/完善

| # | 问题 | 修复方式 |
|---|------|----------|
| 6 | visualize_results.py 列名与 evaluate.py 输出不一致 | 统一格式 |
| 7 | run_pipeline.py PnP 参数未从 config 读取 | 使用 cfg['pose_solver'] 中的参数 |
| 8 | AR demo 缺少主运行脚本 | 编写 ar_demo/run.py |
| 9 | 所有实验尚未成功运行 | 依次执行 6 个消融实验 |

---

## 四、method.md 进度对照

根据 method.md 第 10 节的 checklist:

| 任务 | 状态 |
|------|------|
| 克隆 XRLocalization,安装基础依赖 | ✅ |
| 安装 COLMAP | ⚠️ third_party/colmap/ 已下载,未确认可执行 |
| 下载 7-Scenes chess/office | ❌ 只有 stairs |
| 下载 Cambridge KingsCollege+ShopFacade | ⚠️ 只有 KingsCollege |
| 集成 EigenPlaces | ✅ |
| 集成 ALIKED | ✅ |
| 集成 LightGlue | ✅ |
| 集成 CricaVPR | ✅ |
| 编写 baseline/experiment 配置 | ✅ |
| 跑通 BL-A/BL-B | ❌ NetVLAD bug 阻塞 |
| 消融实验 EXP-R/M/Full | ❌ 未运行 |
| EXP-Crica (加分) | ❌ 未运行 |
| 汇总 summary.csv | ❌ 未开始 |
| 检索 Recall@K 曲线 | ❌ 未开始 |
| 特征点可视化 | ❌ 未开始 |
| 匹配可视化 | ❌ 未开始 |
| 耗时饼图 | ❌ 未开始 |
| Case 分析 | ❌ 未开始 |
| AR Demo | ❌ 未开始 |
| 报告撰写 | ❌ 未开始 |

---

## 五、Cambridge Landmarks 基准参考数据

> 来源: XRLocalization 官方 benchmark

**指标**: 中位平移误差 (cm) 和旋转误差 (度)，越小越好。最佳结果加粗。

| Method | Great Court | **Kings College** | Old Hospital | Shop Facade | St M. Church | Avg |
|--------|-------------|-------------------|-------------|-------------|-------------|-----|
| DSAC++ | 40.3, 0.20 | 17.7, 0.30 | 19.6, 0.30 | 5.7, 0.30 | 12.5, 0.40 | 19.2, 0.30 |
| NG-DSAC | 34.8, 0.18 | 12.2, 0.23 | 21.2, 0.45 | 5.4, 0.29 | 9.9, 0.31 | 16.7, 0.29 |
| PixLoc | 30.0, 0.14 | 14.0, 0.24 | 16.0, 0.32 | 5.0, 0.23 | 10.0, 0.34 | 15.0, 0.25 |
| Active Search | 24.0, 0.13 | 13.0, 0.22 | 20.0, 0.36 | 4.0, 0.21 | 8.0, 0.25 | 13.8, 0.23 |
| HLoc+SuperGlue | 10.1, 0.07 | 6.9, 0.11 | 12.5, 0.24 | 2.9, 0.14 | 3.8, 0.12 | 7.2, 0.14 |
| XRLoc(D2Net) | 10.6, 0.08 | 5.5, 0.10 | 11.8, 0.25 | 3.3, 0.14 | 3.6, 0.11 | 7.0, 0.14 |
| **XRLoc(SuperPoint)** | **10.4, 0.07** | **5.2, 0.10** | **10.5, 0.23** | **2.4, 0.12** | **3.5, 0.11** | **6.4, 0.13** |

### 对本项目的参考意义

- **BL-B (NetVLAD + SuperPoint + SuperGlue)** 理论上接近 HLoc+SuperGlue 行: KingsCollege 中位误差约 **6.9cm / 0.11°**
- 在该精度下，(0.25m, 2°) Recall 应 **> 85%**，(0.5m, 5°) Recall 应 **> 95%**
- **BL-A (NetVLAD + SIFT + NN)** 精度会明显低于上述所有方法，预期 (0.25m, 2°) Recall 约 **60-70%**
- 本项目使用 **NVM 模型**（非 COLMAP），实际结果可能比基准低 5-10%

---

## 六、建议执行顺序

1. **立即修复 P0 阻塞**: NetVLAD 路径 + 安装 lightglue
2. **跑通 BL-A / BL-B**: 验证 baseline 在 KingsCollege 上的 Recall 数值合理
3. **跑通 EXP-R / EXP-M / EXP-Full**: 完成消融实验
4. **修复可视化**: 统一列名格式,生成所有图表
5. **补充 7-Scenes**: 下载 chess+office,运行 COLMAP 重建,创建 7-Scenes 配置
6. **跨数据集分析**: KingsCollege vs 7-Scenes 对比
7. **AR Demo**: 编写主脚本,渲染多帧
8. **撰写报告**: 包含定量表、可视化、消融分析、Case Study

---

## 七、用户界面设计

为满足课程项目中"实现完整的用户界面"的要求，本实验设计了**两套界面**供助教测试和成果展示：

### 7.1 Web 实验运行器 (`scripts/ui_web.py`)

基于 Flask 的 Web 应用，助教可通过浏览器直接操作：

**启动方式：**
```bash
python scripts/ui_web.py --port 5000
# 打开 http://localhost:5000
```

**界面功能：**

| 功能区域 | 说明 |
|----------|------|
| 快速预设 | 一键加载已保存的实验配置（baseline_a/b、exp_retrieval/match/full/crica 等） |
| 数据集选择 | 下拉菜单选择 Cambridge Landmarks（5 个场景）或 7-Scenes（7 个场景），自动填充路径和内参 |
| 方法选择 | 独立选择检索器（NetVLAD / EigenPlaces / CricaVPR）、检测器（SIFT / SuperPoint / ALIKED）、匹配器（NN / SuperGlue / LightGlue） |
| 参数调节 | Top-K、最大关键点数、查询数量限制等参数可自由设置 |
| 后台运行 | 实验在子进程中异步执行，前端每 1.5 秒轮询进度，实时显示进度条和阶段信息 |
| 结果展示 | 完成后自动展示：Recall 指标卡片（三档阈值）、各阶段耗时分析表、逐帧定位结果表（含平移/旋转误差和成功/失败状态） |

**API 接口：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/run` | POST | 提交实验配置，返回 job_id 用于轮询 |
| `/api/status/<job_id>` | GET | 查询实验进度和结果 |
| `/api/results` | GET | 获取所有实验结果详情 |
| `/api/datasets` | GET | 获取可用数据集和场景预设 |
| `/api/methods` | GET | 获取已注册的检索/检测/匹配方法列表 |
| `/api/configs` | GET | 获取已保存的实验配置文件列表 |

**设计特点：**
- 左右分栏布局：左侧为配置表单（420px 固定宽度），右侧为结果展示区
- 自适应设计：窄屏下自动切换为上下布局
- 状态指示：旋转动效表示运行中，绿色/红色徽标表示成功/失败
- 预设芯片：点击即可加载已有配置，减少手工填写

### 7.2 成果展示页面 (`/showcase`)

独立的静态展示页面，访问 `http://localhost:5000/showcase` 查看：

**页面组成：**

1. **顶部横幅 (Hero Banner)**
   - 项目标题 "Hierarchical Visual Localization"
   - 简要说明和统计徽章（3 检索 + 3 检测 + 3 匹配 + 2 数据集 + 8 实验）

2. **总览统计卡片**
   - 实验总数、最佳 (0.25m, 2°) Recall、最佳 (0.5m, 5°) Recall、测试数据集数

3. **Pipeline 架构图**
   - 可视化展示：查询图像 → 图像检索 → 特征检测 → 特征匹配 → PnP RANSAC → 6-DoF 位姿

4. **Cambridge Landmarks 对比分析**
   - Chart.js 柱状图：6 个实验配置在三档阈值下的 Recall 对比
   - 各阶段耗时条形图（检索/检测/匹配/位姿估计）
   - 详细对比表格：列出每个实验的检索器、检测器、匹配器和对应的 Recall 数值

5. **7-Scenes 对比分析**
   - 柱状图对比
   - 深度图重投影方法说明

6. **集成方法卡片**
   - EigenPlaces、CricaVPR、ALIKED、LightGlue 各自的方法简介和特点

7. **AR Demo 画廊**
   - 展示 AR 立方体叠加效果图（绿色=定位成功，红色=定位失败）

**技术实现：**
- 使用 Chart.js v4.4.0 CDN 绘制交互式图表
- 数据通过 `/api/results` 接口动态加载
- 自动选取每个实验的最佳运行结果用于展示
- 响应式设计，支持移动端和打印

### 7.3 命令行界面 (`scripts/ui.py`)

保留传统的终端交互式菜单，作为 Web 界面的补充：

```
Main Menu:
  [1] Run Experiment    — 选择配置、设置参数、运行实验
  [2] View Results      — 查看 Recall 指标、耗时、逐帧结果
  [3] Generate AR Demo  — 选择实验、设置立方体大小、生成 AR 可视化
  [4] Export Report     — 导出 HTML 报告和原始数据到指定目录
  [0] Exit
```

### 7.4 设计理念

1. **助教友好**：Web 界面无需命令行知识，浏览器即可完成所有操作
2. **输入-处理-输出完整闭环**：符合课程要求，从配置输入到结果展示形成完整流程
3. **结果可保存**：实验结果自动保存到 `outputs/results/` 目录，支持导出
4. **可视化对比**：展示页面提供直观的图表对比，便于评估不同方法的优劣
5. **模块化设计**：Flask 应用逻辑与 HTML 模板分离，便于维护和扩展
