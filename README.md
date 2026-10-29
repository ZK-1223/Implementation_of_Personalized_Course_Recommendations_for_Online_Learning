# LearnPath · 智学路径

> 在线学习的个性化课程推荐与学习路径规划系统
>
> 《智能应用系统设计》课程设计 · 第 5 题「在线学习的个性化课程推荐实现」

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.162-009688?logo=fastapi&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="ECharts" src="https://img.shields.io/badge/ECharts-5.5-AA344D">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 功能特性](#2-功能特性)
- [3. 系统架构](#3-系统架构)
- [4. 核心算法](#4-核心算法)
- [5. 快速开始](#5-快速开始)
- [6. 数据集说明](#6-数据集说明)
- [7. HTTP 接口一览](#7-http-接口一览)
- [8. 离线评测结果](#8-离线评测结果)
- [9. 项目结构](#9-项目结构)
- [10. 交付物](#10-交付物)
- [11. 开发与测试](#11-开发与测试)
- [12. 常见问题](#12-常见问题)

---

## 1. 项目简介

在线教育平台动辄有数万门课程，而学习者的知识基础、学习目标、学习风格、可投入时间差异巨大。
传统推荐系统只用"分类标签 + 热度排序"，无法回答三个关键问题：

1. **该学什么** —— 我的目标岗位上，我到底还缺哪些知识点？
2. **按什么顺序学** —— 课程之间存在先修依赖，顺序错了会学不动；
3. **学得动吗** —— 以我每周的时间和能力，这门课我能学完吗？

**LearnPath** 把这三个问题拆成一条完整的技术链路：

```
学习者多维画像  ──┐
                  ├─→  知识缺口识别  ──→  混合推荐（可解释）  ──→  学习路径规划  ──→  自适应重规划
课程知识图谱    ──┘                              ↑                      ↑                    ↑
                                            学习效果预测  ───────────────┴────────────────────┘
                                                 （完成概率 / 掌握度增益）
```

系统的核心主张是：**推荐不是"猜你喜欢"，而是在知识图谱约束下，为特定学习目标找到一条"能力可达成、时间可承受、难度平滑上升"的课程序列。**

### 技术特色

| 特色 | 说明 |
|---|---|
| **知识图谱约束** | 420 个知识点构成 DAG，先修闭包 / 拓扑分层 / 环检测保证推荐结果"学得动" |
| **多目标优化** | 用 NSGA-II 在「学习收益 / 时间成本 / 难度平滑度 / 风格匹配度」四目标上求帕累托前沿，再用 TOPSIS 选妥协解 |
| **可解释推荐** | 每条推荐给出具体中文理由（缺口数值、先修就绪度、同伴完成率），而非黑盒打分 |
| **自适应** | 学习反馈驱动 BKT 掌握度更新，自动触发"补强 / 跳课 / 收敛 / 加速"四类重规划 |
| **开箱即跑** | 数据集入库、模型可复现，无需 GPU、无需联网、无需构建步骤 |
| **全链路可视化** | FastAPI + 单页应用 6 个视图，ECharts 本地渲染，可离线演示 |

---

## 2. 功能特性

题目列出的 8 条实现要求全部落地，逐条对照见 [docs/REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md)。

| # | 实现要求 | 落地模块 | 可视化出口 |
|---|---|---|---|
| ① | 学习者建模（知识水平 / 学习偏好 / 时间安排） | `learnpath/learner/` | 学习者画像 |
| ② | 课程知识图谱（依赖关系 + 知识点关联） | `learnpath/knowledge/graph.py` | 学习路径 · 系统评测 |
| ③ | 学习路径规划（基于图谱与学习目标） | `learnpath/planning/planner.py` | 学习路径 |
| ④ | 多目标优化（效果 / 时间 / 难度梯度） | `learnpath/planning/multiobjective.py` | 学习路径 · 帕累托前沿 |
| ⑤ | 学习效果预测（掌握程度 + 完成概率） | `learnpath/prediction/` | 智能推荐 · 仪表板 |
| ⑥ | 自适应调整（按进度与反馈动态调整） | `learnpath/adaptive/engine.py` | 智能推荐（反馈提交） |
| ⑦ | 社交学习（同伴推荐 + 协作学习） | `learnpath/social/cohort.py` | 社交学习 |
| ⑧ | 学习分析仪表板（进度可视化 + 学习报告） | `learnpath/analytics/dashboard.py` | 学习分析仪表板 |

---

## 3. 系统架构

系统采用**三层架构**，引擎层与表现层完全解耦：算法不依赖 Web 框架，Web 层只做「参数解析 → 调用引擎 → 序列化」。

```
┌──────────────────────────────────────────────────────────────┐
│  表现层  web/                                                │
│  index.html · css/ · js/views/*.js · vendor/echarts.min.js   │
│  6 个视图：画像 / 推荐 / 路径 / 仪表板 / 社交 / 评测          │
└───────────────────────────┬──────────────────────────────────┘
                            │ fetch JSON（契约见 docs/API.md）
┌───────────────────────────▼──────────────────────────────────┐
│  服务层  app/                                                │
│  main.py（应用装配）· services.py（单例容器 + 缓存 + 降级）   │
│  routers/*.py（10 个路由模块，仅做参数解析与序列化）           │
└───────────────────────────┬──────────────────────────────────┘
                            │ Python 调用
┌───────────────────────────▼──────────────────────────────────┐
│  引擎层  learnpath/                                          │
│                                                              │
│  data/        数据契约 · 生成器 · 读写                       │
│  knowledge/   知识点 DAG（先修闭包 / 拓扑分层 / 环检测）      │
│  learner/     BKT 掌握度追踪 · 多维学习者画像                 │
│  recommend/   内容推荐 · 协同过滤 · 混合排序 · MMR 重排       │
│  planning/    路径规划 · NSGA-II 多目标优化 · TOPSIS          │
│  prediction/  完成概率分类 · 掌握度增益回归                   │
│  adaptive/    反馈驱动的重规划引擎                            │
│  social/      同伴匹配 · 学习小组组建                         │
│  analytics/   仪表板聚合 · 离线评测                           │
└──────────────────────────────────────────────────────────────┘
```

**关键设计决策**

- **数据契约先行**：`learnpath/data/schema.py` 用 dataclass 定义全部实体并在构造时强校验，
  任何模块都只依赖这份契约，因此数据层、算法层、服务层可并行开发且不会"各说各话"。
- **引擎无状态化**：`LearnerModel` 承载单个学习者的可变状态，其余组件（图谱、推荐器、预测器）
  都是只读的，可安全复用与缓存。
- **服务层单例 + 线程锁**：数据集与模型只在首次请求时加载一次；
  任一算法模块缺失时对应接口返回 503 而非让整个服务崩溃（便于分阶段交付）。

---

## 4. 核心算法

### 4.1 课程知识图谱（`knowledge/graph.py`）

知识点构成有向无环图 $G=(V,E)$，边 $(u,v)$ 表示「$u$ 是 $v$ 的先修」。核心操作：

- **环检测**：Kahn 拓扑排序，若出队节点数 $<|V|$ 则存在环并抛 `CycleError`；
- **先修闭包**：$\mathrm{Pre}(S)=\bigcup_{v\in S}\{v\}\cup\mathrm{Pre}(\mathrm{pred}(v))$，决定"要学目标，先得补什么"；
- **拓扑分层**：反复剥离入度为 0 的节点，得到分层序列 $L_0,L_1,\dots$，是路径规划的排序骨架；
- **课程级先修**：$\mathrm{Pre}(c)=\mathrm{Pre}(\mathrm{kp}(c))\setminus \mathrm{kp}(c)$，把知识点依赖投影到课程依赖。

### 4.2 学习者建模与掌握度追踪（`learner/`）

每个知识点维护掌握度 $m_{k}\in[0,1]$。学习行为发生后按 **BKT（贝叶斯知识追踪）** 更新：

$$
P(L_{t}\mid \text{obs}) = \frac{P(\text{obs}\mid L_t)\,P(L_t)}{P(\text{obs})},
\qquad
P(L_{t+1}) = P(L_t\mid \text{obs}) + \bigl(1-P(L_t\mid \text{obs})\bigr)\,p_{\text{learn}}\cdot a
$$

其中 $a$ 为学习者能力调制系数，$p_{\text{slip}},p_{\text{guess}}$ 为失误与蒙对概率（见 `config.BKT`）；
长期未复习的知识点按 $m\leftarrow m\cdot\gamma^{\Delta w}$ 衰减（$\gamma=$ `MASTERY_DECAY_PER_WEEK`）。

在此基础上派生四个决策用的量：**先修就绪度** `readiness_for`、**风格匹配度** `style_fit`、
**时间可承受度** `time_fit`、**预期掌握度增益** `expected_gain`。

### 4.3 混合推荐（`recommend/`）

三路召回 + 加权融合 + MMR 去冗余：

$$
\text{score}(c)=\sum_{j\in\{\text{content},\text{cf},\text{quality},\text{readiness},\text{social}\}} w_j\cdot s_j(c)
$$

- **内容分** $s_{\text{content}}$：课程对「目标闭包内缺口」的覆盖增益 / 时间成本，并对先修未就绪做惩罚；
- **协同分** $s_{\text{cf}}$：以 `Interaction.implicit_feedback` 构造隐式反馈矩阵，TruncatedSVD 隐语义分解后打分；
- **质量分** $s_{\text{quality}}$：评分与报名量的贝叶斯平滑；
- **社交分** $s_{\text{social}}$：相似同伴在该课程上的完成率。

最后用 **MMR** 重排保证列表多样性：$\;c^*=\arg\max_{c}\bigl[\lambda\cdot \text{score}(c)-(1-\lambda)\cdot \max_{c'\in S}\text{sim}(c,c')\bigr]$。

### 4.4 路径规划（`planning/planner.py`）

```
目标知识点 T → 先修闭包 Pre(T) → 缺口 G = {k | m_k < 门限}
     → 候选课程 C（覆盖 G、先修可达、时长可承受）
     → 分层贪心集合覆盖（按 graph.layers(G) 自底向上，每层取「边际收益/时长」最大者）
     → 拓扑合法性修复（若 B 覆盖的某知识点以 A 覆盖的知识点为先修，则 A 必须排在 B 前）
     → 多目标优化（见 4.5）
     → 周次装箱 + 里程碑生成
```

### 4.5 多目标优化（`planning/multiobjective.py`）

四个目标（全部归一化到 $[0,1]$）：

| 目标 | 方向 | 定义 |
|---|---|---|
| $f_{\text{gain}}$ | 最大化 | 路径累计预期掌握度增益 |
| $f_{\text{time}}$ | 最小化 | 路径总时长（折算为可承受度） |
| $f_{\text{smooth}}$ | 最大化 | $1-\text{norm}\bigl(\sum_i (\ell_{i+1}-\ell_i)^2\bigr)$，难度梯度平滑度 |
| $f_{\text{fit}}$ | 最大化 | 平均 $(\text{style\_fit}\times\text{time\_fit})$ |

用 **NSGA-II** 求帕累托前沿（快速非支配排序 + 拥挤度距离 + 锦标赛选择 + 模拟二进制交叉 + 多项式变异），
再用 **TOPSIS** 按用户偏好预设（均衡 / 速通 / 夯实）选取妥协解：

$$
C_i^{*}=\frac{D_i^{-}}{D_i^{+}+D_i^{-}},\qquad
D_i^{\pm}=\sqrt{\sum_j w_j\bigl(z_{ij}-z_j^{\pm}\bigr)^2}
$$

### 4.6 学习效果预测（`prediction/`）

从 `(学习者, 课程, 图谱)` 三元组构造 14+ 维特征（能力、平均掌握度、先修就绪度、风格匹配、
难度差、时长、覆盖知识点数、先修缺口比例、历史完成率、历史测验均值、口碑、报名量对数……），
**严格避免标签泄漏**（不使用该学习者在该课程上的任何历史结果）。

- 完成概率：`LogisticRegression` vs `HistGradientBoostingClassifier`，5 折分层交叉验证，报告 AUC / Accuracy / F1；
- 掌握度增益：`Ridge` vs `HistGradientBoostingRegressor`，报告 MAE / RMSE / R²；
- 可解释性：由特征重要性/系数给出 Top-k 影响因子，直接进入推荐理由。

### 4.7 自适应重规划（`adaptive/engine.py`）

| 触发条件 | 动作 |
|---|---|
| 测验分 $<$ `ADAPTIVE.low_quiz_score` | 判定未掌握 → 插入同知识点更基础的补强课 → 重规划 |
| 测验分 $>$ `ADAPTIVE.high_quiz_score` | 跳过路径中覆盖同一知识点的后续课程 |
| 实际/计划进度 $<$ `behind_schedule` | 收敛路径（消减低收益课程 / 切"速通"预设） |
| 实际/计划进度 $>$ `ahead_schedule` | 追加进阶课程 / 切"夯实"预设 |

### 4.8 社交学习（`social/cohort.py`）

复合相似度：

$$
\text{sim}(u,v)=\alpha\cos(\mathbf{m}_u,\mathbf{m}_v)+\beta\,\mathbb{1}[\text{同目标}]+\gamma\,\text{Jaccard}(\text{兴趣})
$$

学习小组按「能力互补」组建（有人已掌握、有人未掌握同一知识点），协同度衡量互补程度；
群体的课程完成率反过来作为推荐的社会信号 $s_{\text{social}}$。

---

## 5. 快速开始

### 5.1 环境要求

| 项 | 要求 |
|---|---|
| 操作系统 | Windows / Linux / macOS |
| Python | 3.11 ~ 3.12（开发环境 3.12.14） |
| 内存 | ≥ 4 GB |
| GPU | **不需要** |
| 网络 | 仅安装依赖时需要，运行/演示可完全离线 |

### 5.2 安装

```bash
# 1) 创建并激活虚拟环境
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 2) 安装依赖
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 5.3 生成数据与训练模型（可选，仓库已内置数据）

```bash
# 重新生成数据集（固定种子，结果完全可复现）
python scripts/gen_data.py --scale full --out data --force

# 训练学习效果预测模型，产出 models/model_metrics.json
python scripts/train_models.py
```

### 5.4 启动服务

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000> 即可使用；交互式 API 文档在 <http://127.0.0.1:8000/docs>。

### 5.5 一条命令跑完自检

```bash
python -m pytest tests -q
```

---

## 6. 数据集说明

数据集由 `learnpath/data/generator.py` **程序化合成**（固定随机种子 `20261029`，完全可复现，不含任何真实个人信息）。

| 实体 | 规模 | 说明 |
|---|---|---|
| 知识点 | 420 | 分布于 7 个领域，构成 DAG（28 个根、56 个叶、最大深度 8、483 条先修边） |
| 课程 | 560 | 覆盖全部知识点；难度 1~5 呈纺锤形分布；4 种承载模态 |
| 职业目标 | 8 | 数据分析师 / 前端 / 后端 / 算法 / 产品 / 网络安全 / 商务英语 / 数学强化 |
| 学习者 | 720 | 多维画像：能力、风格、时间预算、初始掌握度、兴趣 |
| 交互行为 | 14,000 | 「学习者—课程」记录，完成率 48.2% |

**完成标签不是一个随机数**：它由逻辑斯蒂模型生成（能力 + 难度差 + 时长 + 时间预算 + 先修就绪度 + 模态匹配 + 课程质量），
因此学习效果预测任务有真实可学的信号。生成器自检 AUC = **0.7829**（落在"信号明确但不过分干净"的区间）。

领域分布：

| 领域 | 典型知识点 |
|---|---|
| 编程开发 | 变量与数据类型、面向对象、并发编程、数据库分库分表、设计模式 |
| 数据与人工智能 | 数据清洗、假设检验、回归建模、聚类与客户分群、模型评估与 ROC 曲线 |
| 前端与设计 | HTML 语义化、CSS 布局、组件化开发、性能优化、前端自动化测试体系 |
| 网络与安全 | 网络分层模型、TCP/UDP、Web 攻防、加密与证书、渗透测试 |
| 产品与职场 | 需求调研、用户故事、原型设计、数据分析看板、增长实验 |
| 语言学习 | 音标与连读、核心词汇、基础句型、学术写作、商务表达 |
| 数学基础 | 函数与极限、线性代数、概率论、最优化、数值计算 |

---

## 7. HTTP 接口一览

完整字段定义见 [docs/API.md](docs/API.md)。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/meta` | 数据集统计、领域、目标、预设、学习者列表 |
| GET | `/api/learners` | 学习者分页检索 |
| GET | `/api/learners/{id}` | 学习者画像（多维 + 掌握度 + 缺口 + 历史） |
| GET | `/api/recommend/{id}` | 个性化推荐（含分项得分、中文理由、完成概率） |
| GET | `/api/path/{id}` | 学习路径（含周次排布、里程碑、三套备选方案） |
| GET | `/api/path/{id}/pareto` | 帕累托前沿（时间—收益权衡） |
| POST | `/api/feedback` | 提交学习反馈，触发自适应重规划 |
| GET | `/api/social/{id}` | 同伴推荐与学习小组 |
| GET | `/api/analytics/{id}` | 学习分析仪表板聚合数据 |
| GET | `/api/knowledge-graph` | 知识图谱（可按领域筛选、按掌握度染色） |
| GET | `/api/courses` | 课程库分页检索 |
| GET | `/api/evaluation` | 离线评测指标（推荐算法对比 / 效果预测模型 / 消融实验） |

---

## 8. 离线评测结果

> 运行 `python scripts/evaluate.py` 生成，完整数据见 [`models/evaluation.json`](models/evaluation.json) 与
> [`models/model_metrics.json`](models/model_metrics.json)。协议：按学习者分组留出法，每位学习者保留 20% 正反馈作为测试集，
> 在「该学习者未交互过的课程」上排序，K = 10，共评测 713 位学习者 / 1,472 条测试正反馈。

### 8.1 推荐策略对比

| 策略 | Precision@10 | Recall@10 | **NDCG@10** | MAP | Coverage | Diversity | Novelty | HitRate |
|---|---|---|---|---|---|---|---|---|
| Popularity 基线 | 0.0132 | 0.0622 | 0.0328 | 0.0168 | 0.029 | 0.905 | 0.844 | 0.1234 |
| Content 内容推荐 | 0.0302 | 0.1495 | 0.0902 | 0.0526 | 0.282 | 0.891 | 0.923 | 0.2763 |
| CF-SVD 协同过滤 | 0.0167 | 0.0847 | 0.0511 | 0.0302 | 0.468 | 0.977 | 0.911 | 0.1599 |
| **Hybrid 混合推荐（本系统）** | **0.0374** | **0.1840** | **0.1026** | **0.0571** | 0.377 | 0.969 | 0.931 | **0.3310** |

**混合推荐相对热度基线：NDCG@10 提升 3.13 倍（+212.8%），Precision@10 +183%、Recall@10 +196%、MAP +240%、HitRate +168%。**
相对最强单路策略（Content）：NDCG +13.7%、Precision +23.8%、Recall +23.1%、HitRate +19.8%。**四项排序指标全部第一。**

### 8.2 消融实验

| 变体 | NDCG@10 | Δ NDCG | Diversity | 结论 |
|---|---|---|---|---|
| 完整混合推荐（基线） | 0.1026 | — | 0.969 | — |
| 去掉知识图谱先修约束 | 0.0639 | **−0.0387（−37.7%）** | 0.994 | **知识图谱是最大单一贡献项** |
| 去掉 MMR 多样性重排 | 0.1084 | +0.0058 | 0.904 | 用 0.6% NDCG 换 6.5% 多样性——显式设计权衡 |
| 去掉协同过滤信号 | 0.1075 | +0.0048 | 0.959 | 对排序精度无增益，价值在 Coverage（0.468）与冷启动兜底 |
| 去掉社交同伴信号 | 0.1013 | −0.0013 | 0.970 | 贡献在噪声量级，属锦上添花 |

> 诚实结论：只有**知识图谱约束**对排序精度有实质贡献；MMR 是**多样性**的显式权衡（准确率略降、多样性上升），
> 协同过滤的价值体现在**覆盖率与冷启动**而非 NDCG。这些结论已在报告中如实讨论，未做美化。

### 8.3 学习效果预测

| 任务 | 模型 | 指标 |
|---|---|---|
| 课程完成概率（分类） | 逻辑回归 | AUC 0.8072 / Accuracy 0.7332 / F1 0.7207 |
| 课程完成概率（分类） | **梯度提升树** | **AUC 0.8820** / Accuracy 0.7973 / F1 0.7899 |
| 学习效果（回归） | **岭回归** | **MAE 0.0579 / RMSE 0.0724 / R² 0.6431** |
| 学习效果（回归） | 梯度提升树 | MAE 0.0595 / RMSE 0.0746 / R² 0.6205 |

Top-5 特征重要性：`hist_completion_rate` 0.255、`learner_ability` 0.200、`readiness` 0.102、
`level_gap` 0.079、`hist_avg_quiz` 0.070 —— 与教育学直觉一致，且已据此生成推荐理由。

### 8.4 运行性能

| 环节 | 耗时 | 说明 |
|---|---|---|
| 推荐 `recommend(k=10)` | 4.5 ms | 单学习者单次 |
| 路径规划 `plan()` | 0.27 s / 单 preset | 560 课程 / 420 知识点规模 |
| 数据生成 `gen_data.py --scale full` | 1.4 s | 含 AUC 自评 |
| 效果模型训练 `train_models.py` | 约 40 s | 5 折交叉验证 |
| 完整离线评测 `evaluate.py` | 约 50 s | 4 策略对比 + 消融 |

---

## 9. 项目结构

```
.
├── learnpath/                 # 引擎层（纯 Python，不依赖 Web 框架）
│   ├── config.py              # 全局超参数与枚举
│   ├── paths.py               # 路径约定
│   ├── data/                  # 数据契约 / 生成器 / 读写 / 微型 fixture
│   ├── knowledge/             # 知识点 DAG
│   ├── learner/               # BKT 掌握度追踪 + 学习者画像
│   ├── recommend/             # 内容 / 协同 / 混合推荐
│   ├── planning/              # 路径规划 + 多目标优化
│   ├── prediction/            # 完成概率 + 掌握度增益
│   ├── adaptive/              # 自适应重规划
│   ├── social/                # 同伴匹配 + 学习小组
│   └── analytics/             # 仪表板聚合 + 离线评测
├── app/                       # 服务层（FastAPI）
│   ├── main.py                # 应用装配
│   ├── services.py            # 单例容器（加载 / 缓存 / 降级）
│   ├── schemas.py             # Pydantic 响应模型
│   └── routers/               # 10 个路由模块
├── web/                       # 表现层（单页应用，无构建步骤）
│   ├── index.html
│   ├── css/                   # base / components
│   ├── js/                    # api / charts / router / views/*
│   └── vendor/echarts.min.js  # 本地 ECharts，保证离线可用
├── data/                      # 数据集（入库，开箱即跑）
├── models/                    # 模型指标（入库）；二进制模型不入库、可复现
├── scripts/                   # 数据生成 / 模型训练 / 评测 / 启动
├── tests/                     # pytest 单元测试与接口测试
└── docs/                      # 架构、接口、对照表、报告、PPT
```

---

## 10. 交付物

| 交付物 | 位置 |
|---|---|
| 课程设计报告（Word，**直接填进老师模板**） | [docs/report/](docs/report/) |
| 工程答辩 PPT | [docs/slides/](docs/slides/) |
| 报告/PPT 的可复现生成脚本 | [docs/report/build_report.py](docs/report/build_report.py) · [docs/slides/build_slides.py](docs/slides/build_slides.py) |
| 实现要求对照表 | [docs/REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md) |
| 接口契约 | [docs/API.md](docs/API.md) |
| 系统架构说明 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |

> **关于报告**：`docs/report/build_report.py` 不是「另建一份文档」，而是把老师下发的
> 《智能应用系统课程设计报告-2025模版.doc》转成 docx 后，**在模板自带的表格单元格里逐节填入正文**。
> 因此成品保留模板原有的封面、校徽、说明页、`题目序号/任务名称` 行、`课程设计内容` 表格结构、
> 页面设置与宋体/Times New Roman 字体规范。该 `.doc` 模板属课程原始材料，已在 `.gitignore` 中排除，
> 运行脚本前需自行放回仓库根目录（或通过 `LEARNPATH_REPORT_TEMPLATE` 环境变量指定路径）。

---

## 11. 开发与测试

```bash
# 全部测试
python -m pytest tests -q

# 按模块
python -m pytest tests/test_knowledge_graph.py -q      # 知识图谱
python -m pytest tests/test_learner.py -q              # 学习者建模
python -m pytest tests/test_recommend.py -q            # 推荐
python -m pytest tests/test_planning.py -q             # 路径规划与多目标优化
python -m pytest tests/test_prediction.py -q           # 效果预测
python -m pytest tests/test_api.py -q                  # HTTP 接口
```

编码约定：

- 全部模块使用中文 docstring 说明**算法原理**与参数含义，便于报告引用；
- 完整类型注解，核心数据结构用 `dataclass(slots=True)`；
- 引擎层禁止 `print`、禁止网络请求、禁止依赖 Web 框架；
- 随机性一律走显式 `seed`，保证结果可复现。

---

## 12. 常见问题

**Q：没有 GPU 能跑吗？**
A：能。本项目不含深度学习训练，效果预测用 sklearn 的逻辑回归与梯度提升树，全流程在 CPU 上秒级完成。

**Q：`data/` 里的数据是真实的吗？**
A：不是。全部为程序化合成的教学数据，不含任何真实个人信息；生成器与随机种子都在仓库里，可完全复现。

**Q：前端为什么不打包？**
A：刻意选择无构建步骤的方案（原生 ES Module + 本地 ECharts），避免引入 Node 工具链，
使得"下载仓库 → 装 Python 依赖 → 启动"即可演示，评审环境零配置成本。

**Q：为什么 `models/` 里没有 `.joblib`？**
A：二进制模型可通过 `python scripts/train_models.py` 一键复现，不入库以避免仓库体积膨胀；
`models/model_metrics.json` 保留了训练结果的指标与特征重要性，可追溯。

---

## License

本项目为课程设计作品，代码采用 MIT 许可证，仅供学习与教学使用。
