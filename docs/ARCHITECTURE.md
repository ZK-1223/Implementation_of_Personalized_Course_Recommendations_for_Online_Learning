# 系统架构说明

本文说明 LearnPath 的分层架构、模块职责、关键接口与设计取舍，供阅读源码与答辩讲解使用。

---

## 1. 总体分层

系统严格分为三层，**引擎层不依赖任何 Web 框架，表现层不包含任何算法逻辑**：

```
表现层 web/            ← 只消费 HTTP JSON，不含业务规则
    ↑ HTTP
服务层 app/            ← 参数解析、对象装配、缓存、序列化、降级
    ↑ Python 调用
引擎层 learnpath/      ← 全部算法与数据结构
```

这样划分带来三个直接好处：

1. **可测试**：算法可以脱离 HTTP 直接用 pytest 验证（`tests/test_*.py` 中大量用例属于此类）；
2. **可替换**：把 FastAPI 换成 CLI 或桌面 GUI，引擎层一行不用改；
3. **可并行开发**：数据契约固定后，数据、算法、界面三条线可以互不阻塞地推进。

---

## 2. 引擎层模块职责

| 模块 | 职责 | 对外主要类 / 函数 |
|---|---|---|
| `learnpath/config.py` | 全局枚举与超参数（领域、BKT 参数、推荐权重、多目标权重、自适应阈值） | `DOMAINS` `BKT` `RECOMMEND_WEIGHTS` `PLAN_PRESETS` `ADAPTIVE` |
| `learnpath/paths.py` | 仓库内资源定位，避免业务代码硬编码相对路径 | `DATA_DIR` `MODELS_DIR` `WEB_DIR` |
| `learnpath/data/schema.py` | **全项目数据契约**，dataclass 强校验 + JSON 序列化 | `Dataset` `KnowledgePoint` `Course` `Learner` `Interaction` `Goal` |
| `learnpath/data/generator.py` | 确定性合成数据集（固定种子，可完全复现） | `generate_dataset(scale)` |
| `learnpath/data/loader.py` | 数据集拆分读写 | `load_dataset()` `save_dataset()` `load_or_build()` |
| `learnpath/data/fixtures.py` | 微型数据集，供单元测试与早期开发 | `build_tiny_dataset()` |
| `learnpath/knowledge/graph.py` | 知识点 DAG：环检测、先修闭包、拓扑分层、课程级先修推导、ECharts 导出 | `KnowledgeGraph` `CycleError` `GraphStats` |
| `learnpath/learner/bkt.py` | 贝叶斯知识追踪，掌握度更新与遗忘衰减 | `MasteryTracker` |
| `learnpath/learner/profile.py` | 多维学习者模型：掌握度、缺口、先修就绪度、风格/时间匹配、预期增益 | `LearnerModel` `ProfileSnapshot` |
| `learnpath/recommend/content.py` | 基于图谱缺口的内容推荐 | `ContentRecommender` |
| `learnpath/recommend/collaborative.py` | 隐式反馈矩阵 + TruncatedSVD 协同过滤 | `CollaborativeRecommender` |
| `learnpath/recommend/hybrid.py` | 多路召回 + 加权融合 + MMR 重排 + 可解释理由 | `HybridRecommender` `Recommendation` |
| `learnpath/planning/multiobjective.py` | 四目标建模、快速非支配排序、拥挤度、NSGA-II、TOPSIS | `evaluate_objectives` `nsga2` `select_compromise` |
| `learnpath/planning/planner.py` | 分层贪心集合覆盖 + 拓扑修复 + 周次装箱 + 里程碑 | `PathPlanner` `LearningPath` `PathStep` |
| `learnpath/prediction/features.py` | 无泄漏特征工程 | `build_features` |
| `learnpath/prediction/effect.py` | 完成概率分类 + 掌握度增益回归 + 可解释归因 | `EffectPredictor` `CompletionPrediction` `GainPrediction` |
| `learnpath/adaptive/engine.py` | 反馈驱动的掌握度更新与四类重规划触发 | `AdaptiveEngine` `FeedbackOutcome` `AdaptationResult` |
| `learnpath/social/cohort.py` | 复合相似度同伴匹配、能力互补小组组建 | `SocialMatcher` `PeerMatch` `StudyGroup` |
| `learnpath/analytics/dashboard.py` | 仪表板数据聚合 | `build_dashboard` |
| `learnpath/analytics/evaluation.py` | 留出法离线评测、多策略对比、消融实验 | `OfflineEvaluator` `build_evaluation` |

---

## 3. 关键数据流

### 3.1 冷启动：一次推荐是怎么产生出来的

```
① 加载 Dataset（data/*.json）与 KnowledgeGraph（缓存于服务层单例）
        ↓
② 构造 LearnerModel(learner, dataset, graph)
        ├─ 初始掌握度：learner.initial_mastery + 未给出者按先修就绪度推算先验
        └─ 回放 learner 的历史交互 → BKT 更新 → 得到当前掌握度向量
        ↓
③ 目标闭包：Pre(goal.target_kp_ids)  ← 决定「该学什么」的范围
        ↓
④ 缺口识别：G = { kp | mastery(kp) < MASTERY_GAP }
        ↓
⑤ 三路召回（限定在闭包内、覆盖 G 的课程）
        ├─ 内容召回：缺口覆盖增益 / 时间成本
        ├─ 协同召回：SVD 隐向量打分
        └─ 口碑召回：评分与报名量平滑
        ↓
⑥ 加权融合（five signals）→ 过滤（已学 / 先修不足）→ MMR 多样性重排 → Top-K
        ↓
⑦ 每条推荐附加：分项得分、预期增益、完成概率、风险等级、中文理由
```

### 3.2 路径规划数据流

```
缺口 G ──→ 候选课程 C ──→ graph.layers(G) 逐层贪心覆盖 ──→ 初始序列
                                                        ↓
                                  拓扑合法性修复（先修课必须在前）
                                                        ↓
             NSGA-II（种群=候选序列的邻域）──→ 帕累托前沿 ──→ TOPSIS 选解
                                                        ↓
                             周次装箱（按 time_budget_hours_per_week）
                                                        ↓
                                  LearningPath（steps / weekly_plan / milestones）
```

### 3.3 学习闭环

```
学习者提交反馈（进度 / 测验分 / 评分）
        ↓  POST /api/feedback
AdaptiveEngine
        ├─ MasteryTracker 更新掌握度（BKT）
        ├─ 判定四类触发条件
        └─ 必要时调用 PathPlanner.replan() 重排路径
        ↓
返回：掌握度增量 + 动作说明 + 新画像 + 新路径
```

---

## 4. 服务层设计

`app/services.py` 是**进程级单例容器**，负责三件事：

1. **懒加载 + 线程锁**：数据集、图谱、推荐器、预测器均只在首次请求时构造一次；
2. **依赖注入**：路由函数通过 `services.get_*()` 取用，路由本身保持极薄；
3. **优雅降级**：任一算法模块导入失败时，相关接口返回 HTTP 503 并给出明确 `detail`，
   服务本身仍可启动——支持分阶段交付与排障。

路由层按领域拆分（`meta / learners / recommend / planning / feedback / social /
analytics / graph / courses / evaluation`），每个文件只做：解析参数 → 调用服务 → 返回 JSON。

---

## 5. 表现层设计

- **无构建步骤**：原生 ES Module + 手写 CSS，`index.html` 直接可跑，评审环境零配置；
- **哈希路由**：`#profile / #recommend / #path / #dashboard / #social / #evaluation` 六个视图；
- **图表**：ECharts 5.5.1 本地引入（`web/vendor/echarts.min.js`），保证完全离线演示；
- **契约驱动**：前端只依赖 `docs/API.md` 定义的字段，不依赖后端实现细节；
- **离线演示模式**：后端算法模块尚未就绪时，前端自动回退到 `web/js/mock.js` 的契约示例数据，
  并在页面顶部显示提示条——保证界面永远可展示。

---

## 6. 关键设计取舍

| 决策 | 选择 | 理由 |
|---|---|---|
| 是否用深度学习 | **不用** | 题目为 ★★ 难度，核心矛盾在「图谱约束 + 多目标权衡」而非模型容量；无 GPU 环境下也能完整复现 |
| 知识点粒度 vs 课程粒度 | 以**知识点**为图谱节点，课程通过覆盖关系映射 | 课程粒度的先修关系难以表达"部分覆盖"，知识点粒度才能精确刻画缺口 |
| 多目标求解 | NSGA-II + TOPSIS | 四目标互相冲突且不可微，进化算法无需目标可导；TOPSIS 给出可解释的偏好选择 |
| 协同过滤实现 | TruncatedSVD | 无需安装 implicit/scipy 之外的重型库，且隐向量可直接用于相似学习者检索 |
| 数据集 | 程序化合成 + 入库 | 保证可复现、可离线、无隐私风险；生成机制本身也是可交付的工程资产 |
| 前后端耦合方式 | 单一 REST 契约文档 | 让四个人（数据/推荐/规划/前端）能并行开发，集成成本最低 |

---

## 7. 扩展点

- **换数据集**：只要满足 `schema.py` 契约，替换 `data/*.json` 即可，无需改算法；
- **加一路召回**：在 `recommend/hybrid.py` 增加召回函数与权重项（`RECOMMEND_WEIGHTS` 已预留结构）；
- **加一个目标函数**：在 `planning/multiobjective.py` 的 `evaluate_objectives` 增加一项，
  NSGA-II 与非支配排序无需改动；
- **换推荐/规划策略**：服务层通过接口取用实现，替换具体类即可（策略模式）。
