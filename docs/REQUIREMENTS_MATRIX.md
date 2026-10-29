# 实现要求对照表

> 本表把课程设计题目第 5 题列出的 8 条实现要求，逐条映射到本项目的代码模块、
> 核心算法、HTTP 接口与前端视图，用于自检与答辩说明。
>
> 状态图例：✅ 已实现并自测通过

| # | 题目要求 | 对应模块 / 文件 | 核心算法与技术 | 对外接口 | 前端视图 | 报告章节 | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | **学习者建模**：构建包含知识水平、学习偏好、时间安排等多维度的学习者模型 | `learnpath/learner/profile.py`<br>`learnpath/learner/bkt.py` | 多维画像（知识水平 / 风格偏好 / 时间预算 / 能力 / 兴趣）；贝叶斯知识追踪（BKT）四参数模型更新掌握度；遗忘衰减；先修就绪度 sigmoid 映射 | `GET /api/learners/{id}` | 学习者画像 | 三、五 | ✅ |
| 2 | **课程知识图谱**：建立课程间的依赖关系和知识点关联，形成结构化的知识图谱 | `learnpath/knowledge/graph.py` | 知识点有向无环图（DAG）；Kahn 拓扑排序 + 环检测；先修闭包 / 后继闭包；拓扑分层；最长链深度；课程→知识点覆盖映射推导课程级先修 | `GET /api/knowledge-graph` | 学习路径（图谱局部图）<br>系统评测（全图） | 三、五 | ✅ |
| 3 | **学习路径规划**：基于知识图谱和学习目标，自动生成个性化的学习路径 | `learnpath/planning/planner.py` | 目标先修闭包 → 缺口识别 → 候选课程召回 → 分层贪心集合覆盖 → 拓扑合法性修复 → 周次装箱与里程碑生成 | `GET /api/path/{id}` | 学习路径 | 三、六 | ✅ |
| 4 | **多目标优化**：平衡学习效果、时间成本、难度梯度等多个目标 | `learnpath/planning/multiobjective.py` | 四目标建模（学习收益↓最大化 / 时间成本↓最小化 / 难度平滑度↑ / 风格匹配度↑）；NSGA-II（快速非支配排序 + 拥挤度 + 锦标赛选择 + 模拟二进制交叉 + 多项式变异）；TOPSIS 加权妥协解 | `GET /api/path/{id}/pareto` | 学习路径（帕累托散点图） | 三、四、七 | ✅ |
| 5 | **学习效果预测**：预测学习者对不同课程的掌握程度和完成概率 | `learnpath/prediction/effect.py`<br>`learnpath/prediction/features.py` | 特征工程（14+ 维，无标签泄漏）；完成概率分类（逻辑回归 vs 梯度提升树，5 折分层交叉验证，AUC/F1）；掌握度增益回归（Ridge vs 梯度提升树，MAE/R²）；特征重要性驱动的可解释归因 | 由 `/api/recommend/{id}`、`/api/analytics/{id}` 消费 | 智能推荐<br>学习分析仪表板 | 三、六、七 | ✅ |
| 6 | **自适应调整**：根据学习进度和反馈，动态调整推荐策略 | `learnpath/adaptive/engine.py` | 反馈驱动的掌握度更新 + 四类重规划触发器（低分补强 / 高分跳课 / 进度落后收敛 / 进度超前加速）；阈值见 `config.AdaptiveThresholds` | `POST /api/feedback` | 智能推荐（反馈提交后刷新） | 三、六 | ✅ |
| 7 | **社交学习功能**：集成同伴推荐和协作学习机制 | `learnpath/social/cohort.py` | 掌握度向量余弦相似度 + 目标一致性 + 兴趣重叠的复合相似度；能力互补度驱动的小组组建；群体完成率作为推荐信号 | `GET /api/social/{id}` | 社交学习 | 三、六 | ✅ |
| 8 | **学习分析仪表板**：提供学习进度可视化和个性化学习报告 | `learnpath/analytics/dashboard.py` | KPI 聚合、掌握度推进曲线、领域雷达、难度分布、模态构成、时间分配、完成概率预测、缺口表、历史活跃度 | `GET /api/analytics/{id}` | 学习分析仪表板 | 三、六 | ✅ |

## 附加工程能力（超出题目最低要求）

| 能力 | 位置 | 说明 |
|---|---|---|
| 混合推荐与可解释性 | `learnpath/recommend/hybrid.py` | 多路召回 + 加权融合 + MMR 多样性重排；每条推荐给出 4~5 条具体中文理由 |
| 协同过滤 | `learnpath/recommend/collaborative.py` | 隐式反馈矩阵 + TruncatedSVD 隐语义分解 + 相似学习者检索 |
| 离线评测体系 | `learnpath/analytics/evaluation.py` | 留一法协议下的 Precision/Recall/NDCG/MAP/Coverage/Diversity/Novelty/HitRate，四种策略横向对比 + 消融实验 |
| 可视化服务 | `app/` + `web/` | FastAPI 分层服务（引擎层 / 服务层 / 表现层），单页应用 6 个视图，ECharts 本地离线渲染 |
| 可复现性 | `learnpath/data/generator.py` | 固定随机种子 + 确定性生成，数据集本身入库，开箱即跑 |

## 需求覆盖自检结论

- 题目 8 条实现要求 **全部实现**，且每条都有独立的代码模块、接口与可视化出口；
- 不存在「仅演示、无实际算法」的模块：所有评分与指标均来自真实计算；
- 系统在无 GPU、普通笔记本环境下可完整运行与演示。
