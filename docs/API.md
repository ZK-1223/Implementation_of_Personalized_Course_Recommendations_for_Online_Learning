# LearnPath REST API 契约

> 本文件是**前后端并行开发的唯一接口基准**。后端路由必须严格返回此处定义的字段名与类型，
> 前端只依赖此处字段，不得依赖后端内部实现细节。
>
> 所有接口前缀 `/api`；静态页面挂载在 `/`。
> 除特别说明外，所有接口均为 `GET`，返回 `application/json; charset=utf-8`。

---

## 0. 通用约定

| 项 | 约定 |
|---|---|
| 浮点精度 | 概率 / 分数统一保留 3 位小数，落在 `[0, 1]` |
| 时长单位 | 小时（`float`） |
| 周次 | 从 1 开始的整数 |
| 中文文本 | 直接返回 UTF-8 中文，不做拼音化 |
| 错误格式 | `{"detail": "错误说明"}`，HTTP 4xx/5xx |
| 分页 | `{"items": [...], "total": int, "page": int, "page_size": int}` |

---

## 1. 系统信息

### `GET /api/health`
```json
{"status": "ok", "version": "1.0.0", "dataset_loaded": true}
```

### `GET /api/meta`
前端首屏加载，一次拿全所有枚举与统计。
```json
{
  "stats": {
    "n_knowledge_points": 420, "n_kp_edges": 536, "n_courses": 560,
    "n_learners": 720, "n_goals": 8, "n_interactions": 14000,
    "n_completed": 6900, "completion_rate": 0.4929,
    "domains": ["编程开发", "数据与人工智能", "..."]
  },
  "graph": {
    "n_nodes": 420, "n_edges": 536, "n_roots": 21, "n_leaves": 63,
    "max_depth": 9, "avg_out_degree": 1.276, "domains": ["..."]
  },
  "domains": [
    {"name": "编程开发", "icon": "code", "n_kps": 62, "n_courses": 84}
  ],
  "goals": [
    {"goal_id": "goal_da", "name": "数据分析师", "target_role": "数据分析师",
     "description": "...", "n_target_kps": 12, "default_weeks": 12}
  ],
  "presets": [
    {"key": "balanced", "name": "均衡方案",
     "weights": {"gain": 0.4, "time": 0.25, "smoothness": 0.2, "fit": 0.15}}
  ],
  "learners": [
    {"learner_id": "L-0001", "nickname": "学员01", "goal_id": "goal_da",
     "goal_name": "数据分析师", "ability": 0.61, "time_budget_hours_per_week": 8.0}
  ],
  "modalities": [{"key": "video", "name": "视频课"}],
  "motivations": ["求职", "晋升", "兴趣", "学业", "转行"]
}
```

---

## 2. 学习者画像（实现要求 ①）

### `GET /api/learners`
分页 + 关键词检索。`?q=&goal_id=&page=1&page_size=20`

### `GET /api/learners/{learner_id}`
```json
{
  "learner": { "...": "Learner.to_dict()，见 learnpath/data/schema.py" },
  "goal": { "...": "Goal.to_dict()" },
  "profile": {
    "learner_id": "L-0001",
    "mastery": {"KP-01-01": 0.72},
    "mastery_names": [{"kp_id": "KP-01-01", "name": "变量与数据类型", "value": 0.72}],
    "domain_mastery": {"编程开发": 0.55},
    "target_kps": ["KP-02-01"],
    "target_kp_names": ["数据清洗"],
    "gaps": {"KP-02-03": 0.48},
    "gap_names": ["假设检验"],
    "readiness": 0.37,
    "style_vector": {"theory": 0.6, "practice": 0.4, "video": 0.3},
    "time_budget": 8.0,
    "total_available_hours": 96.0,
    "mastered_count": 5,
    "gap_count": 7,
    "completion_rate": 0.42
  },
  "history": [
    {"course_id": "C-0007", "title": "...", "domain": "编程开发",
     "progress": 1.0, "completed": true, "rating": 4.5,
     "hours_spent": 6.2, "quiz_score": 0.86, "timestamp": "2026-03-01T10:00:00"}
  ]
}
```

---

## 3. 智能推荐（实现要求 ①）

### `GET /api/recommend/{learner_id}?k=10`
```json
{
  "learner_id": "L-0001",
  "k": 10,
  "count": 10,
  "items": [
    {
      "rank": 1,
      "course_id": "C-0123",
      "title": "数据清洗实战：从脏数据到可用数据集",
      "domain": "数据与人工智能",
      "level": 2,
      "duration_hours": 8.5,
      "modality": "project",
      "modality_name": "项目实战",
      "rating": 4.7,
      "enroll_count": 18234,
      "price": 99.0,
      "kp_names": ["数据清洗", "缺失值处理"],
      "score": 0.873,
      "components": {
        "content": 0.91, "collaborative": 0.62, "quality": 0.88,
        "readiness": 0.79, "social": 0.55
      },
      "reasons": [
        "填补目标缺口「数据清洗」（当前掌握度 0.32）",
        "先修知识已掌握 3/4，可顺利开始",
        "与你背景相近的 12 位学员完成率 78%"
      ],
      "expected_gain": 0.164,
      "p_completion": 0.72,
      "risk_level": "中",
      "readiness": 0.79,
      "style_fit": 0.83,
      "time_fit": 0.91
    }
  ]
}
```

---

## 4. 学习路径规划（实现要求 ③④）

### `GET /api/path/{learner_id}?preset=balanced`
`preset ∈ {balanced, fast, thorough}`
```json
{
  "learner_id": "L-0001",
  "preset": "balanced",
  "preset_name": "均衡方案",
  "goal": {"goal_id": "goal_da", "name": "数据分析师"},
  "total_hours": 62.5,
  "weeks": 11,
  "total_gain": 2.84,
  "feasibility": 0.93,
  "objectives": {"gain": 0.71, "time": 0.66, "smoothness": 0.82, "fit": 0.75},
  "steps": [
    {"order": 1, "course_id": "C-0007", "title": "...", "domain": "编程开发",
     "level": 1, "hours": 4.5, "cumulative_hours": 4.5, "week": 1,
     "kp_names": ["数据清洗"], "expected_gain": 0.21, "modality": "video"}
  ],
  "weekly_plan": [{"week": 1, "hours": 4.5, "course_ids": ["C-0007"]}],
  "milestones": [{"week": 4, "label": "掌握 假设检验", "kp_names": ["假设检验"]}],
  "alternatives": [
    {"preset": "fast", "preset_name": "速通方案", "total_hours": 48.0,
     "weeks": 8, "total_gain": 2.31, "feasibility": 1.0,
     "objectives": {"gain": 0.58, "time": 0.88, "smoothness": 0.61, "fit": 0.7}}
  ]
}
```

### `GET /api/path/{learner_id}/pareto`
帕累托前沿，用于散点图展示「时间—收益」权衡。
```json
{
  "learner_id": "L-0001",
  "x": "hours", "y": "gain",
  "points": [
    {"id": "p1", "hours": 48.0, "gain": 2.31, "smoothness": 0.61,
     "fit": 0.7, "n_courses": 9, "is_selected": false, "preset": null}
  ]
}
```

---

## 5. 自适应反馈（实现要求 ⑥）

### `POST /api/feedback`
请求体：
```json
{"learner_id": "L-0001", "course_id": "C-0007",
 "progress": 1.0, "quiz_score": 0.85, "rating": 4.5, "hours_spent": 5.1}
```
响应：
```json
{
  "learner_id": "L-0001",
  "course_id": "C-0007",
  "mastery_updates": {"KP-02-01": 0.78},
  "delta": {"KP-02-01": 0.26},
  "actions": ["掌握度提升显著，已解锁后续课程 3 门", "路径已自动重排"],
  "replanned": true,
  "reason": "结课测验 0.85 高于阈值 0.60，按 BKT 更新掌握度并跳过冗余课程",
  "profile": {"...": "同 /api/learners/{id} 的 profile 字段"},
  "path": {"...": "同 /api/path 响应，已重规划"}
}
```

---

## 6. 社交学习（实现要求 ⑦）

### `GET /api/social/{learner_id}?k=6`
```json
{
  "learner_id": "L-0001",
  "peers": [
    {"learner_id": "L-0042", "nickname": "学员42", "ability": 0.71,
     "goal_name": "数据分析师", "similarity": 0.86,
     "shared_goal_kps": ["数据清洗", "假设检验"],
     "shared_courses": ["数据清洗实战"],
     "ahead_in": ["描述性统计"],
     "completed_count": 7}
  ],
  "groups": [
    {"group_id": "G-01", "kp_names": ["假设检验", "回归建模"],
     "members": [{"learner_id": "L-0001", "nickname": "学员01", "ability": 0.61}],
     "avg_completion": 0.68, "synergy": 0.74}
  ],
  "cohort_summary": {
    "n_peers": 6, "avg_similarity": 0.71,
    "cohort_completion_rate": 0.63, "your_completion_rate": 0.42
  }
}
```

---

## 7. 学习分析仪表板（实现要求 ⑧）

### `GET /api/analytics/{learner_id}`
```json
{
  "learner_id": "L-0001",
  "kpi": {
    "overall_progress": 0.42, "mastered_kps": 5, "total_target_kps": 12,
    "avg_mastery": 0.51, "readiness": 0.37,
    "planned_hours": 62.5, "available_hours": 96.0,
    "risk_courses": 2, "learning_streak_weeks": 5
  },
  "mastery_curve": [{"week": 0, "avg_mastery": 0.31, "mastered": 2}],
  "domain_radar": [{"domain": "编程开发", "mastery": 0.55, "target": 0.8}],
  "difficulty_distribution": [{"level": 1, "count": 12}],
  "modality_mix": [{"modality": "video", "name": "视频课", "hours": 21.5}],
  "time_allocation": [{"week": 1, "hours": 4.5, "planned": 8.0}],
  "completion_forecast": [
    {"order": 1, "course_id": "C-0007", "course_title": "...",
     "p_completion": 0.72, "risk_level": "中"}
  ],
  "gap_table": [
    {"kp_id": "KP-02-03", "name": "假设检验", "mastery": 0.32,
     "difficulty": 3, "domain": "数据与人工智能", "in_path": true}
  ],
  "history_activity": [{"week": 1, "hours": 5.2, "completed": 1}]
}
```

---

## 8. 知识图谱可视化

### `GET /api/knowledge-graph?domain=&learner_id=`
`domain` 可选（缺省返回全图，节点较多时前端按领域筛选）；`learner_id` 可选（回填掌握度染色）。
```json
{
  "nodes": [{"id": "KP-01-01", "name": "变量与数据类型", "category": 0,
             "domain": "编程开发", "difficulty": 1, "depth": 0,
             "mastery": 0.72, "symbolSize": 16, "highlight": false}],
  "links": [{"source": "KP-01-01", "target": "KP-01-02"}],
  "categories": ["编程开发", "数据与人工智能"],
  "focus": {"target_kps": ["KP-02-01"], "path_kps": ["KP-01-01"]}
}
```

---

## 9. 课程库

### `GET /api/courses?q=&domain=&level=&modality=&page=1&page_size=20`
返回分页结构，`items` 元素为 `Course.to_dict()` 并附加 `"kp_names": [...]`。

---

## 10. 系统评测（离线指标）

### `GET /api/evaluation`
```json
{
  "generated_at": "2026-10-29T18:00:00",
  "recommenders": [
    {"name": "热度基线 Popularity", "precision_at_k": 0.061, "recall_at_k": 0.184,
     "ndcg_at_k": 0.212, "map": 0.131, "coverage": 0.041,
     "diversity": 0.32, "novelty": 0.18, "hit_rate": 0.31},
    {"name": "内容推荐 Content", "...": "..."},
    {"name": "协同过滤 CF-SVD", "...": "..."},
    {"name": "混合推荐 Hybrid（本系统）", "...": "..."}
  ],
  "effect_model": {
    "task": "course_completion",
    "positive_rate": 0.49,
    "models": [
      {"name": "逻辑回归", "auc": 0.79, "accuracy": 0.72, "f1": 0.74,
       "cv_mean": 0.785, "cv_std": 0.011},
      {"name": "梯度提升树", "auc": 0.83, "accuracy": 0.76, "f1": 0.78,
       "cv_mean": 0.826, "cv_std": 0.008}
    ],
    "best": "梯度提升树",
    "features": [{"name": "learner_ability", "importance": 0.21}]
  },
  "gain_model": {
    "task": "mastery_gain_regression", "best": "梯度提升树",
    "models": [{"name": "梯度提升树", "mae": 0.041, "rmse": 0.058, "r2": 0.71}]
  },
  "ablation": [
    {"variant": "去掉知识图谱先修约束", "ndcg_at_k": 0.198, "delta": -0.031},
    {"variant": "去掉多目标优化", "ndcg_at_k": 0.206, "delta": -0.023}
  ],
  "notes": ["离线评测采用留一法划分：每位学习者保留 20% 的正反馈作为测试集", "..."]
}
```

---

## 11. 前端路由（静态页面）

| 路径 | 说明 |
|---|---|
| `/` | 单页应用入口 `web/index.html` |
| `/static/*` | 静态资源（css / js / vendor） |

单页应用通过 URL hash 切换视图：`#profile` / `#recommend` / `#path` / `#dashboard` / `#social` / `#evaluation`。
