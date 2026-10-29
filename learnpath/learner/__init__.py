"""学习者建模模块（实现要求第 1 条 + 掌握度知识追踪）。

对外契约：

    # learnpath/learner/bkt.py
    class MasteryTracker:
        \"\"\"贝叶斯知识追踪（BKT-lite）四参数模型的掌握度推断。\"\"\"
        def __init__(self, params: BKTParams = BKT) -> None
        def update(self, prior: float, observed_score: float,
                   kp_weight: float, ability: float) -> float
        def decay(self, mastery: float, idle_weeks: float) -> float

    # learnpath/learner/profile.py
    @dataclass
    class ProfileSnapshot:            # 供 API / 前端序列化
        learner_id: str
        mastery: dict[str, float]
        domain_mastery: dict[str, float]
        target_kps: list[str]
        gaps: dict[str, float]        # 缺口 kp -> 待提升幅度
        readiness: float              # 目标就绪度 0~1
        style_vector: dict[str, float]
        time_budget: float

    class LearnerModel:
        def __init__(self, learner: Learner, dataset: Dataset,
                     graph: KnowledgeGraph) -> None
        def mastery_of(self, kp_id: str) -> float
        def mastery_vector(self) -> dict[str, float]
        def knowledge_gap(self) -> dict[str, float]
        def courses_covered(self) -> frozenset[str]
        def readiness_for(self, course: Course) -> float       # 0~1 是否具备先修
        def style_fit(self, course: Course) -> float           # 0~1 偏好匹配
        def time_fit(self, course: Course) -> float            # 0~1 时间可承受度
        def expected_gain(self, course: Course) -> float       # 预期掌握度提升量
        def apply_outcome(self, course: Course, score: float,
                          effort_ratio: float) -> dict[str, float]
        def snapshot(self) -> ProfileSnapshot
"""
