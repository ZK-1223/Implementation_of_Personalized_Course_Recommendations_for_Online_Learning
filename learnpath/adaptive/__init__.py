"""自适应调整引擎（实现要求第 6 条）。

对外契约：

    # learnpath/adaptive/engine.py
    @dataclass
    class FeedbackOutcome:
        learner_id: str
        course_id: str
        progress: float
        quiz_score: float | None
        rating: float | None
        hours_spent: float

    @dataclass
    class AdaptationResult:
        mastery_updates: dict[str, float]     # kp -> 新掌握度
        delta: dict[str, float]
        actions: list[str]                    # 中文动作说明，如「跳过已掌握课程」
        replanned: bool
        new_path: LearningPath | None
        reason: str

    class AdaptiveEngine:
        def __init__(self, dataset: Dataset, graph: KnowledgeGraph,
                     planner: PathPlanner, tracker: MasteryTracker) -> None
        def apply_feedback(self, model: LearnerModel,
                           outcome: FeedbackOutcome,
                           preset: str = "balanced") -> AdaptationResult
        def progress_report(self, model: LearnerModel,
                            path: LearningPath) -> dict

触发逻辑由 :class:`learnpath.config.AdaptiveThresholds` 控制。
"""

from __future__ import annotations

from learnpath.adaptive.engine import AdaptationResult, AdaptiveEngine, FeedbackOutcome

__all__ = ["AdaptiveEngine", "FeedbackOutcome", "AdaptationResult"]
