"""路径规划与多目标优化（实现要求第 3、4 条）。

对外契约：

    # learnpath/planning/planner.py
    @dataclass
    class PathStep:
        order: int
        course: Course
        kp_ids: list[str]
        hours: float
        level: int
        cumulative_hours: float
        week: int
        expected_gain: float

    @dataclass
    class LearningPath:
        learner_id: str
        goal: Goal
        steps: list[PathStep]
        total_hours: float
        weeks: int
        total_gain: float
        objectives: dict[str, float]      # gain / time / smoothness / fit
        feasibility: float                # 0~1，时间预算可行度
        preset: str

    class PathPlanner:
        def __init__(self, dataset: Dataset, graph: KnowledgeGraph) -> None
        def build_candidates(self, model: LearnerModel) -> list[Course]
        def plan(self, model: LearnerModel, preset: str = "balanced",
                 max_courses: int = MAX_PATH_COURSES) -> LearningPath
        def pareto_front(self, model: LearnerModel) -> list[LearningPath]
        def replan(self, model: LearnerModel, done_course_ids: Iterable[str],
                   preset: str = "balanced") -> LearningPath

    # learnpath/planning/multiobjective.py
    def evaluate_objectives(steps, model) -> dict[str, float]
    def non_dominated_sort(candidates: list) -> list[list]      # 帕累托分层
    def crowding_distance(front: list) -> list[float]
    def nsga2(...) -> list[LearningPath]
    def select_compromise(front: list[LearningPath],
                          weights: ObjectiveWeights) -> LearningPath   # TOPSIS
"""

from __future__ import annotations

from learnpath.planning.multiobjective import (
    OBJECTIVE_KEYS,
    LearnerContext,
    ParetoSolution,
    clamp01,
    crowding_distance,
    evaluate_objectives,
    learner_context,
    non_dominated_sort,
    nsga2,
    polynomial_mutation,
    select_compromise,
    simulated_binary_crossover,
    topsis_scores,
)
from learnpath.planning.planner import LearningPath, PathPlanner, PathStep

__all__ = [
    "PathPlanner",
    "LearningPath",
    "PathStep",
    "OBJECTIVE_KEYS",
    "LearnerContext",
    "ParetoSolution",
    "clamp01",
    "crowding_distance",
    "evaluate_objectives",
    "learner_context",
    "non_dominated_sort",
    "nsga2",
    "polynomial_mutation",
    "select_compromise",
    "simulated_binary_crossover",
    "topsis_scores",
]
