"""个性化学习路径规划（实现要求 ③「基于知识图谱和学习目标自动生成个性化学习路径」）。

算法总览（7 步主流程，:meth:`PathPlanner.plan`）
------------------------------------------------
设学习目标为 ``goal``，目标知识点集合 ``T = goal.target_kp_ids``，
学习者能力 ``A``、每周时长 ``W``、截止周数 ``D``、**时间预算 ``B = W·D``**。

**第 1 步：先修闭包**
    ``K = graph.prerequisite_closure(T)`` —— 要掌握 ``T`` 必须先掌握的全部知识点。

**第 2 步：知识缺口**
    ``G = { kp ∈ K | mastery_of(kp) < MASTERY_GAP(0.60) }``
    已掌握的直接跳过，这是「个性化」的第一层体现：同一个目标，
    不同学习者得到的缺口集合完全不同。

**第 3 步：候选课程集**
    课程 ``c`` 入选当且仅当：
      (a) ``c`` 至少覆盖 ``G`` 中一个知识点（相关性）；
      (b) ``duration(c) ≤ B``（时间上装得下）；
      (c) **先修不悬空**：``prereq(c) \\ 已掌握 ⊆ G``
          —— ``c`` 缺的先修必须会被本路径补齐，否则这门课现在排进来也无法完成；
      (d) 若 ``prereq(c)`` 已全部掌握，仍要求 ``readiness_for(c) ≥ READINESS_MIN``。

**第 4 步：预算约束下的加权最大覆盖（0/1 背包式三阶段贪心）**

    问题建模::

        maximize   Σ_{kp ∈ 被覆盖} w(kp)          w(目标知识点) = 1.0，w(先修) = 0.5
        s.t.       Σ_i c_i · x_i ≤ B              ← 时间预算硬约束
                   x_i 选的课，先修必须被更早的课程覆盖

    三阶段（每步先检查「已花费 + 本课课时 ≤ B」，超预算就不选）：

    1. **目标知识点（权重 1.0）**：按偏好顺序挑还能覆盖新目标知识点的课，
       选中前用「本课课时 + 缺失先修的最便宜补齐成本」做保守可行性估计；
    2. **先修补齐（权重 0.5，硬需求）**：不动点迭代，直到待补集合为空或预算耗尽；
    3. **剩余缺口（软需求）**：用剩余预算的 ``extra_ratio`` 部分继续覆盖非目标缺口。
       ``extra_ratio`` 由 NSGA-II 决策向量的最后一个分量给出，于是
       ``fast``（时间权重大 → 小 ``extra_ratio``）路径更短、
       ``thorough``（收益权重大 → 大 ``extra_ratio``）覆盖更多。

    **预算不足时不会给出一条超支的假路径**：而是返回「预算内最优的部分路径」，
    并用 ``coverage_ratio`` / ``uncovered_target_kp_ids`` 如实列出没覆盖到的目标。

**第 5 步：拓扑合法性修复**
    课程级 DAG：若课程 ``A`` 覆盖的某知识点是课程 ``B`` 覆盖知识点的（传递）先修，
    则连边 ``A → B``。对该 DAG 做 Kahn 拓扑排序，同层（入度同时为 0）时按
    ``(覆盖知识点最小图谱深度, 课程难度, 课程 id)`` 出队，
    得到**满足全部先修约束且确定性可复现**的课程顺序。

**第 6 步：多目标优化（NSGA-II + TOPSIS）**
    决策向量 ``[w_1 … w_m, β]``：前 ``m`` 个分量是候选课程的「入选倾向」，
    末位 ``β`` 是「预算投入野心」。解码器（第 4 步）把任何实数向量变成一条
    结构合法的可行路径，于是 SBX 交叉 + 多项式变异可以直接在连续空间里搜索。
    NSGA-II 在前沿上铺开 (gain, time, smoothness, fit) 的权衡曲线，
    再用 TOPSIS 按 preset 权重选出主方案。
    （算法细节与公式见 :mod:`learnpath.planning.multiobjective`。）

**第 7 步：可行性评估与输出**
    ``feasibility = min(1, B / 路径总时长) × 周负载均衡度``；
    再把步骤按「累计时长 ÷ 每周时长」摊到周次上，输出 ``weekly_plan``
    （每周课时与课程）与 ``milestones``（每周掌握的目标知识点）。

复杂度：候选集构建 ``O(|C|·|K|)``；预算覆盖 ``O(|C|²·|kp|)``；
拓扑修复 ``O(n²·|kp|)``（``n ≤ 40``）；
NSGA-II ``O(代 × 种群 × |C|log|C|)``。
full 数据集（420 知识点 / 560 课程）上单次 ``plan`` 实测 < 1 s（目标 < 3 s）。

关键实现经验（报告可直接引用）
------------------------------
「边际收益 / 时长」的纯贪心在本数据上会被**浅层入门课完全支配**：入门课掌握度
缺口最大（约 0.6）、课时最短（3~6 h），性价比是深层课程的 2~3 倍。第一版实现因此
把预算全花在 layer-0 的入门课上，深度 2~6 的目标知识点课程一门都进不了路径，
产出「走不到目标的路径」。修复方式就是上面的**加权最大覆盖**：目标知识点权重
1.0、先修权重 0.5，并且把预算做成硬约束，从根上改变贪心的取舍。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

from learnpath.config import (
    MASTERY_GAP,
    MAX_PATH_COURSES,
    NSGA_CROSSOVER_RATE,
    NSGA_GENERATIONS,
    NSGA_MUTATION_RATE,
    NSGA_POPULATION,
    PLAN_PRESETS,
    PLAN_PRESET_NAMES,
    EVAL_SEED,
)
from learnpath.data.schema import Course, Dataset, Goal
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.planning.multiobjective import (
    LearnerContext,
    ParetoSolution,
    evaluate_objectives,
    learner_context,
    non_dominated_sort,
    nsga2,
    select_compromise,
    topsis_scores,
    weights_of,
)

# ============================================================
# 规划超参数（报告可直接引用）
# ============================================================

#: 候选课程的先修「已掌握率」下限（低于此值且先修无法在路径内补齐则剔除）
READINESS_MIN: Final[float] = 0.25

#: 候选池裁剪：每个缺口知识点最多保留的候选课程数（按 收益/时长 降序）。
#: 把 NSGA-II 的决策维度控制在 ~100 量级，保证单次规划 < 3 s。
CANDIDATES_PER_GAP: Final[int] = 4

#: 种子个体中「被贪心算法选中」课程的入选倾向
SEED_SELECTED: Final[float] = 0.95

#: 种子个体中「未被贪心算法选中」课程的入选倾向
SEED_UNSELECTED: Final[float] = 0.05

#: 输出的里程碑条数上限
MAX_MILESTONES: Final[int] = 12

#: NSGA-II 搜索结果缓存的容量上限（按 FIFO 淘汰，避免进程常驻时无限增长）
MAX_SEARCH_MEMO: Final[int] = 64

#: 数值保护
_EPS: Final[float] = 1e-9


# ============================================================
# 数据结构
# ============================================================


@dataclass(slots=True)
class PathStep:
    """学习路径中的一个步骤（= 一门课程在路径中的排位与排期）。

    Attributes:
        order: 全局顺序，从 1 开始，**满足先修约束**（见模块 docstring 第 5 步）。
        course: 课程实体。
        kp_ids: 该课程覆盖的知识点 id 列表。
        hours: 该课程预计投入小时数。
        level: 课程难度等级（1~5）。
        cumulative_hours: 含本课在内的累计投入小时数。
        week: 该课程**开始**的周次（从 1 开始）。
        expected_gain: 该课程的预期掌握度增益（``LearnerModel.expected_gain``）。
    """

    order: int
    course: Course
    kp_ids: list[str]
    hours: float
    level: int
    cumulative_hours: float
    week: int
    expected_gain: float

    @property
    def course_id(self) -> str:
        return self.course.course_id

    def to_dict(self, graph: KnowledgeGraph | None = None) -> dict[str, Any]:
        """序列化为 ``docs/API.md`` 第 4 节的 ``steps[i]``。"""
        return {
            "order": self.order,
            "course_id": self.course.course_id,
            "title": self.course.title,
            "domain": self.course.domain,
            "level": int(self.level),
            "hours": round(float(self.hours), 2),
            "cumulative_hours": round(float(self.cumulative_hours), 2),
            "week": int(self.week),
            "kp_names": list(graph.names_of(self.kp_ids))
            if graph is not None
            else list(self.kp_ids),
            "expected_gain": round(float(self.expected_gain), 4),
            "modality": self.course.modality,
        }


@dataclass(slots=True)
class LearningPath:
    """一条完整的个性化学习路径（可用 ``docs/API.md`` §4 直接序列化）。

    Attributes:
        learner_id: 学习者 id。
        goal: 该学习者追求的 :class:`~learnpath.data.schema.Goal`。
        steps: 有序步骤列表。
        total_hours: 路径总时长（小时）——**恒 ≤ budget_hours**。
        weeks: 按每周时长摊算的所需周数。
        total_gain: 路径累计预期掌握度增益（**原始量纲**，是各步之和）。
        objectives: 四个归一化目标 ``gain/time/smoothness/fit``，均 ∈ [0,1]。
        feasibility: 时间预算可行度 ∈ [0,1]（``min(1,B/总时长) × 周负载均衡度``）。
        preset: 生成该路径的偏好 preset（前沿上未被任何预设选中的解为 ``""``）。
        weekly_plan: ``[{"week": int, "hours": float, "course_ids": [...]}]``。
        milestones: ``[{"week": int, "label": str, "kp_names": [...]}]``。
        is_selected: 是否为**当前请求 preset 下的 TOPSIS 妥协解**。
        coverage: 覆盖率诊断（见 :meth:`PathPlanner._make_path`）。
        coverage_ratio: 目标知识点达成率 = (路径覆盖 ∪ 本来已掌握) / 目标总数。
        covered_target_kp_ids: 路径实际覆盖到的目标知识点 id。
        uncovered_target_kp_ids: 仍未覆盖且未掌握的目标知识点 id。
        covered_target_kp_names: 与 ``covered_target_kp_ids`` **一一对应**的中文名称
            （构造时就用知识点图翻译好，服务层直接读属性即可，不需要再传 graph）。
        uncovered_target_kp_names: 与 ``uncovered_target_kp_ids`` 一一对应的中文名称。
        budget_hours: 学习者时间预算（``每周时长 × 截止周数``）。
        over_budget: 是否超预算（正常情况下恒为 ``False``，保留作防御性断言）。
    """

    learner_id: str
    goal: Goal
    steps: list[PathStep]
    total_hours: float
    weeks: int
    total_gain: float
    objectives: dict[str, float]
    feasibility: float
    preset: str
    weekly_plan: list[dict[str, Any]] = field(default_factory=list)
    milestones: list[dict[str, Any]] = field(default_factory=list)
    is_selected: bool = False
    coverage: dict[str, Any] = field(default_factory=dict)
    coverage_ratio: float = 0.0
    covered_target_kp_ids: list[str] = field(default_factory=list)
    uncovered_target_kp_ids: list[str] = field(default_factory=list)
    covered_target_kp_names: list[str] = field(default_factory=list)
    uncovered_target_kp_names: list[str] = field(default_factory=list)
    budget_hours: float = 0.0
    over_budget: bool = False

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def course_ids(self) -> list[str]:
        return [step.course.course_id for step in self.steps]

    @property
    def kp_ids(self) -> list[str]:
        """路径覆盖的全部知识点（去重保序）。"""
        seen: dict[str, None] = {}
        for step in self.steps:
            for kp_id in step.kp_ids:
                seen.setdefault(kp_id, None)
        return list(seen)

    def to_dict(
        self,
        graph: KnowledgeGraph | None = None,
        preset_name: str | None = None,
    ) -> dict[str, Any]:
        """序列化为 ``docs/API.md`` 第 4 节的 ``/api/path`` 响应体。"""
        return {
            "learner_id": self.learner_id,
            "preset": self.preset,
            "preset_name": preset_name
            if preset_name is not None
            else PLAN_PRESET_NAMES.get(self.preset, self.preset),
            "goal": {"goal_id": self.goal.goal_id, "name": self.goal.name},
            "total_hours": round(float(self.total_hours), 2),
            "weeks": int(self.weeks),
            "total_gain": round(float(self.total_gain), 4),
            "feasibility": round(float(self.feasibility), 3),
            "objectives": {k: round(float(v), 3) for k, v in self.objectives.items()},
            "coverage_ratio": round(float(self.coverage_ratio), 3),
            "budget_hours": round(float(self.budget_hours), 2),
            "over_budget": bool(self.over_budget),
            "covered_target_kp_ids": list(self.covered_target_kp_ids),
            "covered_target_kp_names": list(self.covered_target_kp_names)
            or (list(graph.names_of(self.covered_target_kp_ids)) if graph is not None else []),
            "uncovered_target_kp_ids": list(self.uncovered_target_kp_ids),
            "uncovered_target_kp_names": list(self.uncovered_target_kp_names)
            or (list(graph.names_of(self.uncovered_target_kp_ids)) if graph is not None else []),
            "steps": [step.to_dict(graph) for step in self.steps],
            "weekly_plan": [
                {
                    "week": int(item["week"]),
                    "hours": round(float(item["hours"]), 2),
                    "course_ids": list(item["course_ids"]),
                }
                for item in self.weekly_plan
            ],
            "milestones": [
                {
                    "week": int(item["week"]),
                    "label": item["label"],
                    "kp_names": list(item["kp_names"]),
                }
                for item in self.milestones
            ],
            "is_selected": bool(self.is_selected),
            "coverage": {
                key: (round(value, 3) if isinstance(value, float) else value)
                for key, value in self.coverage.items()
            },
            "alternatives": [],
        }


@dataclass(slots=True)
class _StepEval:
    """NSGA-II 高频评估用的轻量步骤视图（只含目标函数需要的 4 个字段）。"""

    course: Course
    hours: float
    level: int
    expected_gain: float


@dataclass(slots=True)
class _Search:
    """一次 NSGA-II 搜索的完整上下文（供 ``plan`` / ``pareto_front`` 复用）。"""

    pool: list[Course]
    gap_ids: tuple[str, ...]
    gap_mag: dict[str, float]
    mastered: frozenset[str]
    ctx: LearnerContext
    front: list[ParetoSolution]
    history: list[dict[str, float]]
    gain_cache: dict[str, float]
    fit_cache: dict[str, tuple[float, float]]
    budget: float
    weekly_budget: float
    deadline_weeks: int
    reference_gain: float = 0.0
    target_ids: tuple[str, ...] = ()


# ============================================================
# 规划器
# ============================================================


class PathPlanner:
    """基于知识图谱的个性化学习路径规划器。

    Args:
        dataset: 全量数据集。
        graph: 知识点图（:class:`~learnpath.knowledge.graph.KnowledgeGraph`）。

    Note:
        本类**无状态污染**：不修改 ``model``；所有重规划都通过显式传入
        ``done_course_ids`` 的 :meth:`replan` 完成。
        内部对「课程先修闭包」与「NSGA-II 搜索结果」做了缓存，
        因此同一次请求里连续调用 :meth:`plan` / :meth:`pareto_front` 不会重复搜索。
    """

    def __init__(self, dataset: Dataset, graph: KnowledgeGraph) -> None:
        self.dataset = dataset
        self.graph = graph
        self.courses: list[Course] = sorted(dataset.courses.values(), key=lambda c: c.course_id)
        self._course_by_id: dict[str, Course] = {c.course_id: c for c in self.courses}
        self._prereq_cache: dict[str, frozenset[str]] = {}
        self._depth_key_cache: dict[str, int] = {}
        self._search_memo: dict[Any, _Search] = {}

        # kp -> 覆盖该知识点的课程（候选集构建用倒排索引）
        self._kp_courses: dict[str, list[Course]] = {}
        for course in self.courses:
            for kp_id in course.kp_ids:
                self._kp_courses.setdefault(kp_id, []).append(course)

    # --------------------------------------------------------
    # 基础查询
    # --------------------------------------------------------
    def course_prerequisites(self, course: Course) -> frozenset[str]:
        """课程要求的先修知识点（不含课程自身覆盖的知识点），带缓存。"""
        cached = self._prereq_cache.get(course.course_id)
        if cached is not None:
            return cached
        own = set(course.kp_ids)
        prereq = frozenset(self.graph.prerequisite_closure(course.kp_ids) - own)
        self._prereq_cache[course.course_id] = prereq
        return prereq

    def _course_depth_key(self, course: Course) -> int:
        """课程在知识图谱中的「起始层级」= 其覆盖知识点的最小深度。"""
        cached = self._depth_key_cache.get(course.course_id)
        if cached is not None:
            return cached
        depth = min(self.graph.depth(kp_id) for kp_id in course.kp_ids)
        self._depth_key_cache[course.course_id] = depth
        return depth

    @staticmethod
    def learner_of(model: Any) -> Any:
        """从 ``LearnerModel`` 取 :class:`~learnpath.data.schema.Learner`。"""
        learner = getattr(model, "learner", None)
        if learner is None:  # pragma: no cover - 契约保证存在
            raise AttributeError("LearnerModel 缺少公开属性 learner")
        return learner

    def goal_of(self, model: Any) -> Goal:
        """取学习者的目标；优先用 ``model.goal``，回退到 dataset 查表。"""
        goal = getattr(model, "goal", None)
        if goal is not None:
            return goal
        return self.dataset.goals[self.learner_of(model).goal_id]

    # --------------------------------------------------------
    # 第 1~3 步：闭包 / 缺口 / 候选课程
    # --------------------------------------------------------
    def knowledge_gaps(
        self,
        model: Any,
        extra_mastered: Iterable[str] = (),
    ) -> tuple[frozenset[str], dict[str, float], frozenset[str]]:
        """计算目标先修闭包、知识缺口与已掌握集合。

        Args:
            model: 学习者模型。
            extra_mastered: 额外视为「已掌握」的知识点（:meth:`replan` 中由
                已完成课程覆盖的知识点注入）。

        Returns:
            ``(closure, gaps, mastered)``，其中 ``gaps`` 为 ``kp -> 缺口幅度``。
        """
        goal = self.goal_of(model)
        closure = self.graph.prerequisite_closure(goal.target_kp_ids)
        extra = frozenset(extra_mastered)

        required = float(getattr(goal, "required_mastery", 0.80) or 0.80)
        gaps: dict[str, float] = {}
        mastered: set[str] = set()
        for kp_id in closure:
            mastery = float(model.mastery_of(kp_id))
            if kp_id in extra or mastery >= MASTERY_GAP:
                mastered.add(kp_id)
                continue
            gaps[kp_id] = max(0.0, required - mastery)

        return closure, gaps, frozenset(mastered)

    def build_candidates(
        self,
        model: Any,
        extra_mastered: Iterable[str] = (),
    ) -> list[Course]:
        """构建候选课程集（模块 docstring 第 3 步的四条判据）。

        Returns:
            按 ``(收益/时长)`` 降序、``course_id`` 升序排定的候选课程列表。
        """
        learner = self.learner_of(model)
        budget = float(getattr(learner, "total_available_hours", 0.0) or 0.0)
        _closure, gaps, mastered = self.knowledge_gaps(model, extra_mastered)
        if not gaps:
            return []
        gap_ids = frozenset(gaps)

        # 只遍历「覆盖了某个缺口知识点」的课程（倒排索引）
        seen: set[str] = set()
        candidates: list[Course] = []
        for kp_id in gap_ids:
            for course in self._kp_courses.get(kp_id, ()):
                if course.course_id in seen:
                    continue
                seen.add(course.course_id)
                if budget > 0 and float(course.duration_hours) > budget:
                    continue
                unresolved = self.course_prerequisites(course) - mastered
                if unresolved and not unresolved <= gap_ids:
                    continue
                if not unresolved and float(model.readiness_for(course)) < READINESS_MIN:
                    continue
                candidates.append(course)

        def _score(course: Course) -> tuple[float, str]:
            gain = sum(gaps[kp] * w for kp, w in course.kp_weights.items() if kp in gaps)
            ratio = gain / max(0.5, float(course.duration_hours))
            return (-ratio, course.course_id)

        candidates.sort(key=_score)
        return candidates

    def _trim_pool(
        self,
        candidates: Sequence[Course],
        gap_mag: Mapping[str, float],
    ) -> list[Course]:
        """候选池裁剪：每个缺口知识点保留至多 ``CANDIDATES_PER_GAP`` 门课。

        排序基准仍是「边际收益 / 时长」，与覆盖算法的评价口径一致，
        因此裁剪不会丢掉优质课程，同时把 NSGA-II 的决策维度压到 ~100 维以内。
        """
        quota: dict[str, int] = {}
        pool: list[Course] = []
        for course in candidates:
            useful = [
                kp
                for kp in course.kp_ids
                if kp in gap_mag and quota.get(kp, 0) < CANDIDATES_PER_GAP
            ]
            if not useful:
                continue
            pool.append(course)
            for kp in useful:
                quota[kp] = quota.get(kp, 0) + 1
        return pool

    # --------------------------------------------------------
    # 第 4 步：预算约束下的加权最大覆盖（0/1 背包式三阶段贪心）
    # --------------------------------------------------------
    def _budgeted_cover(
        self,
        order: Sequence[int],
        pool: Sequence[Course],
        pool_kp_ids: Sequence[tuple[str, ...]],
        pool_prereq: Sequence[frozenset[str]],
        pool_hours: Sequence[float],
        target_ids: frozenset[str],
        gap_ids: frozenset[str],
        budget: float,
        max_courses: int,
        extra_ratio: float = 1.0,
    ) -> list[Course]:
        """**在时间预算内**做加权最大覆盖，产出课程集合。

        问题建模（0/1 背包 + 集合覆盖）::

            maximize   Σ_{kp ∈ 被覆盖} w(kp)      w(目标) = 1.0, w(先修) = 0.5
            s.t.       Σ_i c_i · x_i ≤ B          （时间预算硬约束）
                       选中课程的先修必须被更早的课程覆盖（拓扑合法性）

        三阶段（每步都先检查「已花费 + 本课课时 ≤ 预算」）：

        **阶段 1 —— 目标知识点（权重 1.0）**
            按偏好顺序挑「还能覆盖新目标知识点」的课。选中前用
            ``本课课时 + 其缺失先修的最便宜补齐成本`` 做**保守可行性估计**，
            估计超预算就跳过（换下一门能覆盖该目标的课），
            避免选了深课却没钱补先修、最后得到一条不可执行的路径。

        **阶段 2 —— 先修补齐（权重 0.5，硬需求）**
            不动点迭代：每选入一门课就把它的先修缺口并入待补集合，
            直到待补集合为空或预算耗尽。

        **阶段 3 —— 剩余缺口（软需求，受 ``extra_ratio`` 控制）**
            把剩余预算的 ``extra_ratio`` 部分用于覆盖更多非目标缺口。
            ``fast``（时间权重大 → 倾向小 ``extra_ratio``）到此基本停手，
            ``thorough``（收益权重大 → 倾向大 ``extra_ratio``）把预算用满。

        Args:
            order: 候选课程下标的偏好顺序（NSGA-II 里由决策向量降序给出）。
            pool: 候选课程池。
            pool_kp_ids: 每门候选课覆盖的知识点。
            pool_prereq: 每门候选课的先修知识点集合（已与缺口集求交）。
            pool_hours: 每门候选课的课时。
            target_ids: 目标知识点集合。
            gap_ids: 全部缺口知识点集合。
            budget: 时间预算（小时）——**硬约束**，结果课时之和不会超过它。
            max_courses: 课程数上限。
            extra_ratio: 阶段 3 允许动用的剩余预算比例 ``∈ [0,1]``。

        Returns:
            选出的课程列表（尚未做拓扑排序），课时之和 **≤ budget**。
        """
        # 每个知识点「最便宜的课程课时」，用于快速估计补齐先修的成本
        cheapest: dict[str, float] = {}
        for index, kps in enumerate(pool_kp_ids):
            hours = pool_hours[index]
            for kp in kps:
                if hours < cheapest.get(kp, math.inf):
                    cheapest[kp] = hours

        chosen: list[Course] = []
        chosen_idx: set[int] = set()
        covered: set[str] = set()
        required: set[str] = set()
        spent = 0.0
        targets = set(target_ids & gap_ids)

        def _estimate(index: int) -> float:
            """本课 + 其缺失先修的「最便宜补齐」成本（保守下界估计）。"""
            cost = pool_hours[index]
            for kp in pool_prereq[index] - covered:
                cost += cheapest.get(kp, 0.0)
            return cost

        def _take(index: int) -> None:
            nonlocal spent
            chosen.append(pool[index])
            chosen_idx.add(index)
            spent += pool_hours[index]
            covered.update(pool_kp_ids[index])
            # 刚被这门课补上的先修需求要从待补集合里移除（否则会过度覆盖）
            required.difference_update(pool_kp_ids[index])
            # 新课程自身也会带来新的先修需求（级联到阶段 2）
            required.update(pool_prereq[index] - covered)

        # ---- 阶段 1：目标知识点（权重 1.0） ----
        pending = set(targets)
        for index in order:
            if len(chosen) >= max_courses or not pending:
                break
            if index in chosen_idx:
                continue
            hits = [kp for kp in pool_kp_ids[index] if kp in pending]
            if not hits:
                continue
            if spent + pool_hours[index] > budget + _EPS:
                continue
            if spent + _estimate(index) > budget + _EPS:
                continue
            _take(index)
            pending.difference_update(hits)

        # ---- 阶段 2：先修补齐（权重 0.5，硬需求） ----
        while required and len(chosen) < max_courses:
            progressed = False
            for index in order:
                if len(chosen) >= max_courses:
                    break
                if index in chosen_idx:
                    continue
                if not any(kp in required for kp in pool_kp_ids[index]):
                    continue
                if spent + pool_hours[index] > budget + _EPS:
                    continue
                _take(index)
                progressed = True
            if not progressed:
                break

        # ---- 阶段 3：剩余缺口（软需求，按 extra_ratio 动用剩余预算） ----
        limit = spent + max(0.0, (budget - spent) * max(0.0, min(1.0, extra_ratio)))
        remaining = set(gap_ids - covered)
        for index in order:
            if len(chosen) >= max_courses or not remaining:
                break
            if index in chosen_idx:
                continue
            hits = [kp for kp in pool_kp_ids[index] if kp in remaining]
            if not hits:
                continue
            if spent + pool_hours[index] > limit + _EPS:
                continue
            _take(index)
            remaining.difference_update(hits)

        return chosen

    def greedy_cover(
        self,
        model: Any,
        candidates: Sequence[Course],
        gap_mag: Mapping[str, float],
        budget: float,
        max_courses: int = MAX_PATH_COURSES,
        target_ids: Iterable[str] = (),
    ) -> list[Course]:
        """「性价比优先」的确定性基线解（NSGA-II 的种子个体）。

        排序基准：``边际收益 / 时长``（收益按覆盖知识点的缺口幅度 × 覆盖权重求和），
        再走与解码器相同的预算约束三阶段覆盖流程；
        口碑只做 < 3% 的微调用于打破平局。
        """
        del model  # 保留参数以对齐解码器签名
        if not gap_mag:
            return []
        ids = [c.course_id for c in candidates]
        pool_kp = [tuple(c.kp_ids) for c in candidates]
        pool_prereq = [
            frozenset(self.course_prerequisites(c) & set(gap_mag)) for c in candidates
        ]
        pool_hours = [float(c.duration_hours) for c in candidates]

        def _score(index: int) -> float:
            course = candidates[index]
            gain = 0.0
            for kp in pool_kp[index]:
                gap = gap_mag.get(kp, 0.0)
                if gap > 0.0:
                    gain += gap * course.kp_weights.get(kp, 1.0)
            ratio = gain / max(0.5, float(course.duration_hours))
            return ratio * (1.0 + 0.02 * (float(course.rating) - 4.0))

        order = sorted(range(len(candidates)), key=lambda i: (-_score(i), ids[i]))
        return self._budgeted_cover(
            order,
            candidates,
            pool_kp,
            pool_prereq,
            pool_hours,
            frozenset(target_ids),
            frozenset(gap_mag),
            budget,
            max_courses,
            extra_ratio=1.0,
        )

    # --------------------------------------------------------
    # 第 5 步：拓扑合法性修复
    # --------------------------------------------------------
    def order_courses(self, courses: Sequence[Course]) -> list[Course]:
        """按先修约束给课程排序（拓扑合法性修复）。

        规则：若课程 ``A`` 覆盖的某知识点是课程 ``B`` 覆盖知识点的（传递）先修，
        则 ``A`` 必须排在 ``B`` 之前。实现为课程级 DAG 的 Kahn 拓扑排序，
        同层候选按 ``(覆盖知识点最小图谱深度, 课程难度, 课程 id)`` 出队，
        保证结果**确定性可复现**。

        若因共享知识点出现环（理论上应由 :meth:`KnowledgeGraph.validate` 拦截），
        退化为按同一 key 全排序。
        """
        items = list(courses)
        n = len(items)
        if n <= 1:
            return items

        own = [set(c.kp_ids) for c in items]
        prereq = [set(self.course_prerequisites(c)) for c in items]

        succ: list[list[int]] = [[] for _ in range(n)]
        indeg = [0] * n
        for j in range(n):
            need = prereq[j]
            if not need:
                continue
            for i in range(n):
                if i == j:
                    continue
                if own[i] & need:
                    succ[i].append(j)
                    indeg[j] += 1

        heap: list[tuple[tuple[int, int, str], int]] = []
        for i in range(n):
            if indeg[i] == 0:
                course = items[i]
                key = (self._course_depth_key(course), int(course.level), course.course_id)
                heapq.heappush(heap, (key, i))

        ordered: list[Course] = []
        while heap:
            _key, i = heapq.heappop(heap)
            ordered.append(items[i])
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    course = items[j]
                    key = (self._course_depth_key(course), int(course.level), course.course_id)
                    heapq.heappush(heap, (key, j))

        if len(ordered) != n:  # pragma: no cover - 图已校验无环，理论不可达
            ordered = sorted(
                items,
                key=lambda c: (self._course_depth_key(c), int(c.level), c.course_id),
            )
        return ordered

    # --------------------------------------------------------
    # gain 目标的归一化基准
    # --------------------------------------------------------
    def reference_gain(
        self,
        candidates: Sequence[Course],
        gap_mag: Mapping[str, float],
        gain_cache: Mapping[str, float],
        budget: float,
    ) -> float:
        """``gain`` 目标的归一化基准：**预算内理想覆盖增益**。

        两步构造：

        1. 先确定「预算内**负担得起**的缺口知识点集合 ``A``」：
           把候选课程按「收益 / 时长」降序排列，在 ``budget`` 内依次累计课时，
           把这些课程覆盖的缺口知识点收集为 ``A``。预算越小，``A`` 越小。
        2. 对 ``kp ∈ A`` 求「单门课能给它带来的最大增益份额」::

               share(c, k) = expected_gain(c) · w_ck / Σ_{k'∈c∩G} w_ck'
               ideal(k)    = max_{c ∋ k} share(c, k)

           ``reference_gain = Σ_{k∈A} ideal(k)``。

        为什么不用「全部缺口幅度之和」或「全缺口理想收益」：
            前者把 ``gain`` 压到 0.1 量级；后者对预算紧张的学习者不可达，
            同样会把 ``gain`` 压扁。以「预算内负担得起的理想收益」为分母，
            ``gain`` 的语义是**「这条路径取得了同预算下理想覆盖收益的百分之多少」**，
            量级稳定落在 0.4~0.9，且对预算大小不敏感。
        """
        if not candidates or not gap_mag:
            return 0.0
        gap_set = set(gap_mag)

        by_efficiency = sorted(
            candidates,
            key=lambda c: (
                -gain_cache.get(c.course_id, 0.0) / max(0.5, float(c.duration_hours)),
                c.course_id,
            ),
        )
        affordable: set[str] = set()
        spent = 0.0
        for course in by_efficiency:
            duration = float(course.duration_hours)
            if spent + duration > budget + _EPS:
                continue
            spent += duration
            affordable.update(kp for kp in course.kp_ids if kp in gap_set)
        if not affordable:
            affordable = set(gap_set)

        best: dict[str, float] = {kp: 0.0 for kp in affordable}
        for course in candidates:
            gain = gain_cache.get(course.course_id, 0.0)
            if gain <= 0.0:
                continue
            weights = course.kp_weights
            hits = [kp for kp in course.kp_ids if kp in affordable]
            if not hits:
                continue
            total_weight = sum(float(weights.get(kp, 1.0)) for kp in hits)
            for kp in hits:
                share = (
                    gain * float(weights.get(kp, 1.0)) / total_weight
                    if total_weight > _EPS
                    else gain / len(hits)
                )
                if share > best[kp]:
                    best[kp] = share
        return float(sum(best.values()))

    # --------------------------------------------------------
    # NSGA-II 解码
    # --------------------------------------------------------
    def _decode_chromosome(
        self,
        chromosome: np.ndarray,
        pool: Sequence[Course],
        pool_kp_ids: Sequence[tuple[str, ...]],
        pool_prereq: Sequence[frozenset[str]],
        pool_hours: Sequence[float],
        target_ids: frozenset[str],
        gap_ids: frozenset[str],
        budget: float,
        max_courses: int,
    ) -> list[Course]:
        """把实数决策向量解码成课程集合。

        决策向量结构：``[w_1 … w_m, β]``。前 ``m`` 个分量是候选课程的**入选倾向**
        （``[0,1]`` 实数，可直接做 SBX 与多项式变异），末位 ``β ∈ [0,1]`` 是
        **预算投入野心**，乘在「阶段 3 可用剩余预算」上：

        * ``β = 0`` → 只在目标与先修上花钱，路径尽可能短（对应 ``fast``）；
        * ``β = 1`` → 把剩余预算全部用于覆盖更多缺口（对应 ``thorough``）。

        解码器（:meth:`_budgeted_cover`）保证总课时 ≤ 预算、先修不悬空、目标优先。
        多样性来源：不同倾向向量意味着「同一个知识点选哪门课」
        （短课省钱 / 长课高收益 / 高口碑 / 高风格匹配）以及「目标能装上几个」。
        """
        m = len(pool)
        order = np.argsort(-chromosome[:m], kind="stable")
        beta = float(chromosome[m]) if chromosome.shape[0] > m else 1.0
        beta = max(0.0, min(1.0, beta))
        return self._budgeted_cover(
            [int(i) for i in order],
            pool,
            pool_kp_ids,
            pool_prereq,
            pool_hours,
            target_ids,
            gap_ids,
            budget,
            max_courses,
            extra_ratio=beta,
        )

    def _evaluate_courses(
        self,
        courses: Sequence[Course],
        model: Any,
        search_gain: dict[str, float],
        fit_cache: dict[str, tuple[float, float]],
        ctx: LearnerContext,
        reference_gain: float | None = None,
    ) -> dict[str, float]:
        """把课程列表换算成四目标值（内部走轻量步骤视图 + 缓存）。"""
        steps = [
            _StepEval(
                course=course,
                hours=float(course.duration_hours),
                level=int(course.level),
                expected_gain=search_gain.get(course.course_id, 0.0),
            )
            for course in courses
        ]
        return evaluate_objectives(
            steps, model, cache=fit_cache, ctx=ctx, reference_gain=reference_gain
        )

    def _search_key(self, model: Any, extra_mastered: Iterable[str]) -> tuple[Any, ...]:
        """搜索结果缓存的键：学习者 + 缺口签名 + 掌握度指纹。"""
        _closure, gaps, mastered = self.knowledge_gaps(model, extra_mastered)
        fingerprint = (
            round(sum(float(model.mastery_of(kp)) for kp in gaps), 6) if gaps else 0.0
        )
        return (
            self.learner_of(model).learner_id,
            tuple(sorted(gaps)),
            tuple(sorted(extra_mastered)),
            fingerprint,
            len(mastered),
        )

    def _run_search(
        self,
        model: Any,
        extra_mastered: Iterable[str] = (),
        max_courses: int = MAX_PATH_COURSES,
    ) -> _Search:
        """执行一次完整的 NSGA-II 搜索（带缓存）。"""
        extra = tuple(sorted(set(extra_mastered)))
        key = self._search_key(model, extra)
        cached = self._search_memo.get(key)
        if cached is not None:
            return cached

        learner = self.learner_of(model)
        ctx = learner_context(model)
        _closure, gaps, mastered = self.knowledge_gaps(model, extra)
        budget = ctx.total_budget
        weekly_budget = ctx.weekly_budget
        try:
            deadline_weeks = int(getattr(learner, "deadline_weeks", 0) or 0) or 12
        except (TypeError, ValueError):  # pragma: no cover - 脏数据兜底
            deadline_weeks = 12

        candidates = self.build_candidates(model, extra)
        pool = self._trim_pool(candidates, gaps)
        gain_cache: dict[str, float] = {}
        for course in candidates:
            gain_cache[course.course_id] = float(model.expected_gain(course))
        ref_gain = self.reference_gain(candidates, gaps, gain_cache, budget)
        goal = self.goal_of(model)
        target_ids = tuple(kp_id for kp_id in goal.target_kp_ids if kp_id in gaps)

        empty = _Search(
            pool=pool,
            gap_ids=tuple(sorted(gaps)),
            gap_mag=dict(gaps),
            mastered=mastered,
            ctx=ctx,
            front=[],
            history=[],
            gain_cache=gain_cache,
            fit_cache={},
            budget=budget,
            weekly_budget=weekly_budget,
            deadline_weeks=deadline_weeks,
            reference_gain=ref_gain,
            target_ids=target_ids,
        )
        if not pool:
            self._search_memo[key] = empty
            return empty

        gap_ids = tuple(sorted(gaps))
        gap_set = frozenset(gaps)
        target_set = frozenset(target_ids)
        fit_cache: dict[str, tuple[float, float]] = {}
        pool_kp_ids = [tuple(course.kp_ids) for course in pool]
        pool_prereq = [frozenset(self.course_prerequisites(c) & gap_set) for c in pool]
        pool_hours = [float(course.duration_hours) for course in pool]
        decode = lambda chromosome: self._decode_chromosome(  # noqa: E731
            chromosome,
            pool,
            pool_kp_ids,
            pool_prereq,
            pool_hours,
            target_set,
            gap_set,
            budget,
            max_courses,
        )

        # ---- 种子个体：性价比优先的确定性基线解 ----
        seed_courses = self.greedy_cover(
            model, candidates, gaps, budget, max_courses, target_ids
        )
        seed_ids = {c.course_id for c in seed_courses}
        dimension = len(pool) + 1
        seed = np.full(dimension, SEED_UNSELECTED, dtype=float)
        for index, course in enumerate(pool):
            if course.course_id in seed_ids:
                seed[index] = SEED_SELECTED
        seed[-1] = 1.0

        rng = np.random.default_rng(EVAL_SEED)
        population: list[np.ndarray] = [seed]
        while len(population) < NSGA_POPULATION:
            population.append(rng.random(dimension))

        def evaluate(chromosome: np.ndarray) -> dict[str, float]:
            courses = decode(chromosome)
            return self._evaluate_courses(
                courses, model, gain_cache, fit_cache, ctx, ref_gain
            )

        history: list[dict[str, float]] = []
        front = nsga2(
            population,
            evaluate,
            generations=NSGA_GENERATIONS,
            crossover_rate=NSGA_CROSSOVER_RATE,
            mutation_rate=NSGA_MUTATION_RATE,
            decode=decode,
            history=history,
        )

        # ---- 兜底：把启发式贪心解也纳入前沿（若被支配则自动淘汰） ----
        if seed_courses:
            greedy_objectives = self._evaluate_courses(
                seed_courses, model, gain_cache, fit_cache, ctx, ref_gain
            )
            front = list(front) + [
                ParetoSolution(
                    chromosome=seed,
                    objectives=greedy_objectives,
                    payload=list(seed_courses),
                )
            ]
            front = non_dominated_sort(front)[0]

        result = _Search(
            pool=pool,
            gap_ids=gap_ids,
            gap_mag=dict(gaps),
            mastered=mastered,
            ctx=ctx,
            front=front,
            history=history,
            gain_cache=gain_cache,
            fit_cache=fit_cache,
            budget=budget,
            weekly_budget=weekly_budget,
            deadline_weeks=deadline_weeks,
            reference_gain=ref_gain,
            target_ids=target_ids,
        )
        self._search_memo[key] = result
        while len(self._search_memo) > MAX_SEARCH_MEMO:
            self._search_memo.pop(next(iter(self._search_memo)))
        return result

    # --------------------------------------------------------
    # 第 7 步：排期 / 里程碑 / 可行性
    # --------------------------------------------------------
    def _build_schedule(
        self,
        ordered: Sequence[Course],
        gain_cache: Mapping[str, float],
        weekly_budget: float,
        deadline_weeks: int,
    ) -> tuple[list[PathStep], list[dict[str, Any]], list[dict[str, Any]], float]:
        """把排好序的课程摊到每周，产出步骤 / 周计划 / 里程碑 / 总时长。

        排期规则：设第 ``i`` 门课开始时已累计 ``t`` 小时，每周可投入 ``W`` 小时，
        则该课**起始周** ``week = ⌊t / W⌋ + 1``，**结束周**
        ``⌊(t + hours - ε) / W⌋ + 1``，课时按跨度**均摊**到这些周上。
        这样「一门 14 小时的课跨越 5 周」会被正确表达为 5 周 × 2.8 小时，
        而不是把 14 小时全部记在起始周（后者会让周负载均衡度失真）。
        """
        weekly = max(0.5, float(weekly_budget))
        steps: list[PathStep] = []
        week_load: dict[int, float] = {}
        week_courses: dict[int, list[str]] = {}
        cumulative = 0.0

        for index, course in enumerate(ordered, start=1):
            hours = float(course.duration_hours)
            start_week = int(cumulative // weekly) + 1
            end_week = int((cumulative + max(hours, _EPS) - _EPS) // weekly) + 1
            end_week = max(start_week, end_week)
            cumulative += hours

            span = end_week - start_week + 1
            per_week = hours / span
            for week in range(start_week, end_week + 1):
                week_load[week] = week_load.get(week, 0.0) + per_week
                week_courses.setdefault(week, []).append(course.course_id)

            steps.append(
                PathStep(
                    order=index,
                    course=course,
                    kp_ids=list(course.kp_ids),
                    hours=hours,
                    level=int(course.level),
                    cumulative_hours=cumulative,
                    week=max(1, min(start_week, max(1, deadline_weeks))),
                    expected_gain=float(gain_cache.get(course.course_id, 0.0)),
                )
            )

        weekly_plan = [
            {
                "week": week,
                "hours": round(week_load[week], 2),
                "course_ids": week_courses[week],
            }
            for week in sorted(week_load)
        ]

        milestones: list[dict[str, Any]] = []
        for step in steps:
            kp_ids: dict[str, None] = {}
            for kp_id in step.kp_ids:
                kp_ids.setdefault(kp_id, None)
            names = self.graph.names_of(list(kp_ids))[:4]
            if not names:
                continue
            milestones.append(
                {
                    "week": step.week,
                    "label": "掌握 " + "、".join(names),
                    "kp_names": names,
                }
            )
            if len(milestones) >= MAX_MILESTONES:
                break

        return steps, weekly_plan, milestones, cumulative

    @staticmethod
    def _feasibility(
        total_hours: float,
        total_budget: float,
        weekly_plan: Sequence[Mapping[str, Any]],
    ) -> float:
        """时间预算可行度。

        ``feasibility = min(1, 预算 / 总时长) × 周负载均衡度``

        周负载均衡度 ``= 1 - 平均相对偏差``，相对偏差为
        ``|本周时长 - 周均时长| / 周均时长``。越接近 1 表示各周负荷越均匀
        （避免「某一周要学 30 小时」的不可行排期）。
        """
        if total_hours <= 0:
            return 1.0
        coverage = 1.0 if total_budget <= 0 else min(1.0, total_budget / total_hours)
        loads = [float(item["hours"]) for item in weekly_plan]
        if not loads:
            return float(coverage)
        mean_load = sum(loads) / len(loads)
        if mean_load <= _EPS:
            return float(coverage)
        deviation = sum(abs(load - mean_load) for load in loads) / len(loads) / mean_load
        balance = max(0.0, min(1.0, 1.0 - deviation))
        return float(max(0.0, min(1.0, coverage * balance)))

    def _make_path(
        self,
        model: Any,
        courses: Sequence[Course],
        search: _Search,
        objectives: Mapping[str, float],
        preset: str,
    ) -> LearningPath:
        """把课程集合组装成 :class:`LearningPath`（并附覆盖率诊断）。"""
        ordered = self.order_courses(courses)
        steps, weekly_plan, milestones, total_hours = self._build_schedule(
            ordered,
            search.gain_cache,
            search.weekly_budget,
            search.deadline_weeks,
        )
        weeks = (
            int(math.ceil(total_hours / max(0.5, search.weekly_budget)))
            if total_hours > 0
            else 0
        )
        weeks = max(weeks, steps[-1].week if steps else 0, 0)
        total_gain = float(sum(step.expected_gain for step in steps))

        covered_kps: set[str] = set()
        for step in steps:
            covered_kps.update(step.kp_ids)
        goal = self.goal_of(model)
        all_targets = set(goal.target_kp_ids)
        gap_targets = all_targets & set(search.gap_ids)
        # 已经掌握的目标知识点不需要课程，但同样算「已达成」
        mastered_targets = {
            kp_id for kp_id in all_targets if float(model.mastery_of(kp_id)) >= MASTERY_GAP
        }
        covered_targets = gap_targets & covered_kps
        achieved = covered_targets | mastered_targets
        gaps = set(search.gap_ids)
        gap_covered = len(gaps & covered_kps)
        average_weekly_load = total_hours / weeks if weeks > 0 else 0.0
        budget = float(search.budget)
        coverage_ratio = len(achieved) / len(all_targets) if all_targets else 1.0

        return LearningPath(
            learner_id=self.learner_of(model).learner_id,
            goal=goal,
            steps=steps,
            total_hours=float(total_hours),
            weeks=weeks,
            total_gain=total_gain,
            objectives={key: float(value) for key, value in objectives.items()},
            feasibility=self._feasibility(total_hours, budget, weekly_plan),
            preset=preset,
            weekly_plan=weekly_plan,
            milestones=milestones,
            coverage_ratio=round(float(coverage_ratio), 4),
            covered_target_kp_ids=sorted(covered_targets),
            uncovered_target_kp_ids=sorted(all_targets - achieved),
            covered_target_kp_names=self.graph.names_of(sorted(covered_targets)),
            uncovered_target_kp_names=self.graph.names_of(sorted(all_targets - achieved)),
            budget_hours=round(budget, 2),
            over_budget=bool(total_hours > budget + 1e-6),
            coverage={
                "target_total": len(all_targets),
                "target_gap": len(gap_targets),
                "target_covered": len(covered_targets),
                "target_mastered": len(mastered_targets & all_targets),
                "target_achieved": len(achieved),
                "target_ratio": round(coverage_ratio, 3),
                "target_gap_ratio": round(len(covered_targets) / len(gap_targets), 3)
                if gap_targets
                else 1.0,
                "gap_total": len(gaps),
                "gap_covered": int(gap_covered),
                "gap_ratio": round(gap_covered / len(gaps), 3) if gaps else 1.0,
                "n_courses": len(steps),
                "required_hours": round(float(total_hours), 2),
                "available_hours": round(budget, 2),
                "budget_utilisation": round(float(total_hours) / budget, 3)
                if budget > 0
                else 0.0,
                "weeks_needed": int(weeks),
                "deadline_weeks": int(search.deadline_weeks),
                "weekly_budget": round(float(search.weekly_budget), 2),
                "avg_weekly_load": round(float(average_weekly_load), 2),
                "hours_shortfall": round(max(0.0, total_hours - budget), 2),
            },
        )

    def _empty_path(
        self,
        model: Any,
        preset: str,
        search: _Search | None = None,
    ) -> LearningPath:
        """空路径（缺口为空或候选集为空时返回），但仍给出完整覆盖率诊断。"""
        goal = self.goal_of(model)
        all_targets = set(goal.target_kp_ids)
        mastered_targets = {
            kp_id for kp_id in all_targets if float(model.mastery_of(kp_id)) >= MASTERY_GAP
        }
        gap_ids = set(search.gap_ids) if search is not None else set(model.knowledge_gap())
        gap_targets = all_targets & gap_ids
        available = (
            float(search.budget)
            if search is not None
            else float(getattr(model.learner, "total_available_hours", 0.0) or 0.0)
        )
        achieved = mastered_targets & all_targets
        coverage_ratio = len(achieved) / len(all_targets) if all_targets else 1.0
        return LearningPath(
            learner_id=self.learner_of(model).learner_id,
            goal=goal,
            steps=[],
            total_hours=0.0,
            weeks=0,
            total_gain=0.0,
            objectives={"gain": 0.0, "time": 1.0, "smoothness": 1.0, "fit": 0.0},
            feasibility=1.0,
            preset=preset,
            weekly_plan=[],
            milestones=[],
            coverage_ratio=round(float(coverage_ratio), 4),
            covered_target_kp_ids=[],
            uncovered_target_kp_ids=sorted(all_targets - achieved),
            covered_target_kp_names=[],
            uncovered_target_kp_names=self.graph.names_of(sorted(all_targets - achieved)),
            budget_hours=round(available, 2),
            over_budget=False,
            coverage={
                "target_total": len(all_targets),
                "target_gap": len(gap_targets),
                "target_covered": 0,
                "target_mastered": len(mastered_targets),
                "target_achieved": len(achieved),
                "target_ratio": round(coverage_ratio, 3),
                "target_gap_ratio": 0.0 if gap_targets else 1.0,
                "gap_total": len(gap_ids),
                "gap_covered": 0,
                "gap_ratio": 0.0 if gap_ids else 1.0,
                "n_courses": 0,
                "required_hours": 0.0,
                "available_hours": round(available, 2),
                "budget_utilisation": 0.0,
                "weeks_needed": 0,
                "deadline_weeks": int(search.deadline_weeks) if search is not None else 0,
                "weekly_budget": round(float(search.weekly_budget), 2)
                if search is not None
                else 0.0,
                "avg_weekly_load": 0.0,
                "hours_shortfall": 0.0,
            },
        )

    # --------------------------------------------------------
    # 对外主接口
    # --------------------------------------------------------
    def plan(
        self,
        model: Any,
        preset: str = "balanced",
        max_courses: int = MAX_PATH_COURSES,
        *,
        extra_mastered: Iterable[str] = (),
    ) -> LearningPath:
        """生成一条个性化学习路径（**总时长不超过学习者的时间预算**）。

        Args:
            model: 学习者模型（``LearnerModel``，只读）。
            preset: 偏好 ``balanced`` / ``fast`` / ``thorough``，决定 TOPSIS 权重。
            max_courses: 路径课程数上限（默认 ``MAX_PATH_COURSES = 40``）。
            extra_mastered: 额外视为已掌握的知识点（:meth:`replan` 走这里）。

        Returns:
            :class:`LearningPath`。``total_hours`` **恒 ≤** ``budget_hours``；
            预算不足以覆盖全部目标时，``coverage_ratio < 1`` 且
            ``uncovered_target_kp_ids`` 列出未覆盖的目标知识点。
        """
        weights = weights_of(preset, PLAN_PRESETS)
        search = self._run_search(model, extra_mastered, max_courses)
        if not search.front:
            return self._empty_path(model, preset, search)

        chosen = select_compromise(search.front, weights)
        courses = chosen.payload if chosen.payload is not None else []
        path = self._make_path(model, courses, search, chosen.objectives, preset)
        # 用真实 PathStep 再算一次目标值，确保与对外可见的 steps 完全一致
        path.objectives = evaluate_objectives(
            path.steps,
            model,
            cache=search.fit_cache,
            ctx=search.ctx,
            reference_gain=search.reference_gain,
        )
        return path

    def pareto_front(self, model: Any, preset: str = "balanced") -> list[LearningPath]:
        """返回帕累托前沿（供前端画「时长—收益」散点图）。

        Args:
            model: 学习者模型。
            preset: 用来决定「哪个解是当前妥协解」的偏好（默认 ``balanced``）。

        Returns:
            前沿解列表（按 ``total_hours`` 升序）。每个元素：

            * ``preset``：该解会被哪些预设选中（``""`` = 未被任何预设选中）；
            * ``is_selected``：**是否为 ``preset`` 参数对应的 TOPSIS 妥协解**，
              前端可直接据此画星标 / 高亮。

        Note:
            非支配排序在 4 个目标（gain / time / smoothness / fit）上进行，
            因此「在（时长—收益）二维平面上看起来被支配」的点是正常的——
            它在另外两个目标上仍然是非支配的。
        """
        search = self._run_search(model)
        if not search.front:
            return []
        paths: list[LearningPath] = []
        seen: set[tuple[str, ...]] = set()
        for solution in search.front:
            courses = solution.payload or []
            signature = tuple(sorted(c.course_id for c in courses))
            if signature in seen:
                continue
            seen.add(signature)
            paths.append(self._make_path(model, courses, search, solution.objectives, ""))
        if not paths:
            return []

        # 标注各 preset 的妥协解
        for key in PLAN_PRESETS:
            scores = topsis_scores(paths, PLAN_PRESETS[key])
            best = max(range(len(paths)), key=lambda i: (scores[i], -i))
            if not paths[best].preset:
                paths[best].preset = key

        # 标注「当前 preset 的 TOPSIS 妥协解」，供前端画星标
        requested = preset if preset in PLAN_PRESETS else "balanced"
        requested_scores = topsis_scores(paths, PLAN_PRESETS[requested])
        selected_index = max(range(len(paths)), key=lambda i: (requested_scores[i], -i))
        for index, candidate in enumerate(paths):
            candidate.is_selected = index == selected_index

        paths.sort(key=lambda p: (p.total_hours, -p.total_gain))
        return paths

    def replan(
        self,
        model: Any,
        done_course_ids: Iterable[str],
        preset: str = "balanced",
    ) -> LearningPath:
        """基于「已完成课程」重规划。

        语义：已完成课程覆盖的知识点视为已掌握，因此会从缺口集合中消失，
        对应课程不会再出现在新路径里；随之解锁的先修也会松动候选集条件。

        Args:
            model: 学习者模型（本方法**不修改** model 的掌握度，只读）。
            done_course_ids: 已完成的课程 id 集合。
            preset: 偏好。
        """
        extra: set[str] = set()
        for course_id in done_course_ids:
            course = self._course_by_id.get(course_id)
            if course is not None:
                extra.update(course.kp_ids)
        return self.plan(model, preset, MAX_PATH_COURSES, extra_mastered=frozenset(extra))

    def rebuild(
        self,
        model: Any,
        course_ids: Iterable[str],
        preset: str = "balanced",
    ) -> LearningPath:
        """按**给定的课程集合**重建一条 :class:`LearningPath`（排序 + 排期 + 评估）。

        用途：自适应引擎在「插入补强课 / 跳过冗余课 / 追加进阶课」之后，
        需要把改动过的课程集合重新变成一条结构完整、满足先修拓扑顺序的路径。
        与 :meth:`plan` 的区别是本方法**不做搜索**、不改动课程集合。

        Args:
            model: 学习者模型（只读）。
            course_ids: 课程 id 序列（重复项自动去重，未知 id 忽略）。
            preset: 该路径归属的偏好标签（仅用于标注）。
        """
        courses: list[Course] = []
        seen: set[str] = set()
        for course_id in course_ids:
            course = self._course_by_id.get(str(course_id))
            if course is None or course.course_id in seen:
                continue
            seen.add(course.course_id)
            courses.append(course)
        if not courses:
            return self._empty_path(model, preset)

        learner = self.learner_of(model)
        ctx = learner_context(model)
        deadline_weeks = int(getattr(learner, "deadline_weeks", 0) or 0) or 12
        gain_cache = {c.course_id: float(model.expected_gain(c)) for c in courses}
        gaps = dict(model.knowledge_gap())
        goal = self.goal_of(model)
        search = _Search(
            pool=courses,
            gap_ids=tuple(sorted(gaps)),
            gap_mag=gaps,
            mastered=frozenset(),
            ctx=ctx,
            front=[],
            history=[],
            gain_cache=gain_cache,
            fit_cache={},
            budget=ctx.total_budget,
            weekly_budget=ctx.weekly_budget,
            deadline_weeks=deadline_weeks,
            reference_gain=self.reference_gain(courses, gaps, gain_cache, ctx.total_budget),
            target_ids=tuple(kp for kp in goal.target_kp_ids if kp in gaps),
        )
        objectives = self._evaluate_courses(
            courses, model, gain_cache, search.fit_cache, ctx, search.reference_gain
        )
        return self._make_path(model, courses, search, objectives, preset)

    def search_diagnostics(self, model: Any) -> dict[str, Any]:
        """返回本次搜索的可视化 / 报告用诊断信息（不影响主流程）。"""
        search = self._run_search(model)
        return {
            "n_candidates": len(search.pool),
            "n_gaps": len(search.gap_ids),
            "n_targets": len(search.target_ids),
            "n_front": len(search.front),
            "budget_hours": round(search.budget, 2),
            "weekly_budget": round(search.weekly_budget, 2),
            "reference_gain": round(search.reference_gain, 4),
            "history": search.history,
            "front": [
                {
                    "hours": round(float(sum(c.duration_hours for c in (s.payload or []))), 2),
                    "objectives": {k: round(float(v), 4) for k, v in s.objectives.items()},
                    "n_courses": len(s.payload or []),
                }
                for s in search.front
            ],
        }


__all__ = ["PathPlanner", "LearningPath", "PathStep"]
