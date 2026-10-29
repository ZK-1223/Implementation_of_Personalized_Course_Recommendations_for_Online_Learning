"""学习者建模：多维学习者画像 + 基于 BKT 的掌握度状态推断。

对应课程设计报告「实现要求 ①：学习者建模（知识水平、学习偏好、时间安排等多维度学习者模型）」。

--------------------------------------------------------------------------------
一、画像的五个维度
--------------------------------------------------------------------------------
============  ====================================================  ===========================
维度          含义                                                  在代码中的载体
============  ====================================================  ===========================
知识水平      ``kp_id -> 掌握度`` 的 420 维向量（BKT 推断）            :attr:`LearnerModel._mastery`
知识缺口      目标闭包内掌握度未达标的「待提升幅度」                  :meth:`LearnerModel.knowledge_gap`
学习偏好      ``theory_preference`` + ``prefer_modality``            :meth:`LearnerModel.style_fit`
时间安排      每周预算 + 单次专注时长 + 每周可学天数                  :meth:`LearnerModel.time_fit`
能力/兴趣     ``ability`` / ``interests`` / ``motivation``           :attr:`LearnerModel.learner`
============  ====================================================  ===========================

另有衍生信号：**先修就绪度** :meth:`LearnerModel.readiness_for`（「这门课我现在能不能听懂」）
与**预期收益** :meth:`LearnerModel.expected_gain`（「学完能涨多少掌握度」），
二者是内容推荐与路径规划的输入。

--------------------------------------------------------------------------------
二、掌握度初始化：三级先验（不是「全 0」）
--------------------------------------------------------------------------------
直接原因：如果所有知识点都从 0 开始，那么「先修就绪度」永远为 0，冷启动学习者的推荐会全部
退化到「最基础的课」。本系统按如下三级递推给出先验（知识点按拓扑序处理，保证先修先行）::

    base       = p_init * (0.55 + 0.90 * ability)          # 能力越高，先验越高
    domain基    = base * (1.22 if 领域 ∈ interests else 1.0)  # 兴趣领域有基础
    有先修时    = domain基 + (1 - domain基) * 0.55 * lift
                  lift = max(0, mean(先修掌握度) - 0.25) ** 1.2
    难度折扣    = value * (1 - 0.09 * (difficulty - 1))       # 越难越不可能无师自通

其中 ``lift`` 表达「先修扎实 ⇒ 后继知识点很可能已经半懂」这一领域常识：先修均值 0.9 时
后继先验约 0.33（难度 2）/ 0.27（难度 4），先修均值 0.5 时约 0.15，先修均值 0.3 时接近基值。
``learner.initial_mastery`` 里显式给出的知识点**直接采用原值**，不参与递推。

--------------------------------------------------------------------------------
三、历史行为回放（让画像真实化的关键一步）
--------------------------------------------------------------------------------
构造时按 ``timestamp`` 升序回放该学习者的全部交互记录：

1. 相邻两条记录之间若存在空闲间隔，先对**全部**知识点做遗忘衰减
   （``MASTERY_DECAY_PER_WEEK ** 空闲周数``，指数衰减见 :mod:`learnpath.learner.bkt`）；
2. 把该条记录折算成一个「软观测分数」``score``（结课测验优先，其次完成度 / 评分）；
3. 对课程覆盖的每个知识点执行一次 BKT 更新，覆盖权重取 ``KPCoverage.weight``，
   有效观测分数按努力程度微调 ``score * (0.80 + 0.20 * min(effort_ratio, 1))``。

这样「学过 3 门数据分析课的学习者」在建模后自然拥有比冷启动学习者更高的目标领域掌握度，
推荐结果因此具备真实感。

--------------------------------------------------------------------------------
四、性能设计
--------------------------------------------------------------------------------
掌握度用 ``numpy`` 的一维数组保存（下标 = ``list(dataset.knowledge_points)`` 的顺序），
遗忘衰减、领域均值、先修均值都是向量化操作；知识点拓扑序、先修下标、课程先修集合等
图结构属性在模块级缓存中按 ``dataset`` 身份复用一次。实测构造一个 ``LearnerModel``
约 1~2 ms（420 知识点 + 约 20 条历史），因此 720 个学习者全量建模在秒级完成。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import numpy as np

from learnpath.config import (
    BKT,
    MASTERY_GAP,
    MASTERY_MASTERED,
    MODALITIES,
)
from learnpath.data.schema import Course, Dataset, Interaction, Learner
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.bkt import MasteryTracker

__all__ = [
    "LearnerModel",
    "ProfileSnapshot",
    "interactions_by_learner",
    "cached_course_prerequisites",
    "sigmoid",
]

# ---------------------------------------------------------------------------
# 模型常数（报告中引用这些数值做参数说明）
# ---------------------------------------------------------------------------

#: 能力对先验掌握度的调制：``p_init * (PRIOR_ABILITY_BASE + PRIOR_ABILITY_GAIN * ability)``。
#: 与 BKT 的学习率调制保持一致（ability = 0.5 → 乘子 1.0）。
PRIOR_ABILITY_BASE: Final[float] = 0.55
PRIOR_ABILITY_GAIN: Final[float] = 0.90

#: 兴趣领域（``interests``）带来的先验加成系数。
PRIOR_INTEREST_BOOST: Final[float] = 1.22

#: 先修就绪度转化为先验的强度：``lift = max(0, r - 0.25) ** 1.2``，再乘该系数。
PRIOR_SPREAD: Final[float] = 0.55
PRIOR_LIFT_POWER: Final[float] = 1.20
PRIOR_LIFT_FLOOR: Final[float] = 0.25

#: 难度折扣：``value *= 1 - DIFFICULTY_DISCOUNT * (difficulty - 1)``。
DIFFICULTY_DISCOUNT: Final[float] = 0.09

#: 先验截断区间，避免出现 0 或 1 这种「不可修正」的极端先验。
PRIOR_MIN: Final[float] = 0.01
PRIOR_MAX: Final[float] = 0.95

#: 先修就绪度的 sigmoid 参数：``readiness = sigmoid(6.0 * (平均先修掌握度 - 0.55))``。
#: 含义：先修均值 0.55 时刚好「半就绪」，0.80 → 0.82，0.30 → 0.18，斜率保证区分度。
READINESS_SLOPE: Final[float] = 6.0
READINESS_MID: Final[float] = 0.55

#: 风格匹配的两项权重（理论/实践接近度 0.6 + 模态一致度 0.4）。
STYLE_THEORY_WEIGHT: Final[float] = 0.60
STYLE_MODALITY_WEIGHT: Final[float] = 0.40

#: 模态一致度：完全相同 1.0；一方为 mixed 0.65；完全不同 0.30。
MODALITY_MATCH_EXACT: Final[float] = 1.0
MODALITY_MATCH_SOFT: Final[float] = 0.65
MODALITY_MATCH_MISMATCH: Final[float] = 0.30

#: 时间可承受度的三项权重：每周预算 0.45 + 单次专注 0.30 + 完成跨度 0.25。
TIME_BUDGET_WEIGHT: Final[float] = 0.45
TIME_SESSION_WEIGHT: Final[float] = 0.30
TIME_HORIZON_WEIGHT: Final[float] = 0.25

#: 单门课「舒适完成跨度」上限（周），超过则时间匹配度衰减。
TIME_COMFORT_WEEKS: Final[float] = 4.0

#: ``expected_gain`` 使用的名义达标表现（模拟「学完并达到目标」时的一次观测）。
NOMINAL_TARGET_SCORE: Final[float] = 0.85

#: 一条交互记录折算成软观测分数的规则参数（权重和为 1）。
OBS_QUIZ_WEIGHT: Final[float] = 0.80
OBS_PROGRESS_WEIGHT: Final[float] = 0.20
OBS_COMPLETED_BASE: Final[float] = 0.65
OBS_UNFINISHED_FACTOR: Final[float] = 0.50
OBS_RATING_WEIGHT: Final[float] = 0.05


# ---------------------------------------------------------------------------
# 模块级缓存：图结构与交互索引按 dataset 身份复用（不修改 Dataset / KnowledgeGraph）
# ---------------------------------------------------------------------------

_INTERACTIONS_CACHE: dict[int, tuple[Dataset, dict[str, tuple[Interaction, ...]]]] = {}
_PREREQ_CACHE: dict[int, tuple[Dataset, dict[str, tuple[str, ...]]]] = {}
_LAYOUT_CACHE: dict[int, tuple[Dataset, tuple[str, ...], np.ndarray, tuple[np.ndarray, ...]]] = {}


def interactions_by_learner(dataset: Dataset) -> dict[str, tuple[Interaction, ...]]:
    """按学习者分组交互记录（模块级缓存，避免反复全表扫描）。

    ``Dataset.interactions_of`` 每次调用都是 O(N) 扫描，在 14000 条记录 × 720 个学习者的
    场景下会成为明显瓶颈，因此这里提供一次性构建的分组索引。

    Args:
        dataset: 数据集。

    Returns:
        ``learner_id -> (Interaction, ...)``，元素保持原列表顺序。
    """
    key = id(dataset)
    cached = _INTERACTIONS_CACHE.get(key)
    if cached is not None and cached[0] is dataset:
        return cached[1]

    grouped: dict[str, list[Interaction]] = {}
    for inter in dataset.interactions:
        grouped.setdefault(inter.learner_id, []).append(inter)
    result = {lid: tuple(items) for lid, items in grouped.items()}
    _INTERACTIONS_CACHE[key] = (dataset, result)
    return result


def cached_course_prerequisites(dataset: Dataset, graph: KnowledgeGraph) -> dict[str, tuple[str, ...]]:
    """课程的先修知识点集合（模块级缓存）。

    等价于对每门课调用 :meth:`KnowledgeGraph.course_prerequisites`，但只计算一次。
    先修集合**不含**课程自身覆盖的知识点（见 graph 契约）。

    Args:
        dataset: 数据集。
        graph: 知识图谱。

    Returns:
        ``course_id -> (先修 kp_id, ...)``，按 kp_id 排序以保证可复现。
    """
    key = id(dataset)
    cached = _PREREQ_CACHE.get(key)
    if cached is not None and cached[0] is dataset:
        return cached[1]

    result: dict[str, tuple[str, ...]] = {}
    for course_id, course in dataset.courses.items():
        result[course_id] = tuple(sorted(graph.course_prerequisites(course)))
    _PREREQ_CACHE[key] = (dataset, result)
    return result


def _graph_layout(
    dataset: Dataset, graph: KnowledgeGraph
) -> tuple[tuple[str, ...], np.ndarray, tuple[np.ndarray, ...]]:
    """返回知识点顺序、拓扑序下标与每个知识点的先修下标（模块级缓存）。"""
    key = id(dataset)
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None and cached[0] is dataset:
        return cached[1], cached[2], cached[3]

    kp_ids: tuple[str, ...] = tuple(dataset.knowledge_points)
    index = {kp_id: i for i, kp_id in enumerate(kp_ids)}
    topo = np.array([index[k] for k in graph.topological_order()], dtype=np.intp)
    preds = tuple(
        np.array([index[p] for p in graph.predecessors(kp_id)], dtype=np.intp)
        for kp_id in kp_ids
    )
    _LAYOUT_CACHE[key] = (dataset, kp_ids, topo, preds)
    return kp_ids, topo, preds


def sigmoid(x: float) -> float:
    """数值稳定的 logistic 函数，把实数映射到 ``(0, 1)``。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(value: float, low: float, high: float) -> float:
    """截断到 ``[low, high]``。"""
    if value < low:
        return low
    if value > high:
        return high
    return value


# ===========================================================================
# 画像快照
# ===========================================================================


@dataclass(slots=True)
class ProfileSnapshot:
    """学习者画像快照（可 JSON 序列化，对应 ``docs/API.md`` 第 2 节 ``profile``）。

    Attributes:
        learner_id: 学习者 id。
        mastery: 全部知识点的掌握度 ``kp_id -> 0~1``。
        domain_mastery: 各领域平均掌握度 ``domain -> 0~1``。
        target_kps: 目标知识点 id 列表（来自 ``Goal.target_kp_ids``）。
        gaps: 知识缺口 ``kp_id -> 待提升幅度``（目标闭包内、掌握度低于阈值者）。
        readiness: 目标就绪度 0~1（对「学会目标知识点」的整体前置条件满足程度）。
        style_vector: 学习风格向量（theory / practice + 四种模态偏好分布，和为 1）。
        time_budget: 每周可投入小时数。
        total_available_hours: 到截止日期为止的总可投入小时数。
        mastered_count: 已掌握（>= ``MASTERY_MASTERED``）知识点个数。
        gap_count: 知识缺口个数。
        completion_rate: 历史课程的完成率。
        avg_mastery: 全知识点平均掌握度。
    """

    learner_id: str
    mastery: dict[str, float]
    domain_mastery: dict[str, float]
    target_kps: list[str]
    gaps: dict[str, float]
    readiness: float
    style_vector: dict[str, float]
    time_budget: float
    total_available_hours: float = 0.0
    mastered_count: int = 0
    gap_count: int = 0
    completion_rate: float = 0.0
    avg_mastery: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转为 API 可直接返回的字典（浮点保留 3 位小数）。"""
        return {
            "learner_id": self.learner_id,
            "mastery": {k: round(float(v), 3) for k, v in self.mastery.items()},
            "domain_mastery": {k: round(float(v), 3) for k, v in self.domain_mastery.items()},
            "target_kps": list(self.target_kps),
            "gaps": {k: round(float(v), 3) for k, v in self.gaps.items()},
            "readiness": round(float(self.readiness), 3),
            "style_vector": {k: round(float(v), 3) for k, v in self.style_vector.items()},
            "time_budget": round(float(self.time_budget), 2),
            "total_available_hours": round(float(self.total_available_hours), 2),
            "mastered_count": int(self.mastered_count),
            "gap_count": int(self.gap_count),
            "completion_rate": round(float(self.completion_rate), 3),
            "avg_mastery": round(float(self.avg_mastery), 3),
        }


# ===========================================================================
# 学习者模型
# ===========================================================================


class LearnerModel:
    """单个学习者的多维模型：知识水平 + 缺口 + 先修就绪 + 风格 / 时间 / 收益估计。

    构造时完成两件事：按「能力 + 先修就绪度」递推初始先验掌握度，随后按时间顺序回放该
    学习者的历史交互做 BKT 更新。因此模型是**有状态**的：:meth:`apply_outcome` 会原地改变
    掌握度（自适应反馈需要这一语义）；其余方法均为只读。

    Args:
        learner: 学习者实体。
        dataset: 数据集（用于取课程、目标、交互记录）。
        graph: 知识图谱（用于先修闭包与课程先修集合）。

    Raises:
        KeyError: ``learner.goal_id`` 不在数据集的目标集合中。

    Example:
        >>> from learnpath.data.fixtures import build_tiny_dataset
        >>> from learnpath.knowledge.graph import KnowledgeGraph
        >>> ds = build_tiny_dataset()
        >>> graph = KnowledgeGraph.from_dataset(ds)
        >>> model = LearnerModel(ds.learners["L-0001"], ds, graph)
        >>> 0.0 <= model.snapshot().readiness <= 1.0
        True
    """

    __slots__ = (
        "learner",
        "dataset",
        "graph",
        "tracker",
        "_goal",
        "_kp_ids",
        "_index",
        "_mastery",
        "_topo",
        "_preds",
        "_target_kps",
        "_target_closure",
        "_required_mastery",
        "_history",
        "_covered",
        "_version",
        "_readiness_cache",
        "_style_cache",
        "_time_cache",
        "_gap_cache",
        "_domain_cache",
        "_domain_owner",
        "_domain_names",
    )

    def __init__(self, learner: Learner, dataset: Dataset, graph: KnowledgeGraph) -> None:
        self.learner: Learner = learner
        self.dataset: Dataset = dataset
        self.graph: KnowledgeGraph = graph
        self.tracker: MasteryTracker = MasteryTracker(BKT)

        if learner.goal_id not in dataset.goals:
            raise KeyError(f"{learner.learner_id}: 目标 {learner.goal_id!r} 不在数据集中")
        self._goal = dataset.goals[learner.goal_id]

        self._kp_ids, self._topo, self._preds = _graph_layout(dataset, graph)
        self._index: dict[str, int] = {kp_id: i for i, kp_id in enumerate(self._kp_ids)}
        self._mastery: np.ndarray = np.zeros(len(self._kp_ids), dtype=np.float64)

        self._target_kps: tuple[str, ...] = tuple(self._goal.target_kp_ids)
        self._target_closure: frozenset[str] = graph.prerequisite_closure(self._target_kps)
        self._required_mastery: float = float(self._goal.required_mastery)

        # 领域维度（用于 domain_mastery 的向量化分组均值）
        self._domain_names: tuple[str, ...] = tuple(sorted({kp.domain for kp in dataset.knowledge_points.values()}))
        domain_index = {name: i for i, name in enumerate(self._domain_names)}
        self._domain_owner: np.ndarray = np.array(
            [domain_index[dataset.knowledge_points[kp].domain] for kp in self._kp_ids],
            dtype=np.intp,
        )

        self._history: tuple[Interaction, ...] = tuple(
            sorted(
                interactions_by_learner(dataset).get(learner.learner_id, ()),
                key=lambda i: i.timestamp,
            )
        )
        self._covered: frozenset[str] = frozenset(i.course_id for i in self._history)

        self._version: int = 0
        self._readiness_cache: dict[str, tuple[int, float]] = {}
        self._style_cache: dict[str, float] = {}
        self._time_cache: dict[str, float] = {}
        self._gap_cache: tuple[int, dict[str, float]] | None = None
        self._domain_cache: tuple[int, dict[str, float]] | None = None

        self._initialise_priors()
        self._replay_history()

    # ==================================================================
    # 基础属性
    # ==================================================================
    @property
    def goal(self):
        """该学习者的职业 / 学习目标（:class:`learnpath.data.schema.Goal`）。"""
        return self._goal

    @property
    def target_kps(self) -> tuple[str, ...]:
        """目标知识点 id 元组。"""
        return self._target_kps

    @property
    def target_closure(self) -> frozenset[str]:
        """目标知识点的先修闭包（含目标本身）。"""
        return self._target_closure

    @property
    def required_mastery(self) -> float:
        """达成目标所需的最低掌握度。"""
        return self._required_mastery

    @property
    def version(self) -> int:
        """状态版本号，每次掌握度变化 +1（供推荐器做缓存失效）。"""
        return self._version

    @property
    def kp_ids(self) -> tuple[str, ...]:
        """掌握度向量的知识点顺序。"""
        return self._kp_ids

    def __len__(self) -> int:
        return len(self._kp_ids)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"LearnerModel({self.learner.learner_id}, ability={self.learner.ability:.2f}, "
            f"kps={len(self._kp_ids)}, history={len(self._history)})"
        )

    # ==================================================================
    # 掌握度读写
    # ==================================================================
    def mastery_of(self, kp_id: str) -> float:
        """返回单个知识点的掌握度；未知知识点返回 0.0（防御性设计）。"""
        index = self._index.get(kp_id)
        if index is None:
            return 0.0
        return float(self._mastery[index])

    def mastery_vector(self) -> dict[str, float]:
        """全部知识点的掌握度 ``kp_id -> 0~1``（新字典，调用方修改不影响内部状态）。"""
        return {kp_id: float(self._mastery[i]) for i, kp_id in enumerate(self._kp_ids)}

    def mastery_array(self, copy: bool = False) -> np.ndarray:
        """掌握度的 numpy 视图（顺序见 :attr:`kp_ids`）。

        Args:
            copy: True 时返回副本；默认 False 返回内部只读视图（供推荐器向量化计算，
                调用方**不得**原地修改）。
        """
        return self._mastery.copy() if copy else self._mastery

    def domain_mastery(self) -> dict[str, float]:
        """各领域平均掌握度 ``domain -> 0~1``（仅统计该领域内知识点）。"""
        if self._domain_cache is not None and self._domain_cache[0] == self._version:
            return dict(self._domain_cache[1])
        sums = np.bincount(self._domain_owner, weights=self._mastery, minlength=len(self._domain_names))
        counts = np.bincount(self._domain_owner, minlength=len(self._domain_names))
        result = {
            name: float(sums[i] / counts[i]) if counts[i] else 0.0
            for i, name in enumerate(self._domain_names)
        }
        self._domain_cache = (self._version, result)
        return dict(result)

    def knowledge_gap(self) -> dict[str, float]:
        """知识缺口 ``kp_id -> 待提升幅度``。

        统计范围为目标闭包（目标知识点 + 其全部先修），只保留掌握度低于
        :data:`learnpath.config.MASTERY_GAP`（0.60）的知识点，缺口值 =
        ``required_mastery - 当前掌握度``。按缺口从大到小排序，便于前端直接展示。
        """
        if self._gap_cache is not None and self._gap_cache[0] == self._version:
            return dict(self._gap_cache[1])

        required = self._required_mastery
        items: list[tuple[str, float]] = []
        for kp_id in sorted(self._target_closure):
            value = float(self._mastery[self._index[kp_id]])
            if value >= MASTERY_GAP:
                continue
            gap = required - value
            if gap > 1e-6:
                items.append((kp_id, gap))
        items.sort(key=lambda item: (-item[1], item[0]))
        result = {kp_id: gap for kp_id, gap in items}
        self._gap_cache = (self._version, result)
        return dict(result)

    def course_gaps(self, course: Course) -> list[tuple[str, float, float]]:
        """课程覆盖知识点中位于**目标闭包**内的缺口明细。

        Returns:
            ``[(kp_id, 当前掌握度, 待提升幅度), ...]``，按待提升幅度降序；
            空列表表示该课程对本学习者的目标没有缺口贡献（或只覆盖目标外知识）。
        """
        required = self._required_mastery
        result: list[tuple[str, float, float]] = []
        for cov in course.kp_coverage:
            if cov.kp_id not in self._target_closure:
                continue
            value = float(self._mastery[self._index[cov.kp_id]])
            gap = required - value
            if gap > 1e-6:
                result.append((cov.kp_id, value, gap))
        result.sort(key=lambda item: -item[2])
        return result

    def courses_covered(self) -> frozenset[str]:
        """该学习者已经产生过交互记录的课程 id 集合。"""
        return self._covered

    def history_size(self) -> int:
        """历史交互条数。"""
        return len(self._history)

    # ==================================================================
    # 先修就绪度 / 风格 / 时间
    # ==================================================================
    def readiness_for(self, course: Course) -> float:
        """先修就绪度 0~1：该学习者是否具备学习本课程的先修基础。

        计算::

            r = mean(先修知识点掌握度)                  # 无先修 → 直接返回 1.0
            readiness = sigmoid(6.0 * (r - 0.55))

        sigmoid 的作用是把「平均掌握度」这一相关量映射为概率式的就绪度，并使中间区段
        （0.3~0.8）具备足够区分度；先修均值 0.55 恰对应 0.5「半就绪」。
        """
        cached = self._readiness_cache.get(course.course_id)
        if cached is not None and cached[0] == self._version:
            return cached[1]

        prereqs = cached_course_prerequisites(self.dataset, self.graph).get(course.course_id, ())
        if not prereqs:
            value = 1.0
        else:
            idx = [self._index[k] for k in prereqs if k in self._index]
            if not idx:
                value = 1.0
            else:
                mean_mastery = float(self._mastery[idx].mean())
                value = sigmoid(READINESS_SLOPE * (mean_mastery - READINESS_MID))
        value = _clip(value, 0.0, 1.0)
        self._readiness_cache[course.course_id] = (self._version, value)
        return value

    def readiness_detail(self, course: Course) -> tuple[int, int, float]:
        """先修就绪明细 ``(已达标的先修数, 先修总数, 平均掌握度)``。

        「达标」指掌握度 >= :data:`learnpath.config.MASTERY_MASTERED`（0.85），
        用于推荐理由中的「先修已掌握 3/4」这类具体描述。
        """
        prereqs = cached_course_prerequisites(self.dataset, self.graph).get(course.course_id, ())
        if not prereqs:
            return 0, 0, 1.0
        idx = [self._index[k] for k in prereqs if k in self._index]
        if not idx:
            return 0, 0, 1.0
        values = self._mastery[idx]
        return int((values >= MASTERY_MASTERED).sum()), len(idx), float(values.mean())

    def style_fit(self, course: Course) -> float:
        """学习风格匹配度 0~1。

        两个子项加权::

            theory_sim      = 1 - |theory_preference - course.theory_ratio|
            modality_match  = 1.0（完全一致）/ 0.65（一方为 mixed）/ 0.30（其他）
            style_fit       = 0.6 * theory_sim + 0.4 * modality_match
        """
        cached = self._style_cache.get(course.course_id)
        if cached is not None:
            return cached

        style = self.learner.learning_style
        theory_sim = 1.0 - abs(float(style.theory_preference) - float(course.theory_ratio))
        if style.prefer_modality == course.modality:
            modality_match = MODALITY_MATCH_EXACT
        elif "mixed" in (style.prefer_modality, course.modality):
            modality_match = MODALITY_MATCH_SOFT
        else:
            modality_match = MODALITY_MATCH_MISMATCH
        value = STYLE_THEORY_WEIGHT * theory_sim + STYLE_MODALITY_WEIGHT * modality_match
        value = _clip(value, 0.0, 1.0)
        self._style_cache[course.course_id] = value
        return value

    def time_fit(self, course: Course) -> float:
        """时间可承受度 0~1。

        三个子项加权::

            budget_fit  = 1 / (1 + 0.45 * max(0, 课程时长/每周预算 - 1))     # 权重 0.45
            session_fit = 1 / (1 + 0.08 * max(0, 需占用专注次数 - 4))         # 权重 0.30
            horizon_fit = 1 / (1 + 0.50 * max(0, 完成跨度周数/4 - 1))         # 权重 0.25

        其中「需占用专注次数」= ``时长/(单次专注分钟/60)``，「完成跨度周数」=
        ``专注次数 / 每周可学天数``。三者共同刻画「每周预算够不够 / 单次专注能否承载 /
        拖得久不久」。
        """
        cached = self._time_cache.get(course.course_id)
        if cached is not None:
            return cached

        style = self.learner.learning_style
        weekly_budget = max(float(self.learner.time_budget_hours_per_week), 0.5)
        hours = float(course.duration_hours)

        ratio = hours / weekly_budget
        budget_fit = 1.0 / (1.0 + 0.45 * max(0.0, ratio - 1.0))

        session_hours = max(int(style.session_len_min), 10) / 60.0
        sittings = hours / session_hours
        session_fit = 1.0 / (1.0 + 0.08 * max(0.0, sittings - 4.0))

        weekly_days = max(int(style.weekly_days), 1)
        weeks_needed = sittings / weekly_days
        horizon_fit = 1.0 / (1.0 + 0.50 * max(0.0, weeks_needed / TIME_COMFORT_WEEKS - 1.0))

        value = (
            TIME_BUDGET_WEIGHT * budget_fit
            + TIME_SESSION_WEIGHT * session_fit
            + TIME_HORIZON_WEIGHT * horizon_fit
        )
        value = _clip(value, 0.0, 1.0)
        self._time_cache[course.course_id] = value
        return value

    # ==================================================================
    # 收益估计与状态更新
    # ==================================================================
    def expected_gain(self, course: Course) -> float:
        """预期掌握度提升总量（无副作用，可反复调用）。

        对课程覆盖的每个知识点做一次 BKT 单步模拟（名义达标表现 0.85），取
        ``post - prior`` 的正部，再乘「先修就绪折扣」后求和::

            discount = 0.5 + 0.5 * readiness_for(course)
            gain     = Σ_k max(0, BKT_update(m_k, 0.85, w_k, ability) - m_k) * discount

        量纲说明：这是**对覆盖知识点求和**的绝对增量，典型值 0.05~1.0，随「已掌握程度」
        单调递减（边际收益递减），可直接作为路径规划里单门课程的收益贡献。
        """
        readiness = self.readiness_for(course)
        discount = 0.5 + 0.5 * readiness
        ability = float(self.learner.ability)
        total = 0.0
        for cov in course.kp_coverage:
            index = self._index.get(cov.kp_id)
            if index is None:
                continue
            prior = float(self._mastery[index])
            post = self.tracker.update(prior, NOMINAL_TARGET_SCORE, float(cov.weight), ability)
            if post > prior:
                total += (post - prior) * discount
        return float(total)

    def apply_outcome(
        self, course: Course, score: float, effort_ratio: float
    ) -> dict[str, float]:
        """把一次真实学习结果写入模型，**原地**更新掌握度。

        对课程覆盖的每个知识点执行 BKT 更新（与历史回放使用同一状态方程），
        并按契约返回每个知识点的掌握度**增量**。

        Args:
            course: 已学习的课程。
            score: 归一化学习表现 ∈ [0, 1]（结课测验分 / 达标程度）。
            effort_ratio: 实际投入 / 课程时长，>= 0；大于 1 视为超额投入（不再额外增益）。

        Returns:
            ``{kp_id: 掌握度增量}``，键为课程覆盖的知识点；增量为 ``after - before``。
            调用方若需要「新掌握度」，用 :meth:`mastery_of` 逐点读取（或 :meth:`mastery_vector`）。
        """
        deltas = self._apply_course(course, score, effort_ratio)
        self._invalidate()
        return deltas

    def apply_decay(self, idle_weeks: float) -> None:
        """对全部知识点执行遗忘衰减（原地），用于模拟「隔了 N 周才继续学」。

        Args:
            idle_weeks: 空闲周数，<= 0 时不做任何修改。
        """
        factor = self.tracker.decay_factor(idle_weeks)
        if factor >= 1.0:
            return
        self._mastery *= factor
        self._invalidate()

    # ==================================================================
    # 快照
    # ==================================================================
    def snapshot(self) -> ProfileSnapshot:
        """生成画像快照（字段严格对应 ``learner/__init__.py`` 契约）。"""
        mastery = self.mastery_vector()
        gaps = self.knowledge_gap()
        domain_mastery = self.domain_mastery()
        target_readiness = self._goal_readiness()

        total = len(mastery)
        mastered = sum(1 for v in mastery.values() if v >= MASTERY_MASTERED)
        history = self._history
        completion_rate = (
            sum(1 for i in history if i.completed) / len(history) if history else 0.0
        )

        return ProfileSnapshot(
            learner_id=self.learner.learner_id,
            mastery=mastery,
            domain_mastery=domain_mastery,
            target_kps=list(self._target_kps),
            gaps=gaps,
            readiness=round(_clip(target_readiness, 0.0, 1.0), 4),
            style_vector=self.style_vector(),
            time_budget=float(self.learner.time_budget_hours_per_week),
            total_available_hours=float(self.learner.total_available_hours),
            mastered_count=int(mastered),
            gap_count=len(gaps),
            completion_rate=round(float(completion_rate), 4),
            avg_mastery=round(float(sum(mastery.values()) / total) if total else 0.0, 4),
        )

    def style_vector(self) -> dict[str, float]:
        """学习风格向量：``theory`` / ``practice`` + 四种模态偏好分布（和为 1）。"""
        style = self.learner.learning_style
        theory = _clip(float(style.theory_preference), 0.0, 1.0)
        raw = {}
        for modality in MODALITIES:
            if modality == style.prefer_modality:
                raw[modality] = 1.0
            elif modality == "mixed":
                raw[modality] = 0.30
            else:
                raw[modality] = 0.10
        total = sum(raw.values()) or 1.0
        vector: dict[str, float] = {"theory": theory, "practice": 1.0 - theory}
        vector.update({k: v / total for k, v in raw.items()})
        return vector

    # ==================================================================
    # 内部：初始化与回放
    # ==================================================================
    def _initialise_priors(self) -> None:
        """三级先验递推（详见模块 docstring 第二节）。"""
        ability = _clip(float(self.learner.ability), 0.0, 1.0)
        base = float(BKT.p_init) * (PRIOR_ABILITY_BASE + PRIOR_ABILITY_GAIN * ability)
        interests = set(self.learner.interests)
        initial = self.learner.initial_mastery
        kps = self.dataset.knowledge_points

        for index in self._topo:
            index = int(index)
            kp_id = self._kp_ids[index]
            kp = kps[kp_id]
            explicit = initial.get(kp_id)
            if explicit is not None:
                # 显式声明值直接采用（只做概率截断），不参与先验递推
                self._mastery[index] = _clip(float(explicit), 0.0, 0.999)
                continue

            base_domain = base * (PRIOR_INTEREST_BOOST if kp.domain in interests else 1.0)
            preds = self._preds[index]
            if preds.size:
                readiness = float(self._mastery[preds].mean())
                lift = max(0.0, readiness - PRIOR_LIFT_FLOOR) ** PRIOR_LIFT_POWER
                value = base_domain + (1.0 - base_domain) * PRIOR_SPREAD * lift
            else:
                value = base_domain
            value *= 1.0 - DIFFICULTY_DISCOUNT * (int(kp.difficulty) - 1)
            self._mastery[index] = _clip(value, PRIOR_MIN, PRIOR_MAX)

    def _replay_history(self) -> None:
        """按时间顺序回放历史交互：遗忘衰减 + BKT 更新。"""
        last_ts: datetime | None = None
        for inter in self._history:
            timestamp = _parse_ts(inter.timestamp)
            if timestamp is not None and last_ts is not None:
                idle_weeks = (timestamp - last_ts).total_seconds() / (7.0 * 86400.0)
                if idle_weeks > 0:
                    factor = self.tracker.decay_factor(idle_weeks)
                    if factor < 1.0:
                        self._mastery *= factor
            course = self.dataset.courses.get(inter.course_id)
            if course is not None:
                score = self._observed_score(inter)
                effort = (
                    float(inter.hours_spent) / float(course.duration_hours)
                    if course.duration_hours > 0
                    else 0.0
                )
                self._apply_course(course, score, effort)
            if timestamp is not None:
                last_ts = timestamp
        self._invalidate()

    @staticmethod
    def _observed_score(inter: Interaction) -> float:
        """把一条交互记录折算成 ``[0, 1]`` 的软观测分数。

        优先使用结课测验分（最能反映真实掌握），否则按完成情况估计::

            有测验：0.80 * quiz + 0.20 * progress
            已完成：0.65 + 0.20 * progress
            未完成：0.50 * progress
        另外用评分做 ±0.05 的主观满意度修正。
        """
        if inter.quiz_score is not None:
            base = OBS_QUIZ_WEIGHT * float(inter.quiz_score) + OBS_PROGRESS_WEIGHT * float(inter.progress)
        elif inter.completed:
            base = OBS_COMPLETED_BASE + OBS_PROGRESS_WEIGHT * float(inter.progress)
        else:
            base = OBS_UNFINISHED_FACTOR * float(inter.progress)
        if inter.rating is not None:
            base += OBS_RATING_WEIGHT * (float(inter.rating) - 3.0) / 2.0
        return _clip(base, 0.0, 1.0)

    def _apply_course(self, course: Course, score: float, effort_ratio: float) -> dict[str, float]:
        """对一门课执行 BKT 更新（不触发缓存失效，由调用方决定）。"""
        ability = _clip(float(self.learner.ability), 0.0, 1.0)
        effort = _clip(float(effort_ratio), 0.0, 1.0)
        # 努力程度微调有效观测分数：投入不足时表现打折，超额投入不再加分
        effective_score = _clip(float(score) * (0.80 + 0.20 * effort), 0.0, 1.0)

        deltas: dict[str, float] = {}
        for cov in course.kp_coverage:
            index = self._index.get(cov.kp_id)
            if index is None:
                continue
            prior = float(self._mastery[index])
            post = self.tracker.update(prior, effective_score, float(cov.weight), ability)
            self._mastery[index] = post
            deltas[cov.kp_id] = post - prior
        return deltas

    def _invalidate(self) -> None:
        """状态变更后失效所有派生缓存。"""
        self._version += 1
        self._readiness_cache.clear()
        self._gap_cache = None
        self._domain_cache = None

    def _goal_readiness(self) -> float:
        """目标就绪度：对每个目标知识点算「其先修的平均掌握度 → sigmoid」，再取均值。"""
        values: list[float] = []
        for kp_id in self._target_kps:
            index = self._index.get(kp_id)
            if index is None:
                continue
            prereqs = [
                self._index[p]
                for p in self.graph.prerequisite_closure([kp_id])
                if p != kp_id and p in self._index
            ]
            if not prereqs:
                values.append(1.0)
                continue
            mean_mastery = float(self._mastery[prereqs].mean())
            values.append(sigmoid(READINESS_SLOPE * (mean_mastery - READINESS_MID)))
        if not values:
            return 1.0
        return float(sum(values) / len(values))


def _parse_ts(text: str) -> datetime | None:
    """解析 ISO 8601 时间戳，失败返回 None（不因脏数据中断建模）。"""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
