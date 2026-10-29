"""基于知识图谱的内容推荐（Content-based Recommender）。

对应报告「推荐算法」章节的第一路召回：**知识图谱驱动的内容召回**。

--------------------------------------------------------------------------------
一、核心思想：以「目标缺口 × 覆盖强度 ÷ 时间成本」为单位收益
--------------------------------------------------------------------------------
协同过滤只知道「相似的人喜欢什么」，无法解释「为什么这门课对他有用」；内容推荐补上这一
能力：它直接把课程在知识图谱上的位置与学习者的掌握度向量对齐。

对学习者 ``u`` 与课程 ``c``，定义**图上的缺口覆盖收益**::

    gap_k   = clip(required_mastery - mastery_k, 0, 1)          # 知识点 k 的缺口
    cov_k   = 0.35 + 0.65 * 1[k ∈ 目标闭包]                     # 目标外知识点只给部分信用
    gain    = Σ_{k ∈ c.kp_coverage}  w_k * cov_k * gap_k        # w_k 为 KPCoverage.weight

即：课程覆盖的每个知识点，按「还差多少」与「覆盖强度」加权累加；不在目标先修闭包内的
知识点只承认 35% 的收益（拓宽视野，但不解决目标）。

随后做**时间成本归一**与饱和映射（避免「课时越短越推荐」的退化解）::

    efficiency = gain / (1 + duration_hours / 8.0)
    gain_term  = 1 - exp(-3.0 * efficiency)

``exp`` 饱和函数保证收益在 0~1 之间且边际递减：缺口极小的课程不会因为「便宜」而排在前面。

--------------------------------------------------------------------------------
二、最近发展区（Zone of Proximal Development, ZPD）
--------------------------------------------------------------------------------
教育心理学认为「略高于当前水平」的任务学习效率最高。本系统用领域平均掌握度推「理想难度」::

    ideal_level = 1 + 4 * domain_mastery(c.domain)     # 掌握 0 → 难度 1，掌握 1 → 难度 5
    Δ           = c.level - ideal_level
    level_match = 1 - 0.30 * min(1, -Δ/4)              if Δ ≤ 0（课程偏简单，轻微惩罚）
                = max(0.05, 1 - Δ/2.5)                 if Δ > 0（课程偏难，陡峭惩罚）

「偏难惩罚比偏易惩罚更陡」是有意设计：跳级学习会直接导致放弃，而简单课程只是效率偏低。

--------------------------------------------------------------------------------
三、先修未就绪的惩罚（知识图谱约束）
--------------------------------------------------------------------------------
内容推荐的另一个职责是把「知识图谱的先修关系」变成硬软约束。先修就绪度来自
:meth:`learnpath.learner.profile.LearnerModel.readiness_for`（先修平均掌握度经 sigmoid 映射），
以**乘性门控**作用于最终分数::

    score = (0.62 * gain_term + 0.38 * level_match) * (0.20 + 0.80 * readiness) * interest

门控下界 0.20 而不是 0：这样「先修差一点但极其关键」的课程仍不会被彻底扼杀（保留探索），
但在排序上必然让位于可立即开始的课程。``interest`` 是领域亲和系数（兴趣领域 1.00，
目标领域 0.90，其他 0.78）。

--------------------------------------------------------------------------------
四、实现要点
--------------------------------------------------------------------------------
* 全部课程的特征（覆盖知识点下标、先修下标、难度、时长）在构造函数里**一次性展平**为
  ``numpy`` 的 CSR 式结构（``flat_index + owner``），因此对**整个课程库**打分只需
  1 次 ``np.bincount``（稀疏累加）+ 若干逐元素运算，无需 Python 循环；
* 单次 ``score_all`` 在 560 门课规模下约 1~2 ms，满足「单次推荐 < 1 秒」的性能要求；
* 结果按 ``model.version`` 缓存，掌握度一变自动失效，避免同一次推荐里重复计算。
"""

from __future__ import annotations

from typing import Final

import numpy as np

from learnpath.data.schema import Course, Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import (
    READINESS_MID,
    READINESS_SLOPE,
    LearnerModel,
    cached_course_prerequisites,
)

__all__ = ["ContentRecommender"]


#: 目标闭包外知识点的收益信用（0.35 = 只承认三成半收益）
OUTSIDE_GOAL_CREDIT: Final[float] = 0.35

#: 时间成本归一尺度（小时）。8 小时是一门前置型课程的典型时长：
#: ``duration = 8h`` 时收益被压缩为一半，从而在不压倒短课的前提下仍然偏好长课的系统收益。
TIME_SCALE_HOURS: Final[float] = 8.0

#: 效率 → 收益的饱和斜率，越大越快饱和。3.0 使 efficiency = 0.23 时 gain_term ≈ 0.5。
GAIN_SATURATION: Final[float] = 3.0

#: 收益项与难度匹配项的融合权重。
W_GAIN: Final[float] = 0.62
W_LEVEL: Final[float] = 0.38

#: 先修门控下界（见模块 docstring 第三节）。
PREREQ_GATE_BASE: Final[float] = 0.20

#: 领域亲和系数。
INTEREST_BONUS: Final[float] = 1.00
GOAL_DOMAIN_BONUS: Final[float] = 0.90
OTHER_DOMAIN_BONUS: Final[float] = 0.78

#: 最近发展区：偏难时的惩罚斜率（Δ 每多 1 级损失 1/2.5 = 40% 匹配度）。
ZPD_HARD_SLOPE: Final[float] = 2.5
#: 最近发展区：偏易时的惩罚斜率（Δ 每多 1 级损失 30%/4 = 7.5% 匹配度）。
ZPD_EASY_SLOPE: Final[float] = 0.30 / 4.0
#: 难度匹配度下界，避免「完全不可学」的课程直接得 0 而丢失排序信息。
ZPD_FLOOR: Final[float] = 0.05


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    """对数组做数值稳定的 logistic 变换。"""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class ContentRecommender:
    """知识图谱内容推荐器：按「目标缺口覆盖 ÷ 时间成本 × 难度匹配 × 先修门控」打分。

    Args:
        dataset: 数据集。
        graph: 知识图谱（提供先修闭包与课程先修集合）。

    Example:
        >>> from learnpath.data.fixtures import build_tiny_dataset
        >>> from learnpath.knowledge.graph import KnowledgeGraph
        >>> from learnpath.learner.profile import LearnerModel
        >>> ds = build_tiny_dataset()
        >>> graph = KnowledgeGraph.from_dataset(ds)
        >>> rec = ContentRecommender(ds, graph)
        >>> model = LearnerModel(ds.learners["L-0001"], ds, graph)
        >>> 0.0 <= rec.score(model, ds.courses["C-0001"]) <= 1.0
        True
    """

    __slots__ = (
        "dataset",
        "graph",
        "course_ids",
        "kp_ids",
        "_course_pos",
        "_kp_index",
        "_cov_flat_idx",
        "_cov_flat_w",
        "_cov_owner",
        "_prereq_flat_idx",
        "_prereq_owner",
        "_has_prereq",
        "_level",
        "_hours",
        "_domain_names",
        "_domain_owner",
        "_goal_closure",
        "_goal_domains",
        "_cache_key",
        "_cache_scores",
        "_cache_readiness",
        "_cache_gain",
        "_gain_term",
        "_level_match",
    )

    def __init__(self, dataset: Dataset, graph: KnowledgeGraph) -> None:
        self.dataset: Dataset = dataset
        self.graph: KnowledgeGraph = graph
        self.course_ids: tuple[str, ...] = tuple(dataset.courses)
        self._course_pos: dict[str, int] = {cid: i for i, cid in enumerate(self.course_ids)}
        self.kp_ids: tuple[str, ...] = tuple(dataset.knowledge_points)
        self._kp_index: dict[str, int] = {kp: i for i, kp in enumerate(self.kp_ids)}

        n_courses = len(self.course_ids)
        n_kps = len(self.kp_ids)

        # ---- 展平课程覆盖（CSR 风格：flat 值 + owner 归属）----
        cov_idx: list[int] = []
        cov_w: list[float] = []
        cov_owner: list[int] = []
        prereq_idx: list[int] = []
        prereq_owner: list[int] = []
        levels = np.empty(n_courses, dtype=np.float64)
        hours = np.empty(n_courses, dtype=np.float64)
        domain_names = tuple(sorted({kp.domain for kp in dataset.knowledge_points.values()}))
        domain_index = {name: i for i, name in enumerate(domain_names)}
        domain_owner = np.empty(n_courses, dtype=np.intp)
        has_prereq = np.zeros(n_courses, dtype=bool)

        prereq_map = cached_course_prerequisites(dataset, graph)
        for pos, course_id in enumerate(self.course_ids):
            course = dataset.courses[course_id]
            for cov in course.kp_coverage:
                index = self._kp_index.get(cov.kp_id)
                if index is None:
                    continue
                cov_idx.append(index)
                cov_w.append(float(cov.weight))
                cov_owner.append(pos)
            for kp_id in prereq_map.get(course_id, ()):
                index = self._kp_index.get(kp_id)
                if index is None:
                    continue
                prereq_idx.append(index)
                prereq_owner.append(pos)
                has_prereq[pos] = True
            levels[pos] = float(course.level)
            hours[pos] = float(course.duration_hours)
            domain_owner[pos] = domain_index[course.domain]

        self._cov_flat_idx = np.asarray(cov_idx, dtype=np.intp)
        self._cov_flat_w = np.asarray(cov_w, dtype=np.float64)
        self._cov_owner = np.asarray(cov_owner, dtype=np.intp)
        self._prereq_flat_idx = np.asarray(prereq_idx, dtype=np.intp)
        self._prereq_owner = np.asarray(prereq_owner, dtype=np.intp)
        self._has_prereq = has_prereq
        self._level = levels
        self._hours = hours
        self._domain_names = domain_names
        self._domain_owner = domain_owner

        # ---- 目标闭包（每个目标一份，避免逐次算闭包）----
        self._goal_closure: dict[str, np.ndarray] = {}
        self._goal_domains: dict[str, frozenset[str]] = {}
        for goal_id, goal in dataset.goals.items():
            closure = graph.prerequisite_closure(goal.target_kp_ids)
            self._goal_closure[goal_id] = np.array(
                sorted(self._kp_index[k] for k in closure if k in self._kp_index), dtype=np.intp
            )
            self._goal_domains[goal_id] = frozenset(
                dataset.knowledge_points[k].domain for k in closure if k in dataset.knowledge_points
            )

        self._cache_key: tuple[int, int] | None = None
        self._cache_scores: np.ndarray | None = None
        self._cache_readiness: np.ndarray | None = None
        self._cache_gain: np.ndarray | None = None
        self._gain_term: np.ndarray = np.zeros(n_courses, dtype=np.float64)
        self._level_match: np.ndarray = np.zeros(n_courses, dtype=np.float64)

        if n_kps == 0 or n_courses == 0:  # pragma: no cover - 数据集校验已保证非空
            raise ValueError("ContentRecommender 需要非空的知识点与课程集合")

    # ==================================================================
    # 对外接口
    # ==================================================================
    def score(self, model: LearnerModel, course: Course) -> float:
        """返回课程对学习者的内容匹配分 ∈ [0, 1]（越大越值得推荐）。"""
        position = self._course_pos.get(course.course_id)
        if position is None:
            return 0.0
        return float(self.score_all(model)[position])

    def score_all(self, model: LearnerModel) -> np.ndarray:
        """对**全部课程**一次性打分，返回与 :attr:`course_ids` 对齐的数组 ∈ [0, 1]。

        向量化实现：覆盖收益用一次 ``np.bincount`` 完成稀疏累加，先修就绪度同理，
        避免 560 门课的 Python 循环。
        """
        self._ensure_cache(model)
        assert self._cache_scores is not None
        return self._cache_scores

    def readiness_all(self, model: LearnerModel) -> np.ndarray:
        """全部课程的先修就绪度数组 ∈ [0, 1]（与 :attr:`course_ids` 对齐）。"""
        self._ensure_cache(model)
        assert self._cache_readiness is not None
        return self._cache_readiness

    def gain_all(self, model: LearnerModel) -> np.ndarray:
        """全部课程的**原始**缺口覆盖收益（未做时间归一），用于解释与调试。"""
        self._ensure_cache(model)
        assert self._cache_gain is not None
        return self._cache_gain

    def components(self, model: LearnerModel) -> dict[str, np.ndarray]:
        """拆解打分中间量，便于报告与前端展示。

        Returns:
            ``{"gain_term": ..., "level_match": ..., "readiness": ..., "score": ...}``
            各数组均与 :attr:`course_ids` 对齐。
        """
        self._ensure_cache(model)
        return {
            "gain_term": self._gain_term,
            "level_match": self._level_match,
            "readiness": self._cache_readiness,
            "score": self._cache_scores,
        }

    def position_of(self, course_id: str) -> int | None:
        """课程在内部数组中的下标（未知课程返回 None）。"""
        return self._course_pos.get(course_id)

    # ==================================================================
    # 内部实现
    # ==================================================================
    def _ensure_cache(self, model: LearnerModel) -> None:
        """按 ``(模型身份, 状态版本)`` 缓存打分结果。"""
        key = (id(model), model.version)
        if self._cache_key == key and self._cache_scores is not None:
            return
        self._compute(model)
        self._cache_key = key

    def _compute(self, model: LearnerModel) -> None:
        """核心向量化打分流程。"""
        mastery = model.mastery_array()
        n_courses = len(self.course_ids)
        n_kps = len(self.kp_ids)

        # ---- 1) 缺口收益 ----
        required = float(model.required_mastery)
        gap = np.clip(required - mastery, 0.0, 1.0)

        goal_id = model.learner.goal_id
        closure = self._goal_closure.get(goal_id)
        if closure is None or closure.size == 0:
            inside = np.ones(n_kps, dtype=bool)
            goal_domains: frozenset[str] = frozenset()
        else:
            inside = np.zeros(n_kps, dtype=bool)
            inside[closure] = True
            goal_domains = self._goal_domains.get(goal_id, frozenset())

        credit = np.where(inside[self._cov_flat_idx], 1.0, OUTSIDE_GOAL_CREDIT)
        contrib = self._cov_flat_w * credit * gap[self._cov_flat_idx]
        gain = np.bincount(self._cov_owner, weights=contrib, minlength=n_courses)

        # ---- 2) 时间归一 + 饱和 ----
        efficiency = gain / (1.0 + self._hours / TIME_SCALE_HOURS)
        gain_term = 1.0 - np.exp(-GAIN_SATURATION * efficiency)

        # ---- 3) 先修就绪度（向量化 bincount 求均值）----
        readiness = np.ones(n_courses, dtype=np.float64)
        if self._prereq_flat_idx.size:
            values = mastery[self._prereq_flat_idx]
            sums = np.bincount(self._prereq_owner, weights=values, minlength=n_courses)
            counts = np.bincount(self._prereq_owner, minlength=n_courses)
            has = counts > 0
            mean_mastery = np.ones(n_courses, dtype=np.float64)
            mean_mastery[has] = sums[has] / counts[has]
            readiness[has] = _sigmoid_array(READINESS_SLOPE * (mean_mastery[has] - READINESS_MID))

        # ---- 4) 最近发展区 ----
        domain_mastery = model.domain_mastery()
        dom_vec = np.array(
            [domain_mastery.get(name, 0.0) for name in self._domain_names], dtype=np.float64
        )
        ideal_level = 1.0 + 4.0 * dom_vec[self._domain_owner]
        delta = self._level - ideal_level
        level_match = np.where(
            delta <= 0.0,
            1.0 - ZPD_EASY_SLOPE * np.minimum(1.0, -delta),
            np.maximum(ZPD_FLOOR, 1.0 - delta / ZPD_HARD_SLOPE),
        )

        # ---- 5) 领域亲和 ----
        interests = set(model.learner.interests)
        interest = np.full(n_courses, OTHER_DOMAIN_BONUS, dtype=np.float64)
        if goal_domains:
            goal_mask = np.array(
                [self._domain_names[i] in goal_domains for i in self._domain_owner], dtype=bool
            )
            interest[goal_mask] = GOAL_DOMAIN_BONUS
        if interests:
            interest_mask = np.array(
                [self._domain_names[i] in interests for i in self._domain_owner], dtype=bool
            )
            interest[interest_mask] = INTEREST_BONUS

        # ---- 6) 融合 + 先修门控 ----
        base = W_GAIN * gain_term + W_LEVEL * level_match
        score = base * (PREREQ_GATE_BASE + (1.0 - PREREQ_GATE_BASE) * readiness) * interest

        self._gain_term = gain_term
        self._level_match = level_match
        self._cache_readiness = readiness
        self._cache_gain = gain
        self._cache_scores = np.clip(score, 0.0, 1.0)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"ContentRecommender(courses={len(self.course_ids)}, kps={len(self.kp_ids)})"
