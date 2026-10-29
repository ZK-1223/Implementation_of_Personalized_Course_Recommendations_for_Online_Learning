"""社交学习：相似同伴匹配与协作学习小组（实现要求 ⑦）。

三个能力
--------
1. **同伴匹配** :meth:`SocialMatcher.find_peers`
   相似度由三部分相乘得到（乘性融合，任一项低都会显著压低总分）：

       sim(u, v) = cos(m_u, m_v) × goal_factor × interest_factor

   其中 ``m`` 是学习者的 420 维掌握度向量：

   * **余弦相似度** ``cos = m_u·m_v / (‖m_u‖₂·‖m_v‖₂)`` —— 刻画「知识结构是否相像」，
     比欧氏距离更适合「都懂同一片区域」的语义（绝对水平高低不影响方向）；
   * **同目标加权** ``goal_factor``：同一职业目标 1.0，否则 0.85 ——
     同目标的人分享更多可执行信息（同一批课程、同一批考点），社交价值更高；
   * **兴趣重叠加价** ``interest_factor = 1 + 0.10 × Jaccard(兴趣领域)``。

   三者相乘后裁剪到 ``[0,1]``。另外统计三项「可解释信息」：
   ``shared_goals``（共同目标知识点名称）、``shared_courses``（共同学过 / 在学的课程标题）、
   ``ahead_in``（对方在**我方目标闭包内**掌握度显著更高——差值 ≥ 0.15——
   且自身已达到 0.60 的知识点名称），后者正是「向他/她学什么」的具体建议。

2. **学习小组编组** :meth:`SocialMatcher.form_study_groups`
   对每个目标知识点，把候选学习者按掌握度分成两群人：

   * **已掌握者**（``mastery ≥ MASTERY_MASTERED = 0.85``）→ 可担任「组内小老师」；
   * **未掌握者**（``mastery < MASTERY_GAP = 0.60``）→ 需要帮助的同学。

   然后按 ``1 个已掌握者 + (size-1) 个未掌握者`` 的方式轮转编组，
   让每个小组都有「讲得清的人」和「需要被讲的人」。互补性得分::

       peer_ratio      = 已掌握人数 / 组内人数
       complementarity = 4 · peer_ratio · (1 - peer_ratio)      # 0 单一 → 1 对半，倒 U 型
       synergy         = 0.6 · complementarity + 0.4 · min(1, 组内人数 / size)

   倒 U 型函数刻画「异质小组的认知冲突收益最大，但一旦全是新手就没人是专家」。

3. **同伴信号** :meth:`SocialMatcher.peer_signal`
   相似同伴（同目标 cohort）在该课程上的完成率，三级回退：
   同目标 cohort 完成率 → 全站该课程完成率 → 全站总完成率（0.45）。
   该信号直接作为混合推荐中的 ``social`` 分量（权重 0.08）。

性能
----
同伴的掌握度需要构造其 :class:`~learnpath.learner.profile.LearnerModel`，
构造 + 回放历史约 1~2 ms。因此本类做了两层缓存：
候选池按「同目标优先」裁剪（720 → ~90 人），且 ``LearnerModel`` 按 ``learner_id`` 缓存复用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

from learnpath.config import (
    MASTERY_GAP,
    MASTERY_MASTERED,
    PEER_TOP_K,
    STUDY_GROUP_SIZE,
)
from learnpath.data.schema import Course, Dataset, Learner
from learnpath.knowledge.graph import KnowledgeGraph

#: 余弦相似度之外的同目标加权（同目标 / 不同目标）
GOAL_FACTOR_SAME: Final[float] = 1.00
GOAL_FACTOR_OTHER: Final[float] = 0.85

#: 兴趣重叠（Jaccard）带来的相似度加成系数
INTEREST_BOOST: Final[float] = 0.10

#: "对方更擅长"的判定阈值：掌握度差值下限
AHEAD_DELTA: Final[float] = 0.15

#: ``ahead_in`` 最多给出的知识点条数
AHEAD_LIMIT: Final[int] = 5

#: 每个知识点最多考察的编组候选人数（控制 LearnerModel 构造次数）
MAX_GROUP_CANDIDATES: Final[int] = 24

#: 全站完成率的兜底先验
PRIOR_COMPLETION: Final[float] = 0.45

_EPS: Final[float] = 1e-9


# ============================================================
# 数据结构
# ============================================================


@dataclass(slots=True)
class PeerMatch:
    """一个相似同伴。

    Attributes:
        learner: 同伴实体。
        similarity: 综合相似度 ``∈ [0,1]``（余弦 × 同目标加权 × 兴趣加成）。
        shared_goals: 共同目标知识点**名称**列表。
        shared_courses: 共同学过 / 在学的课程**标题**列表。
        ahead_in: 对方更擅长的知识点**名称**列表（我已在该目标闭包内）。
        completed_count: 对方已完成课程数（便于前端展示「学习量」）。
        goal_name: 对方的职业目标名称。
        cosine: 纯掌握度向量余弦（便于解释相似度来源）。
    """

    learner: Learner
    similarity: float
    shared_goals: list[str]
    shared_courses: list[str]
    ahead_in: list[str]
    completed_count: int = 0
    goal_name: str = ""
    cosine: float = 0.0

    def to_dict(self, graph: KnowledgeGraph | None = None) -> dict[str, Any]:
        """序列化为 ``docs/API.md`` 第 6 节的 ``peers[i]``。"""
        return {
            "learner_id": self.learner.learner_id,
            "nickname": self.learner.nickname,
            "ability": round(float(self.learner.ability), 3),
            "goal_name": self.goal_name,
            "similarity": round(float(self.similarity), 3),
            "shared_goal_kps": list(self.shared_goals),
            "shared_courses": list(self.shared_courses),
            "ahead_in": list(self.ahead_in),
            "completed_count": int(self.completed_count),
        }


@dataclass(slots=True)
class StudyGroup:
    """一个协作学习小组。

    Attributes:
        group_id: 组 id，如 ``G-01``。
        kp_ids: 本组聚焦的知识点 id。
        members: 成员实体列表（已掌握者排在前）。
        avg_completion: 成员平均历史完成率。
        synergy: 互补性得分 ``∈ [0,1]``（公式见模块 docstring 第 2 节）。
        mastery_mean: 组内在聚焦知识点上的平均掌握度（解释用）。
        mastery_std: 组内在聚焦知识点上的掌握度标准差（异质性）。
    """

    group_id: str
    kp_ids: list[str]
    members: list[Learner]
    avg_completion: float
    synergy: float
    mastery_mean: float = 0.0
    mastery_std: float = 0.0

    def to_dict(
        self,
        graph: KnowledgeGraph | None = None,
        dataset: Dataset | None = None,
    ) -> dict[str, Any]:
        """序列化为 ``docs/API.md`` 第 6 节的 ``groups[i]``。"""
        return {
            "group_id": self.group_id,
            "kp_ids": list(self.kp_ids),
            "kp_names": list(graph.names_of(self.kp_ids)) if graph is not None else list(self.kp_ids),
            "members": [
                {
                    "learner_id": member.learner_id,
                    "nickname": member.nickname,
                    "ability": round(float(member.ability), 3),
                }
                for member in self.members
            ],
            "avg_completion": round(float(self.avg_completion), 3),
            "synergy": round(float(self.synergy), 3),
        }


# ============================================================
# 匹配器
# ============================================================


class SocialMatcher:
    """相似学习者检索 + 学习小组编组 + 同伴行为信号。

    Args:
        dataset: 全量数据集。
        graph: 知识点图。

    Note:
        ``LearnerModel`` 的构造是有成本的（回放历史），因此按 ``learner_id`` 缓存；
        缓存**不随外部状态变化失效**——社交匹配用的是「历史画像」而不是实时状态，
        这是刻意设计（避免每次反馈都把全 cohort 的向量重算一遍）。
    """

    def __init__(self, dataset: Dataset, graph: KnowledgeGraph) -> None:
        self.dataset = dataset
        self.graph = graph
        self._models: dict[str, Any] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._goal_closure: dict[str, frozenset[str]] = {}
        self._learner_courses: dict[str, frozenset[str]] = {}
        self._learner_completed: dict[str, int] = {}
        self._learner_history_n: dict[str, int] = {}
        self._course_stats: dict[str, tuple[int, int]] = {}
        self._cohort_by_goal: dict[str, list[str]] = {}
        self._cohort_course_stats: dict[str, dict[str, list[int]]] = {}
        self._records_by_learner: dict[str, tuple[Any, ...]] | None = None
        # 知识点 -> 掌握度向量下标（顺序与 LearnerModel.kp_ids 一致）
        self._kp_positions: dict[str, int] = {
            kp_id: index for index, kp_id in enumerate(dataset.knowledge_points)
        }

        self._build_indexes()

    # --------------------------------------------------------
    # 索引
    # --------------------------------------------------------
    def _build_indexes(self) -> None:
        """一次扫描 ``dataset.interactions`` 建立课程 / 学习者侧统计。"""
        courses_of: dict[str, set[str]] = {}
        completed: dict[str, int] = {}
        history_n: dict[str, int] = {}
        course_stats: dict[str, list[int]] = {}

        for record in self.dataset.interactions:
            learner_id = record.learner_id
            courses_of.setdefault(learner_id, set()).add(record.course_id)
            history_n[learner_id] = history_n.get(learner_id, 0) + 1
            if record.completed:
                completed[learner_id] = completed.get(learner_id, 0) + 1
            stats = course_stats.setdefault(record.course_id, [0, 0])
            stats[0] += 1
            if record.completed:
                stats[1] += 1

        self._learner_courses = {k: frozenset(v) for k, v in courses_of.items()}
        self._learner_completed = completed
        self._learner_history_n = history_n
        self._course_stats = {k: (v[0], v[1]) for k, v in course_stats.items()}

        by_goal: dict[str, list[str]] = {}
        for learner_id, learner in self.dataset.learners.items():
            by_goal.setdefault(learner.goal_id, []).append(learner_id)
        for ids in by_goal.values():
            ids.sort()
        self._cohort_by_goal = by_goal

    def _closure_of_goal(self, goal_id: str) -> frozenset[str]:
        cached = self._goal_closure.get(goal_id)
        if cached is None:
            goal = self.dataset.goals[goal_id]
            cached = self.graph.prerequisite_closure(goal.target_kp_ids)
            self._goal_closure[goal_id] = cached
        return cached

    def _model_of(self, learner_id: str) -> Any:
        """取得（并缓存）某学习者的 ``LearnerModel``。"""
        cached = self._models.get(learner_id)
        if cached is not None:
            return cached
        from learnpath.learner.profile import LearnerModel  # 延迟导入，避免循环依赖

        model = LearnerModel(self.dataset.learners[learner_id], self.dataset, self.graph)
        self._models[learner_id] = model
        self._vectors[learner_id] = np.asarray(model.mastery_array(copy=True), dtype=float)
        return model

    def _vector_of(self, learner_id: str) -> np.ndarray:
        cached = self._vectors.get(learner_id)
        if cached is None:
            self._model_of(learner_id)
            cached = self._vectors[learner_id]
        return cached

    # --------------------------------------------------------
    # 同伴匹配
    # --------------------------------------------------------
    def find_peers(self, model: Any, k: int = PEER_TOP_K) -> list[PeerMatch]:
        """找出 ``k`` 个最相似的学习同伴。

        Args:
            model: 当前学习者的 ``LearnerModel``。
            k: 返回条数（默认 ``PEER_TOP_K = 6``）。

        Returns:
            按 ``similarity`` 降序的 :class:`PeerMatch` 列表。
            若数据集中没有其他学习者则返回空列表。
        """
        me = model.learner
        my_vec = np.asarray(model.mastery_array(copy=True), dtype=float)
        my_norm = float(np.linalg.norm(my_vec))
        my_targets = set(getattr(model, "target_kps", self.dataset.goals[me.goal_id].target_kp_ids))
        my_closure = self._closure_of_goal(me.goal_id)
        my_courses = self._learner_courses.get(me.learner_id, frozenset())
        my_interests = set(me.interests)

        # 候选池：同目标优先，人数不足时按能力接近程度补足
        same_goal = [lid for lid in self._cohort_by_goal.get(me.goal_id, []) if lid != me.learner_id]
        candidates = list(same_goal)
        if len(candidates) < k:
            extra = [
                lid
                for lid in sorted(self.dataset.learners)
                if lid != me.learner_id and lid not in set(candidates)
            ]
            extra.sort(key=lambda lid: abs(float(self.dataset.learners[lid].ability) - float(me.ability)))
            candidates.extend(extra[: max(0, k * 3 - len(candidates))])

        matches: list[PeerMatch] = []
        for other_id in candidates:
            other = self.dataset.learners[other_id]
            vec = self._vector_of(other_id)
            other_norm = float(np.linalg.norm(vec))
            cosine = (
                float(np.dot(my_vec, vec)) / (my_norm * other_norm)
                if my_norm > _EPS and other_norm > _EPS
                else 0.0
            )
            cosine = max(0.0, min(1.0, cosine))

            goal_factor = GOAL_FACTOR_SAME if other.goal_id == me.goal_id else GOAL_FACTOR_OTHER
            interest_factor = 1.0 + INTEREST_BOOST * _jaccard(my_interests, set(other.interests))
            similarity = max(0.0, min(1.0, cosine * goal_factor * interest_factor))
            if similarity <= 0.0:
                continue

            # 共同目标知识点
            other_targets = set(
                self.dataset.goals[other.goal_id].target_kp_ids
                if other.goal_id in self.dataset.goals
                else ()
            )
            shared_names = self.graph.names_of(sorted(my_targets & other_targets))

            # 共同课程
            shared_course_ids = sorted(my_courses & self._learner_courses.get(other_id, frozenset()))
            shared_courses = [
                self.dataset.courses[cid].title
                for cid in shared_course_ids[:3]
                if cid in self.dataset.courses
            ]

            # 对方更擅长的知识点（在我方目标闭包内）
            ahead: list[tuple[str, float]] = []
            for kp_id in my_closure:
                their = float(vec[self._index_of(other_id, kp_id)])
                mine = float(model.mastery_of(kp_id))
                if their >= MASTERY_GAP and their - mine >= AHEAD_DELTA:
                    ahead.append((kp_id, their - mine))
            ahead.sort(key=lambda item: -item[1])
            ahead_names = self.graph.names_of([kp for kp, _ in ahead[:AHEAD_LIMIT]])

            matches.append(
                PeerMatch(
                    learner=other,
                    similarity=similarity,
                    shared_goals=shared_names,
                    shared_courses=shared_courses,
                    ahead_in=ahead_names,
                    completed_count=int(self._learner_completed.get(other_id, 0)),
                    goal_name=self.dataset.goals[other.goal_id].name
                    if other.goal_id in self.dataset.goals
                    else "",
                    cosine=cosine,
                )
            )

        matches.sort(key=lambda m: (-m.similarity, m.learner.learner_id))
        return matches[: max(0, int(k))]

    def _index_of(self, learner_id: str, kp_id: str) -> int:
        """知识点在掌握度向量中的下标（与 ``LearnerModel.kp_ids`` 顺序一致）。"""
        return self._kp_positions.get(kp_id, 0)

    # --------------------------------------------------------
    # 学习小组
    # --------------------------------------------------------
    def form_study_groups(
        self,
        kp_ids: Iterable[str],
        size: int = STUDY_GROUP_SIZE,
    ) -> list[StudyGroup]:
        """按目标知识点编组「能力互补」的协作学习小组。

        Args:
            kp_ids: 需要组队攻克的知识点 id 集合。
            size: 每组人数（默认 ``STUDY_GROUP_SIZE = 4``）。

        Returns:
            :class:`StudyGroup` 列表，按 ``synergy`` 降序。每个知识点最多产出一组。

        Note:
            为保证响应时间，每个知识点的候选学习者按 ``learner_id`` 取前
            ``MAX_GROUP_CANDIDATES = 24`` 人；被考察的学习者 ``LearnerModel``
            会被缓存复用。
        """
        size = max(2, int(size))
        wanted = [kp_id for kp_id in kp_ids if kp_id in self.dataset.knowledge_points]
        groups: list[StudyGroup] = []

        for kp_id in wanted:
            candidates = self._group_candidates(kp_id)
            if len(candidates) < 2:
                continue
            scored: list[tuple[str, float]] = []
            for learner_id in candidates:
                mastery = float(self._model_of(learner_id).mastery_of(kp_id))
                scored.append((learner_id, mastery))
            scored.sort(key=lambda item: (-item[1], item[0]))

            experts = [lid for lid, m in scored if m >= MASTERY_MASTERED]
            novices = [lid for lid, m in scored if m < MASTERY_GAP]
            if not experts or not novices:
                # 没有任何互补性：退化为「按掌握度分层」的同质小组（synergy 会更低）
                members_ids = [lid for lid, _ in scored[:size]]
                if len(members_ids) < 2:
                    continue
                values = [m for _lid, m in scored[:size]]
                groups.append(
                    self._make_group(
                        index=len(groups) + 1,
                        kp_id=kp_id,
                        member_ids=members_ids,
                        values=values,
                        n_experts=sum(1 for v in values if v >= MASTERY_MASTERED),
                        size=size,
                    )
                )
                continue

            take_novices = max(1, size - 1)
            picked = [experts[0]] + novices[:take_novices]
            # 人数不足时用其余学习者补齐
            if len(picked) < size:
                rest = [
                    lid
                    for lid, _m in scored
                    if lid not in set(picked)
                ]
                picked.extend(rest[: size - len(picked)])
            values = [float(self._model_of(lid).mastery_of(kp_id)) for lid in picked]
            groups.append(
                self._make_group(
                    index=len(groups) + 1,
                    kp_id=kp_id,
                    member_ids=picked,
                    values=values,
                    n_experts=sum(1 for v in values if v >= MASTERY_MASTERED),
                    size=size,
                )
            )
            continue

        groups.sort(key=lambda g: (-g.synergy, g.group_id))
        return groups

    def _group_candidates(self, kp_id: str) -> list[str]:
        """目标闭包含该知识点的学习者（截断到 ``MAX_GROUP_CANDIDATES`` 人）。"""
        result: list[str] = []
        for learner_id in sorted(self.dataset.learners):
            learner = self.dataset.learners[learner_id]
            if kp_id in self._closure_of_goal(learner.goal_id):
                result.append(learner_id)
                if len(result) >= MAX_GROUP_CANDIDATES:
                    break
        return result

    def _make_group(
        self,
        index: int,
        kp_id: str,
        member_ids: Sequence[str],
        values: Sequence[float],
        n_experts: int,
        size: int,
    ) -> StudyGroup:
        """组装 :class:`StudyGroup` 并计算 ``synergy``。"""
        members = [self.dataset.learners[lid] for lid in member_ids]
        n = len(members)
        peer_ratio = n_experts / n if n else 0.0
        complementarity = 4.0 * peer_ratio * (1.0 - peer_ratio)  # 倒 U 型
        synergy = 0.6 * complementarity + 0.4 * min(1.0, n / max(1, size))

        rates = [self._completion_rate_of(lid) for lid in member_ids]
        mean_value = float(sum(values) / len(values)) if values else 0.0
        if len(values) > 1:
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            std_value = math.sqrt(variance)
        else:
            std_value = 0.0

        return StudyGroup(
            group_id=f"G-{index:02d}",
            kp_ids=[kp_id],
            members=members,
            avg_completion=float(sum(rates) / len(rates)) if rates else 0.0,
            synergy=float(max(0.0, min(1.0, synergy))),
            mastery_mean=mean_value,
            mastery_std=std_value,
        )

    def _completion_rate_of(self, learner_id: str) -> float:
        total = self._learner_history_n.get(learner_id, 0)
        if total <= 0:
            return PRIOR_COMPLETION
        return self._learner_completed.get(learner_id, 0) / total

    # --------------------------------------------------------
    # 同伴行为信号
    # --------------------------------------------------------
    def peer_signal(self, model: Any, course: Course) -> float:
        """相似同伴（同目标 cohort）在该课程上的完成率 ``∈ [0,1]``。

        三级回退：同目标 cohort 完成率 → 全站该课程完成率 → 全站完成率先验。
        该值直接作为混合推荐里的 ``social`` 分量。

        Args:
            model: 当前学习者模型（只需 ``learner.goal_id``）。
            course: 待评估课程。
        """
        me = model.learner
        stats_of_cohort = self._cohort_course_table(me.goal_id)
        stats = stats_of_cohort.get(course.course_id)
        if stats is not None and stats[0] > 0:
            return float(stats[1] / stats[0])

        global_stats = self._course_stats.get(course.course_id)
        if global_stats is not None and global_stats[0] > 0:
            return float(global_stats[1] / global_stats[0])

        n = len(self.dataset.interactions)
        if n == 0:
            return PRIOR_COMPLETION
        return float(sum(1 for r in self.dataset.interactions if r.completed) / n)

    def _cohort_course_table(self, goal_id: str) -> dict[str, list[int]]:
        """同目标 cohort 在每门课上的 ``[学习人数, 完成人数]``（按 goal 惰性构建并缓存）。"""
        cached = self._cohort_course_stats.get(goal_id)
        if cached is not None:
            return cached
        table: dict[str, list[int]] = {}
        for learner_id in self._cohort_by_goal.get(goal_id, ()):
            for record in self._records_of(learner_id):
                bucket = table.setdefault(record.course_id, [0, 0])
                bucket[0] += 1
                if record.completed:
                    bucket[1] += 1
        self._cohort_course_stats[goal_id] = table
        return table

    def _records_of(self, learner_id: str) -> tuple[Any, ...]:
        """某学习者的全部交互记录（惰性索引）。"""
        if self._records_by_learner is None:
            grouped: dict[str, list[Any]] = {}
            for record in self.dataset.interactions:
                grouped.setdefault(record.learner_id, []).append(record)
            self._records_by_learner = {k: tuple(v) for k, v in grouped.items()}
        return self._records_by_learner.get(learner_id, ())

    # --------------------------------------------------------
    # 汇总
    # --------------------------------------------------------
    def cohort_summary(self, model: Any, peers: Sequence[PeerMatch]) -> dict[str, Any]:
        """``docs/API.md`` 第 6 节的 ``cohort_summary`` 字段。"""
        me = model.learner
        my_rate = self._completion_rate_of(me.learner_id)
        cohort = [lid for lid in self._cohort_by_goal.get(me.goal_id, [])]
        cohort_rate = (
            sum(self._completion_rate_of(lid) for lid in cohort) / len(cohort) if cohort else my_rate
        )
        avg_sim = (
            sum(float(p.similarity) for p in peers) / len(peers) if peers else 0.0
        )
        return {
            "n_peers": len(peers),
            "avg_similarity": round(avg_sim, 3),
            "cohort_completion_rate": round(cohort_rate, 3),
            "your_completion_rate": round(my_rate, 3),
        }


def _jaccard(left: set[str], right: set[str]) -> float:
    """集合 Jaccard 相似度（空集返回 0）。"""
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


__all__ = ["SocialMatcher", "PeerMatch", "StudyGroup"]
