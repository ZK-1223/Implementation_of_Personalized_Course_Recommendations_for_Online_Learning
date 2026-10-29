"""学习效果预测的特征工程（实现要求 ⑤ 的前半部分）。

设计原则
--------
1. **零标签泄漏（no label leakage）**：所有特征在「学习者尚未学过这门课」的
   时点上都可观测。具体地说：
     * ``hist_*`` 系列历史统计**显式排除当前课程**那一条交互记录；
     * 掌握度类特征依赖 ``LearnerModel``，调用方在训练时必须用
       「排除当前课程」的数据集视图构造 model（见
       :meth:`learnpath.prediction.effect.EffectPredictor.fit`），
       否则模型回放历史时会把该课的结课成绩也算进掌握度，造成泄漏。
2. **完全确定性**：不含任何随机数，同一输入永远得到同一输出。
3. **高频可用**：训练阶段要对 14000 条交互逐条构造特征，因此在模块级缓存了
   「交互倒排索引」与「课程先修闭包」，把单次构造成本压到微秒级。

特征清单（23 个，全部为 ``float``，均为「越大越有利于完成 / 增益越高」的
单调方向**不做强制**，方向由模型自行学习）
------------------------------------------------------------------------
==================== ==========================================================
特征名                含义与取值
==================== ==========================================================
learner_ability     学习者综合能力 ``A ∈ [0,1]``（画像自带）
avg_mastery         目标先修闭包内的平均掌握度 ``∈ [0,1]``
target_avg_mastery  目标知识点本身（不含先修）的平均掌握度
readiness           课程先修就绪度 ``LearnerModel.readiness_for(c) ∈ [0,1]``
style_fit           学习风格匹配度 ``∈ [0,1]``
time_fit            时间可承受度 ``∈ [0,1]``
expected_gain_model 模型预估的掌握度增益（BKT 单步模拟，非标签）
course_level        课程难度 ``1~5``
level_gap           难度差 ``= (course.level - 领域现有水平) / 4 ∈ [-1,1]``
duration_hours      课程时长（小时）
log_duration        ``log(1 + duration)``，压缩长尾
n_kp_covered        课程覆盖知识点个数
prereq_uncovered_ratio 先修中「尚未掌握（< 0.60）」的比例
gap_coverage_ratio  课程覆盖知识点中落在目标闭包内的比例
hist_completion_rate 该学习者历史完成率（**排除当前课程**）
hist_avg_quiz       该学习者历史平均测验分（**排除当前课程**，无则 0.5）
hist_log_count      ``log(1 + 历史交互条数)``
course_rating       课程口碑 ``1~5``
log_enroll          ``log(1 + 报名人数)``
domain_match        课程领域是否属于目标领域（0/1）
weekly_budget_log   ``log(1 + 每周可投入小时)``
theory_gap          ``|课程理论占比 - 学习者理论偏好| ∈ [0,1]``
modality_match      模态匹配（完全一致 1.0；含 mixed 0.6；否则 0.2）
==================== ==========================================================
"""

from __future__ import annotations

import math
from typing import Any, Final, Iterable, Mapping, Sequence

from learnpath.config import (
    MASTERY_GAP,
    MAX_LEVEL,
    MIN_LEVEL,
    MODALITY_NAMES,
)
from learnpath.data.schema import Course, Dataset, Interaction
from learnpath.knowledge.graph import KnowledgeGraph

#: 特征顺序（训练矩阵的列序，必须与 :func:`build_features` 的键顺序一致）
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "learner_ability",
    "avg_mastery",
    "target_avg_mastery",
    "readiness",
    "style_fit",
    "time_fit",
    "expected_gain_model",
    "course_level",
    "level_gap",
    "duration_hours",
    "log_duration",
    "n_kp_covered",
    "prereq_uncovered_ratio",
    "gap_coverage_ratio",
    "hist_completion_rate",
    "hist_avg_quiz",
    "hist_log_count",
    "course_rating",
    "log_enroll",
    "domain_match",
    "weekly_budget_log",
    "theory_gap",
    "modality_match",
)

#: 特征的中文名（写报告 / 前端 tooltip 用）
FEATURE_LABELS: Final[dict[str, str]] = {
    "learner_ability": "学习者综合能力",
    "avg_mastery": "目标闭包平均掌握度",
    "target_avg_mastery": "目标知识点平均掌握度",
    "readiness": "先修就绪度",
    "style_fit": "学习风格匹配度",
    "time_fit": "时间可承受度",
    "expected_gain_model": "模型预估增益",
    "course_level": "课程难度",
    "level_gap": "难度差",
    "duration_hours": "课程时长",
    "log_duration": "课时长尾压缩",
    "n_kp_covered": "覆盖知识点数",
    "prereq_uncovered_ratio": "先修未掌握比例",
    "gap_coverage_ratio": "目标相关度",
    "hist_completion_rate": "历史完成率",
    "hist_avg_quiz": "历史平均测验分",
    "hist_log_count": "历史学习量",
    "course_rating": "课程口碑",
    "log_enroll": "课程热度",
    "domain_match": "领域匹配",
    "weekly_budget_log": "每周可投入时长",
    "theory_gap": "理论与实践偏好差",
    "modality_match": "模态匹配",
}

#: 无历史数据时 `hist_completion_rate` 的先验（对应数据集整体完成率量级）
PRIOR_COMPLETION_RATE: Final[float] = 0.45
#: 无历史测验数据时 `hist_avg_quiz` 的先验
PRIOR_QUIZ_SCORE: Final[float] = 0.55
#: 贝叶斯平滑强度：历史样本数越少，越向先验收缩
HISTORY_SMOOTHING: Final[float] = 3.0


# ============================================================
# 模块级缓存（同一 dataset / graph 对象复用索引，避免全表扫描）
# ============================================================


class _HistoryIndex:
    """按学习者聚合的历史行为索引（对 ``dataset.interactions`` 一次扫描建成）。"""

    __slots__ = (
        "by_learner",
        "total",
        "completed",
        "quiz_sum",
        "quiz_n",
        "domain_level",
    )

    def __init__(self, dataset: Dataset) -> None:
        by_learner: dict[str, list[Interaction]] = {}
        total: dict[str, int] = {}
        completed: dict[str, int] = {}
        quiz_sum: dict[str, float] = {}
        quiz_n: dict[str, int] = {}
        level_acc: dict[tuple[str, str], float] = {}
        level_weight: dict[tuple[str, str], float] = {}

        courses = dataset.courses
        for record in dataset.interactions:
            learner_id = record.learner_id
            by_learner.setdefault(learner_id, []).append(record)
            total[learner_id] = total.get(learner_id, 0) + 1
            if record.completed:
                completed[learner_id] = completed.get(learner_id, 0) + 1
            if record.quiz_score is not None:
                quiz_sum[learner_id] = quiz_sum.get(learner_id, 0.0) + float(record.quiz_score)
                quiz_n[learner_id] = quiz_n.get(learner_id, 0) + 1

            course = courses.get(record.course_id)
            if course is None:
                continue
            key = (learner_id, course.domain)
            weight = max(0.05, float(record.progress))
            level_acc[key] = level_acc.get(key, 0.0) + weight * float(course.level)
            level_weight[key] = level_weight.get(key, 0.0) + weight

        self.by_learner = by_learner
        self.total = total
        self.completed = completed
        self.quiz_sum = quiz_sum
        self.quiz_n = quiz_n
        self.domain_level = {
            key: level_acc[key] / level_weight[key]
            for key in level_acc
            if level_weight[key] > 0
        }

    def domain_level_of(self, learner_id: str, domain: str, fallback: float) -> float:
        """学习者在某领域的「现有课程难度水平」（按进度加权的已学课程难度）。"""
        return self.domain_level.get((learner_id, domain), fallback)


class _GraphIndex:
    """知识图谱侧的廉价缓存：目标闭包与课程先修。"""

    __slots__ = ("graph", "closures", "course_prereq")

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self.closures: dict[tuple[str, ...], frozenset[str]] = {}
        self.course_prereq: dict[str, frozenset[str]] = {}

    def closure_of(self, kp_ids: Sequence[str]) -> frozenset[str]:
        key = tuple(kp_ids)
        cached = self.closures.get(key)
        if cached is None:
            cached = self.graph.prerequisite_closure(kp_ids)
            self.closures[key] = cached
        return cached

    def prerequisites(self, course: Course) -> frozenset[str]:
        cached = self.course_prereq.get(course.course_id)
        if cached is None:
            cached = self.graph.course_prerequisites(course)
            self.course_prereq[course.course_id] = cached
        return cached


_HISTORY_CACHE: dict[int, tuple[Dataset, _HistoryIndex]] = {}
_GRAPH_CACHE: dict[int, tuple[KnowledgeGraph, _GraphIndex]] = {}
_MAX_CACHE_ENTRIES: Final[int] = 8


def history_index(dataset: Dataset) -> _HistoryIndex:
    """取得（并缓存）``dataset`` 的历史索引。

    用 ``id(dataset)`` 作用域键，同时**持有对象强引用**，
    因此不会出现 ``id`` 复用导致的错配；超过 8 个数据集按 FIFO 淘汰。
    """
    key = id(dataset)
    entry = _HISTORY_CACHE.get(key)
    if entry is not None and entry[0] is dataset:
        return entry[1]
    index = _HistoryIndex(dataset)
    _HISTORY_CACHE[key] = (dataset, index)
    while len(_HISTORY_CACHE) > _MAX_CACHE_ENTRIES:
        _HISTORY_CACHE.pop(next(iter(_HISTORY_CACHE)))
    return index


def graph_index(graph: KnowledgeGraph) -> _GraphIndex:
    """取得（并缓存）``graph`` 的闭包缓存。"""
    key = id(graph)
    entry = _GRAPH_CACHE.get(key)
    if entry is not None and entry[0] is graph:
        return entry[1]
    index = _GraphIndex(graph)
    _GRAPH_CACHE[key] = (graph, index)
    while len(_GRAPH_CACHE) > _MAX_CACHE_ENTRIES:
        _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
    return index


# ============================================================
# 特征构造
# ============================================================


def build_features(
    dataset: Dataset,
    graph: KnowledgeGraph,
    model: Any,
    course: Course,
) -> dict[str, float]:
    """为「某学习者 × 某课程」构造 23 维特征向量。

    Args:
        dataset: 全量数据集（提供课程/目标/历史交互）。
        graph: 知识点图（提供先修闭包）。
        model: :class:`~learnpath.learner.profile.LearnerModel`。
            ⚠️ **训练时必须传入「排除了当前课程」的 model**，否则会标签泄漏。
        course: 待评估的课程。

    Returns:
        ``{特征名: float}``，键集合恒等于 :data:`FEATURE_NAMES`。

    Note:
        本函数不抛异常、不做 I/O、不含随机数；``hist_*`` 特征用贝叶斯平滑
        （``HISTORY_SMOOTHING = 3``）向先验收缩，避免「只学过 1 门课」的新
        学习者产生极端值。
    """
    learner = getattr(model, "learner", None)
    if learner is None:  # pragma: no cover - 契约保证存在
        raise AttributeError("LearnerModel 缺少公开属性 learner")

    learner_id = str(learner.learner_id)
    goal = getattr(model, "goal", None)
    if goal is None:
        goal = dataset.goals[learner.goal_id]
    target_kps: tuple[str, ...] = tuple(getattr(model, "target_kps", goal.target_kp_ids))

    gidx = graph_index(graph)
    hidx = history_index(dataset)

    closure = gidx.closure_of(target_kps)

    # ---- 掌握度类（3 个） ----
    closure_mastery = [float(model.mastery_of(kp_id)) for kp_id in closure]
    avg_mastery = sum(closure_mastery) / len(closure_mastery) if closure_mastery else 0.0
    target_mastery = [float(model.mastery_of(kp_id)) for kp_id in target_kps]
    target_avg = sum(target_mastery) / len(target_mastery) if target_mastery else 0.0

    # ---- 就绪度 / 匹配度（3 个） ----
    readiness = _clip01(model.readiness_for(course))
    style_fit = _clip01(model.style_fit(course))
    time_fit = _clip01(model.time_fit(course))
    expected_gain = max(0.0, _safe_float(model.expected_gain(course), 0.0))

    # ---- 课程属性（5 个） ----
    ability = _clip01(getattr(learner, "ability", 0.5))
    span = float(MAX_LEVEL - MIN_LEVEL)
    fallback_level = MIN_LEVEL + span * ability
    domain_level = hidx.domain_level_of(learner_id, course.domain, fallback_level)
    level_gap = (float(course.level) - domain_level) / span if span > 0 else 0.0
    duration = float(course.duration_hours)

    # ---- 覆盖 / 先修（3 个） ----
    own_kps = set(course.kp_ids)
    prereq = gidx.prerequisites(course)
    if prereq:
        uncovered = sum(1 for kp_id in prereq if float(model.mastery_of(kp_id)) < MASTERY_GAP)
        prereq_uncovered_ratio = uncovered / len(prereq)
    else:
        prereq_uncovered_ratio = 0.0
    gap_coverage_ratio = len(own_kps & closure) / len(own_kps) if own_kps else 0.0

    # ---- 历史行为（3 个，显式排除当前课程） ----
    records = hidx.by_learner.get(learner_id, ())
    n_total = 0
    n_completed = 0
    quiz_sum = 0.0
    quiz_n = 0
    for record in records:
        if record.course_id == course.course_id:
            continue  # ← 防标签泄漏：绝不使用该学习者在本课程上的任何结果
        n_total += 1
        if record.completed:
            n_completed += 1
        if record.quiz_score is not None:
            quiz_sum += float(record.quiz_score)
            quiz_n += 1
    hist_completion = (n_completed + HISTORY_SMOOTHING * PRIOR_COMPLETION_RATE) / (
        n_total + HISTORY_SMOOTHING
    )
    hist_quiz = (quiz_sum + HISTORY_SMOOTHING * PRIOR_QUIZ_SCORE) / (quiz_n + HISTORY_SMOOTHING)

    # ---- 课程口碑 / 热度（2 个） ----
    rating = float(course.rating)
    log_enroll = math.log1p(max(0, int(course.enroll_count)))

    # ---- 画像匹配（4 个） ----
    target_domains = {graph.domain_of(kp_id) for kp_id in target_kps if kp_id in graph}
    domain_match = 1.0 if course.domain in target_domains else 0.0
    weekly = max(0.1, float(getattr(learner, "time_budget_hours_per_week", 6.0) or 6.0))
    weekly_log = math.log1p(weekly)
    style = getattr(learner, "learning_style", None)
    theory_pref = float(getattr(style, "theory_preference", 0.5) or 0.5)
    theory_gap = abs(float(course.theory_ratio) - theory_pref)
    prefer = str(getattr(style, "prefer_modality", "video") or "video")
    modality_match = _modality_match(prefer, course.modality)

    return {
        "learner_ability": ability,
        "avg_mastery": avg_mastery,
        "target_avg_mastery": target_avg,
        "readiness": readiness,
        "style_fit": style_fit,
        "time_fit": time_fit,
        "expected_gain_model": expected_gain,
        "course_level": float(course.level),
        "level_gap": float(level_gap),
        "duration_hours": duration,
        "log_duration": math.log1p(duration),
        "n_kp_covered": float(len(course.kp_coverage)),
        "prereq_uncovered_ratio": float(prereq_uncovered_ratio),
        "gap_coverage_ratio": float(gap_coverage_ratio),
        "hist_completion_rate": float(hist_completion),
        "hist_avg_quiz": float(hist_quiz),
        "hist_log_count": math.log1p(n_total),
        "course_rating": rating,
        "log_enroll": log_enroll,
        "domain_match": domain_match,
        "weekly_budget_log": weekly_log,
        "theory_gap": float(theory_gap),
        "modality_match": float(modality_match),
    }


def feature_vector(features: Mapping[str, float]) -> list[float]:
    """按 :data:`FEATURE_NAMES` 的固定顺序把特征字典拍平成列表（构造训练矩阵用）。"""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def _modality_match(prefer: str, modality: str) -> float:
    """模态匹配度：完全一致 1.0；任一方为混合式 0.6；否则 0.2。"""
    if prefer == modality:
        return 1.0
    if "mixed" in (prefer, modality):
        return 0.6
    return 0.2


def _clip01(value: Any) -> float:
    number = _safe_float(value, 0.0)
    return 0.0 if number < 0.0 else (1.0 if number > 1.0 else number)


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def modality_name(modality: str) -> str:
    """模态 key → 中文名。"""
    return MODALITY_NAMES.get(modality, modality)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_LABELS",
    "build_features",
    "feature_vector",
    "history_index",
    "graph_index",
    "modality_name",
    "PRIOR_COMPLETION_RATE",
    "PRIOR_QUIZ_SCORE",
    "HISTORY_SMOOTHING",
]
