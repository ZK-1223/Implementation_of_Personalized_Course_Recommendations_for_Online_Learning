"""混合推荐（Hybrid Recommender）：多路召回 → 加权融合 → 约束过滤 → MMR 多样性重排 → 可解释理由。

对应报告「推荐算法」章节的系统核心，也是「实现要求 ①」的最终交付形态。

--------------------------------------------------------------------------------
一、整体流程（五步流水线）
--------------------------------------------------------------------------------
::

     ┌──────────── 多路召回 Recall ────────────┐
     │ ① 内容召回（ContentRecommender 全量打分）│
     │ ② 协同召回（CF-SVD 预测分 Top-N）        │
     │ ③ 口碑召回（贝叶斯平滑评分 + 热度 Top-N）│
     └───────────────┬────────────────────────┘
                     ▼ 候选并集（≈ k × RECOMMEND_FETCH_MULTIPLIER）
     ┌──────────── 约束过滤 Filter ────────────┐
     │ ① 目标先修闭包（核心约束，绝不放宽）      │
     │ ② 已学过（exclude_enrolled）             │
     │ ③ 先修未就绪（readiness < 阈值）          │
     └───────────────┬────────────────────────┘
                     ▼
     ┌──────────── 加权融合 Fusion ────────────┐
     │ score = Σ w_m · s_m   （RECOMMEND_WEIGHTS = 0.40/0.25/0.15/0.12/0.08）│
     └───────────────┬────────────────────────┘
                     ▼
     ┌──────────── MMR 多样性重排 ─────────────┐
     │ MMR = λ·rel(d) - (1-λ)·max_{s∈S} sim(d,s)，λ = MMR_LAMBDA = 0.75      │
     └───────────────┬────────────────────────┘
                     ▼
                 Top-K + 可解释理由

### 五路信号的语义
===============  =======  ============================================================
分量             权重      含义与来源
===============  =======  ============================================================
``content``       0.40     知识图谱内容匹配（目标缺口覆盖 ÷ 时间成本 × 最近发展区 × 先修门控）
``collaborative`` 0.25     CF-SVD 隐语义预测分（相似学习者的群体行为）
``quality``       0.15     贝叶斯平滑评分（0.72）+ 对数热度（0.28）
``readiness``     0.12     先修就绪度（sigmoid 映射后的先修平均掌握度）
``social``        0.08     相似同伴的贝叶斯平滑完成率
===============  =======  ============================================================

``content`` 权重最高是**有意设计**：本系统的核心主张是「知识图谱约束下的个性化」，
纯群体行为（CF）无法处理冷启动，也无法保证「先修 → 后继」的学习顺序；但 CF 权重
0.25 + social 0.08 = 0.33 的群体信号保证了推荐结果不局限于「知识点覆盖最大化」而
错过学习者真实喜欢的高口碑课程。

### 为什么用 MMR 而不是简单去重
学习者如果连看 5 门都讲「数据清洗」的课，收益是重复的。MMR（Maximal Marginal
Relevance）在「相关性」与「与已选集合的最大相似度」之间做权衡::

    MMR(d) = λ · relevance(d) - (1 - λ) · max_{s ∈ selected} sim(d, s)

λ = 0.75 表示「相关性仍占主导，但每多一门与已选高度相似的课就扣 0.25 分的相似度」。
相似度是课程特征向量（知识点覆盖强度 + 领域 + 难度 + 模态）的加权余弦：
覆盖同一知识点的两门课相似度 ≈ 0.99，同领域但知识点不同的两门课 ≈ 0.05，
因此 MMR 主要惩罚「同一知识点的重复课程」，几乎不影响跨领域扩展。

### 目标先修闭包约束（本系统的核心立场）
本系统不是通用课程商城，而是**为特定职业目标规划一条可达路径**的引擎，因此主推荐列表
被硬性限定在 ``graph.prerequisite_closure(goal.target_kp_ids)`` 内：

* 判定口径：课程覆盖的**全部**知识点都落在目标先修闭包内才算「闭包内课程」；
* 闭包内可推荐课程不足 ``k`` 条时，**返回少于 k 条**，绝不用闭包外课程凑数——
  「推荐了 3 条与目标无关的高口碑课」比「只推荐 3 条目标课」更伤害系统可信度；
* ``Recommendation.in_goal_closure`` 字段保留该信息，前端可据此区分主推荐与拓展推荐。

该约束在消融实验中被显式关闭，用于量化「知识图谱约束」带来的排序质量增量。

--------------------------------------------------------------------------------
二、可解释推荐理由
--------------------------------------------------------------------------------
推荐理由必须**具体到可验证的数字**，否则无法取信于学习者，也无法用于答辩演示。
本模块为每条推荐生成至多 5 条理由，覆盖：

1. **目标缺口**：具体知识点名 + 当前掌握度 + 距目标的差距数值；
2. **先修就绪**：已达标先修数 / 总先修数 + 平均掌握度，未达标时点名应补的知识点；
3. **相似同伴**：背景相近学员的人数与完成率；
4. **时间 / 风格匹配**：课程时长占每周预算的比例 + 模态与偏好的匹配度；
5. **口碑数据**：评分、在学人数、与同领域均分的差值。

严禁「为你推荐」「值得一学」这类空洞话术——每条理由都必须携带数值或实体名称。

--------------------------------------------------------------------------------
三、与效果预测模块的解耦
--------------------------------------------------------------------------------
``p_completion`` 字段来自队友模块 :mod:`learnpath.prediction.effect` 的
``EffectPredictor``。该模块可能尚未就绪，因此本类的依赖方式是**运行期可空**：

* 构造参数 ``effect_predictor=None``（默认）时完全不导入对方模块；
* 若传入预测器，则每次调用都被 ``try / except Exception`` 包裹，任何异常都退化为
  ``p_completion = None``（前端据此隐藏该字段），**绝不因预测模块故障而让推荐失败**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterable, Sequence

import numpy as np

from learnpath.config import (
    CF_SVD_COMPONENTS,
    MMR_LAMBDA,
    MODALITY_NAMES,
    PEER_TOP_K,
    RECOMMEND_FETCH_MULTIPLIER,
    RECOMMEND_WEIGHTS,
    RecommendWeights,
)
from learnpath.data.schema import Course, Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import (
    LearnerModel,
    cached_course_prerequisites,
    interactions_by_learner,
)
from learnpath.recommend.collaborative import CollaborativeRecommender
from learnpath.recommend.content import ContentRecommender

__all__ = [
    "Recommendation",
    "HybridRecommender",
    "mmr_rerank",
    "course_similarity",
    "build_course_features",
    "COMPONENT_KEYS",
]

#: ``Recommendation.components`` 的键（顺序固定，供前端与文档引用）。
COMPONENT_KEYS: Final[tuple[str, ...]] = (
    "content",
    "collaborative",
    "quality",
    "readiness",
    "social",
)

# ---------------------------------------------------------------------------
# 课程相似度特征权重（加权余弦，用于 MMR 与评测多样性）
# ---------------------------------------------------------------------------
#: 知识点覆盖强度块的权重（占主导，因为「讲同一个知识点」是重复的本源）。
FEATURE_KP_WEIGHT: Final[float] = 1.00
#: 领域块权重：同领域课程天然相关（约贡献 0.04 的余弦相似度）。
FEATURE_DOMAIN_WEIGHT: Final[float] = 0.20
#: 难度块权重：难度越接近越相似。
FEATURE_LEVEL_WEIGHT: Final[float] = 0.15
#: 模态块权重。
FEATURE_MODALITY_WEIGHT: Final[float] = 0.05

# ---------------------------------------------------------------------------
# 口碑（quality）分量参数
# ---------------------------------------------------------------------------
#: 贝叶斯平滑的虚拟样本量：新课程（enroll 少）向先验均值收缩。
RATING_SMOOTHING: Final[float] = 300.0
#: 评分全局先验均值（5 分制）。
RATING_PRIOR: Final[float] = 4.2
#: 评分与热度的融合权重。
QUALITY_RATING_WEIGHT: Final[float] = 0.72
QUALITY_POPULARITY_WEIGHT: Final[float] = 0.28

# ---------------------------------------------------------------------------
# 社交（social）分量参数
# ---------------------------------------------------------------------------
#: 计算同伴完成率时使用的相似学习者数量（比 PEER_TOP_K 略多，保证样本量）。
SOCIAL_PEER_K: Final[int] = max(int(PEER_TOP_K), 8)
#: 同伴完成率的贝叶斯收缩强度：有效样本量 = 相似度之和，收缩系数 = w/(w+该值)。
SOCIAL_SHRINK: Final[float] = 1.5

# ---------------------------------------------------------------------------
# 过滤参数
# ---------------------------------------------------------------------------
#: 先修就绪度低于该值的课程视为「先修不满足」被过滤（0.30 ≈ 先修平均掌握度 0.36）。
PREREQ_FILTER_THRESHOLD: Final[float] = 0.30


# ===========================================================================
# 课程特征与相似度
# ===========================================================================

_FEATURE_CACHE: dict[int, tuple[Dataset, tuple[str, ...], np.ndarray]] = {}


def build_course_features(dataset: Dataset) -> tuple[tuple[str, ...], np.ndarray]:
    """构造课程特征矩阵（行已做 L2 归一化），用于余弦相似度与 MMR 去冗余。

    特征布局（共 ``n_kps + n_domains + 1 + 4`` 维）::

        [ 知识点覆盖强度块(1.00) | 领域 one-hot(0.20) | 难度(0.15) | 模态 one-hot(0.05) ]

    L2 归一化后行向量两两点积即余弦相似度，可直接用 BLAS 批量计算。

    Args:
        dataset: 数据集。

    Returns:
        ``(course_ids, matrix)``，``matrix`` 形状为 ``(n_courses, d)``。
    """
    key = id(dataset)
    cached = _FEATURE_CACHE.get(key)
    if cached is not None and cached[0] is dataset:
        return cached[1], cached[2]

    course_ids: tuple[str, ...] = tuple(dataset.courses)
    kp_ids: tuple[str, ...] = tuple(dataset.knowledge_points)
    kp_index = {kp: i for i, kp in enumerate(kp_ids)}
    domains = tuple(sorted({kp.domain for kp in dataset.knowledge_points.values()}))
    domain_index = {d: i for i, d in enumerate(domains)}
    modalities = tuple(MODALITY_NAMES)
    modality_index = {m: i for i, m in enumerate(modalities)}

    n_kp, n_dom, n_mod = len(kp_ids), len(domains), len(modalities)
    dim = n_kp + n_dom + 1 + n_mod
    matrix = np.zeros((len(course_ids), dim), dtype=np.float64)

    dom_offset = n_kp
    level_offset = n_kp + n_dom
    mod_offset = level_offset + 1

    for pos, course_id in enumerate(course_ids):
        course = dataset.courses[course_id]
        for cov in course.kp_coverage:
            index = kp_index.get(cov.kp_id)
            if index is not None:
                matrix[pos, index] = FEATURE_KP_WEIGHT * float(cov.weight)
        matrix[pos, dom_offset + domain_index[course.domain]] = FEATURE_DOMAIN_WEIGHT
        matrix[pos, level_offset] = FEATURE_LEVEL_WEIGHT * (int(course.level) - 1) / 4.0
        mod_index = modality_index.get(course.modality)
        if mod_index is not None:
            matrix[pos, mod_offset + mod_index] = FEATURE_MODALITY_WEIGHT

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    _FEATURE_CACHE[key] = (dataset, course_ids, matrix)
    return course_ids, matrix


def course_similarity(course_a: Course, course_b: Course) -> float:
    """两门课程的加权余弦相似度 ∈ [0, 1]（与 :func:`build_course_features` 同一公式）。

    用于 MMR 去冗余与评测中的「品类内平均不相似度」（diversity）指标。
    """
    if course_a.course_id == course_b.course_id:
        return 1.0

    weights_a = course_a.kp_weights
    weights_b = course_b.kp_weights
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for kp_id, weight in weights_a.items():
        value = FEATURE_KP_WEIGHT * weight
        norm_a += value * value
        other = weights_b.get(kp_id)
        if other is not None:
            dot += value * (FEATURE_KP_WEIGHT * other)
    for weight in weights_b.values():
        value = FEATURE_KP_WEIGHT * weight
        norm_b += value * value

    dom = FEATURE_DOMAIN_WEIGHT * FEATURE_DOMAIN_WEIGHT
    norm_a += dom
    norm_b += dom
    if course_a.domain == course_b.domain:
        dot += dom

    level_a = FEATURE_LEVEL_WEIGHT * (int(course_a.level) - 1) / 4.0
    level_b = FEATURE_LEVEL_WEIGHT * (int(course_b.level) - 1) / 4.0
    dot += level_a * level_b
    norm_a += level_a * level_a
    norm_b += level_b * level_b

    mod = FEATURE_MODALITY_WEIGHT * FEATURE_MODALITY_WEIGHT
    norm_a += mod
    norm_b += mod
    if course_a.modality == course_b.modality:
        dot += mod

    denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denominator <= 1e-12:
        return 0.0
    return float(min(1.0, max(0.0, dot / denominator)))


# ===========================================================================
# 推荐结果
# ===========================================================================


@dataclass(slots=True)
class Recommendation:
    """一条推荐结果（对应 ``docs/API.md`` 第 3 节的 ``items`` 元素）。

    Attributes:
        course: 被推荐的课程。
        score: 融合后的总分 ∈ [0, 1]（五路信号的加权和）。
        components: 分项得分 ``content / collaborative / quality / readiness / social``。
        reasons: 中文可解释理由（每条都带数值或实体名称）。
        expected_gain: 预期掌握度提升总量（见 :meth:`LearnerModel.expected_gain`）。
        p_completion: 完成概率 ∈ [0, 1]；效果预测模块不可用时为 None（前端隐藏该字段）。
        readiness: 先修就绪度（冗余自 components，方便前端直接取用）。
        style_fit: 学习风格匹配度。
        time_fit: 时间可承受度。
        risk_level: 风险等级「低 / 中 / 高」，随 p_completion 一起由预测模块给出。
        kp_names: 课程覆盖的知识点名称（便于前端直接展示）。
        in_goal_closure: 课程是否完全落在学习者目标的先修闭包内。主推荐列表恒为 True；
            若前端需要「跨领域拓展」推荐，可用该字段区分（本模块默认不返回 False 的条目）。
    """

    course: Course
    score: float
    components: dict[str, float]
    reasons: list[str]
    expected_gain: float
    p_completion: float | None = None
    readiness: float = 0.0
    style_fit: float = 0.0
    time_fit: float = 0.0
    risk_level: str | None = None
    kp_names: list[str] = field(default_factory=list)
    in_goal_closure: bool = True

    @property
    def course_id(self) -> str:
        """课程 id（便捷属性）。"""
        return self.course.course_id

    def to_dict(self, rank: int | None = None) -> dict[str, Any]:
        """转为 API 可直接返回的字典。"""
        payload: dict[str, Any] = {
            "course_id": self.course.course_id,
            "title": self.course.title,
            "domain": self.course.domain,
            "level": int(self.course.level),
            "duration_hours": round(float(self.course.duration_hours), 2),
            "modality": self.course.modality,
            "modality_name": MODALITY_NAMES.get(self.course.modality, self.course.modality),
            "rating": round(float(self.course.rating), 2),
            "enroll_count": int(self.course.enroll_count),
            "price": round(float(self.course.price), 2),
            "kp_names": list(self.kp_names),
            "score": round(float(self.score), 3),
            "components": {k: round(float(v), 3) for k, v in self.components.items()},
            "reasons": list(self.reasons),
            "expected_gain": round(float(self.expected_gain), 3),
            "p_completion": None if self.p_completion is None else round(float(self.p_completion), 3),
            "risk_level": self.risk_level,
            "readiness": round(float(self.readiness), 3),
            "style_fit": round(float(self.style_fit), 3),
            "time_fit": round(float(self.time_fit), 3),
            "in_goal_closure": bool(self.in_goal_closure),
        }
        if rank is not None:
            payload = {"rank": int(rank), **payload}
        return payload


# ===========================================================================
# MMR 多样性重排
# ===========================================================================


def _course_of(item: Any) -> Course:
    """从候选对象中取出课程（兼容 ``Recommendation`` 与 ``(Course, score)`` 元组）。"""
    course = getattr(item, "course", None)
    if course is not None:
        return course
    return item[0]


def _relevance_of(item: Any) -> float:
    """从候选对象中取出相关性分数。"""
    score = getattr(item, "score", None)
    if score is not None:
        return float(score)
    return float(item[1])


def mmr_rerank(
    candidates: Sequence[Any],
    k: int,
    lambda_: float = MMR_LAMBDA,
    similarity: Callable[[Course, Course], float] | None = None,
) -> list[Any]:
    """最大边际相关性（MMR）重排，兼顾相关性与多样性。

    贪心算法：每轮从未选集合中挑选使下式最大的候选::

        MMR(d) = λ · relevance(d) - (1 - λ) · max_{s ∈ 已选} sim(d, s)

    Args:
        candidates: 候选列表，元素为 :class:`Recommendation` 或 ``(Course, score)`` 元组，
            须已按相关性排序（或至少携带相关性分数）。
        k: 需要返回的数量。
        lambda_: 相关性权重 λ ∈ [0, 1]。1.0 表示纯相关性排序（等价于关闭 MMR），
            0.0 表示纯多样性；默认取 :data:`learnpath.config.MMR_LAMBDA`（0.75）。
        similarity: 自定义相似度函数；默认使用 :func:`course_similarity`。

    Returns:
        重排后的候选列表，长度 ``min(k, len(candidates))``。
    """
    if k <= 0 or not candidates:
        return []

    sim_fn = similarity or course_similarity
    lam = float(lambda_)
    lam = min(1.0, max(0.0, lam))

    pool: list[Any] = list(candidates)
    selected: list[Any] = []
    sim_cache: dict[tuple[str, str], float] = {}
    limit = min(int(k), len(pool))

    while len(selected) < limit and pool:
        best_index = 0
        best_value = -math.inf
        for index, candidate in enumerate(pool):
            course = _course_of(candidate)
            relevance = _relevance_of(candidate)
            if selected:
                max_sim = 0.0
                for chosen in selected:
                    chosen_course = _course_of(chosen)
                    key = (course.course_id, chosen_course.course_id)
                    value = sim_cache.get(key)
                    if value is None:
                        value = sim_fn(course, chosen_course)
                        sim_cache[key] = value
                    if value > max_sim:
                        max_sim = value
            else:
                max_sim = 0.0
            value = lam * relevance - (1.0 - lam) * max_sim
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(pool.pop(best_index))
    return selected


# ===========================================================================
# 混合推荐器
# ===========================================================================


class HybridRecommender:
    """多路召回 + 加权融合 + MMR 重排 + 可解释理由的混合推荐器。

    Args:
        dataset: 数据集。
        graph: 知识图谱。
        effect_predictor: 可选的完成概率 / 掌握度增益预测器（队友模块
            :class:`learnpath.prediction.effect.EffectPredictor`）。传 None 或该模块
            不可用时，``Recommendation.p_completion`` 为 None，其余功能完全正常。
        weights: 五路信号的融合权重，默认 :data:`learnpath.config.RECOMMEND_WEIGHTS`。
            主要用于消融实验（例如把 ``collaborative`` 置 0）。
        use_prerequisite_filter: 是否启用「先修未就绪」硬过滤（消融开关）。
        mmr_lambda: MMR 的相关性权重，默认 :data:`learnpath.config.MMR_LAMBDA`；
            置 1.0 等价于关闭多样性重排（消融开关）。
        prereq_threshold: 先修就绪度过滤阈值。
        only_goal_closure: 是否把推荐候选**限定在目标先修闭包内**（默认 True）。
            这是本系统的核心约束：推荐必须服务于「达成目标的一条可达路径」，
            闭包外的课程再高口碑也不进入主推荐列表（闭包内不足 k 条时返回少于 k 条，
            绝不用闭包外课程凑数）。消融实验会把它关掉以量化该约束的价值。
        content_recommender: 可注入的内容推荐器（复用缓存，评测时避免重复建表）。
        collaborative_recommender: 可注入的协同过滤器（复用已拟合的 SVD）。

    Example:
        >>> from learnpath.data.fixtures import build_tiny_dataset
        >>> from learnpath.knowledge.graph import KnowledgeGraph
        >>> from learnpath.learner.profile import LearnerModel
        >>> ds = build_tiny_dataset()
        >>> graph = KnowledgeGraph.from_dataset(ds)
        >>> engine = HybridRecommender(ds, graph)
        >>> model = LearnerModel(ds.learners["L-0001"], ds, graph)
        >>> recs = engine.recommend(model, k=5)
        >>> len(recs) <= 5 and all(r.reasons for r in recs)
        True
    """

    __slots__ = (
        "dataset",
        "graph",
        "effect_predictor",
        "weights",
        "use_prerequisite_filter",
        "only_goal_closure",
        "mmr_lambda",
        "prereq_threshold",
        "content",
        "collab",
        "course_ids",
        "_course_pos",
        "_goal_course_mask",
        "_quality",
        "_enroll_log",
        "_domain_rating",
        "_completion_prior",
        "_population_rate",
        "_features",
        "_social_cache",
        "_cf_ready",
    )

    def __init__(
        self,
        dataset: Dataset,
        graph: KnowledgeGraph,
        effect_predictor: Any = None,
        *,
        weights: RecommendWeights | None = None,
        use_prerequisite_filter: bool = True,
        mmr_lambda: float = MMR_LAMBDA,
        prereq_threshold: float = PREREQ_FILTER_THRESHOLD,
        only_goal_closure: bool = True,
        content_recommender: ContentRecommender | None = None,
        collaborative_recommender: CollaborativeRecommender | None = None,
    ) -> None:
        self.dataset: Dataset = dataset
        self.graph: KnowledgeGraph = graph
        self.effect_predictor = effect_predictor
        self.weights: RecommendWeights = weights or RECOMMEND_WEIGHTS
        self.use_prerequisite_filter = bool(use_prerequisite_filter)
        self.only_goal_closure = bool(only_goal_closure)
        self.mmr_lambda = float(mmr_lambda)
        self.prereq_threshold = float(prereq_threshold)

        self.course_ids: tuple[str, ...] = tuple(dataset.courses)
        self._course_pos: dict[str, int] = {cid: i for i, cid in enumerate(self.course_ids)}
        self._goal_course_mask: dict[str, np.ndarray] = self._build_goal_course_mask()

        self.content = content_recommender or ContentRecommender(dataset, graph)
        self.collab = collaborative_recommender or CollaborativeRecommender(
            dataset, CF_SVD_COMPONENTS
        )
        self._cf_ready = collaborative_recommender is not None and getattr(
            collaborative_recommender, "fitted", False
        )

        _, self._features = build_course_features(dataset)
        self._quality, self._enroll_log, self._domain_rating = self._build_quality()
        self._completion_prior, self._population_rate = self._build_completion_prior()
        self._social_cache: tuple[
            int, int, dict[str, tuple[int, int, float, float]], list[tuple[str, float]]
        ] | None = None

    # ==================================================================
    # 预热
    # ==================================================================
    def _build_goal_course_mask(self) -> dict[str, np.ndarray]:
        """为每个目标预计算「课程是否完全落在其先修闭包内」的布尔掩码。

        判定口径：课程覆盖的**全部**知识点都在目标先修闭包
        ``graph.prerequisite_closure(goal.target_kp_ids)`` 内。这样推荐列表天然只包含
        「为达成该目标服务」的课程，跨领域的高口碑课程不会混入主推荐。
        """
        masks: dict[str, np.ndarray] = {}
        course_kps = {cid: self.dataset.courses[cid].kp_ids for cid in self.course_ids}
        for goal_id, goal in self.dataset.goals.items():
            closure = self.graph.prerequisite_closure(goal.target_kp_ids)
            masks[goal_id] = np.array(
                [all(kp in closure for kp in course_kps[cid]) for cid in self.course_ids],
                dtype=bool,
            )
        return masks

    def goal_courses(self, learner_id: str) -> list[str]:
        """返回落在该学习者**目标先修闭包内**的全部课程 id（按课程 id 排序）。

        供 API / 前端做「目标相关课程」筛选，也便于验证推荐是否越界。
        """
        goal_id = self.dataset.learners[learner_id].goal_id
        mask = self._goal_course_mask.get(goal_id)
        if mask is None:
            return list(self.course_ids)
        return [cid for cid, keep in zip(self.course_ids, mask) if keep]

    def warm_up(self) -> "HybridRecommender":
        """提前完成 CF 的 SVD 拟合，把耗时移出首个请求（可选）。"""
        if not self._cf_ready:
            self.collab.fit()
            self._cf_ready = True
        return self

    def _ensure_cf(self) -> None:
        if not self._cf_ready:
            self.collab.fit()
            self._cf_ready = True

    # ==================================================================
    # 主流程
    # ==================================================================
    def recommend(
        self,
        model: LearnerModel,
        k: int = 10,
        exclude_enrolled: bool = True,
    ) -> list[Recommendation]:
        """为学习者生成 Top-K 个性化推荐。

        推荐口径（重要）：默认 ``only_goal_closure=True``，候选集被限定在目标先修闭包
        ``graph.prerequisite_closure(goal.target_kp_ids)`` 内的课程，因此返回的每一条都
        ``in_goal_closure = True``，直接服务于「达成目标的可达路径」。若闭包内可推荐课程
        不足 ``k`` 条，**返回条数会少于 k**，绝不用闭包外的高口碑课程凑数。

        Args:
            model: 学习者模型（掌握度状态来自它，:meth:`LearnerModel.apply_outcome`
                之后重新调用即可得到更新后的推荐）。
            k: 推荐条数上限。
            exclude_enrolled: 是否排除该学习者已产生交互的课程。

        Returns:
            :class:`Recommendation` 列表，按 MMR 重排后的最终顺序，长度
            ``min(k, 闭包内可推荐课程数)``。
        """
        if k <= 0:
            return []
        self._ensure_cf()

        n_courses = len(self.course_ids)
        content_scores = self.content.score_all(model)
        readiness_scores = self.content.readiness_all(model)
        cf_scores = self.collab.score_all(model.learner.learner_id)
        quality_scores = self._quality

        # ---- 1) 约束过滤 ----
        # ① 目标先修闭包（核心约束，永不放宽）
        base_pool = np.ones(n_courses, dtype=bool)
        if self.only_goal_closure:
            mask = self._goal_course_mask.get(model.learner.goal_id)
            if mask is not None and mask.any():
                base_pool = mask
        # ② 已学过
        pool = base_pool.copy()
        if exclude_enrolled:
            for course_id in model.courses_covered():
                position = self._course_pos.get(course_id)
                if position is not None:
                    pool[position] = False
        # ③ 先修未就绪（软约束，可用消融开关关闭）
        if self.use_prerequisite_filter:
            filtered = pool & (readiness_scores >= self.prereq_threshold)
            if int(filtered.sum()) >= k:
                pool = filtered
            elif int(pool.sum()) == 0:
                pool = filtered  # 先修过滤后为空时，退一步只看闭包内（仍不越界）
        if int(pool.sum()) == 0:
            return []

        # ---- 2) 多路召回 ----
        candidates = self._recall(pool, content_scores, cf_scores, quality_scores, k)
        if not candidates:
            return []

        # ---- 3) 五路信号打分 ----
        cand_positions = np.asarray(candidates, dtype=np.intp)
        social = self._social_scores(model, cand_positions)
        components = {
            "content": content_scores[cand_positions],
            "collaborative": cf_scores[cand_positions],
            "quality": quality_scores[cand_positions],
            "readiness": readiness_scores[cand_positions],
            "social": social,
        }
        weights = self.weights
        weight_vector = np.array(
            [weights.content, weights.collaborative, weights.quality,
             weights.readiness, weights.social],
            dtype=np.float64,
        )
        if weight_vector.sum() > 0:
            weight_vector = weight_vector / weight_vector.sum()
        fused = np.zeros(len(cand_positions), dtype=np.float64)
        for column, key in enumerate(COMPONENT_KEYS):
            fused += weight_vector[column] * components[key]
        fused = np.clip(fused, 0.0, 1.0)

        order = np.argsort(-fused, kind="stable")
        candidates = [candidates[int(i)] for i in order]
        fused = fused[order]
        components = {key: values[order] for key, values in components.items()}

        # ---- 4) MMR 多样性重排 ----
        scored = [(self.dataset.courses[self.course_ids[pos]], float(score))
                  for pos, score in zip(candidates, fused)]
        feature_rows = self._features[np.asarray(candidates, dtype=np.intp)]
        pair_sim = feature_rows @ feature_rows.T
        pos_in_candidates = {pos: i for i, pos in enumerate(candidates)}

        def _similarity(course_a: Course, course_b: Course) -> float:
            ia = pos_in_candidates.get(self._course_pos.get(course_a.course_id, -1))
            ib = pos_in_candidates.get(self._course_pos.get(course_b.course_id, -1))
            if ia is None or ib is None:
                return course_similarity(course_a, course_b)
            return float(pair_sim[ia, ib])

        reranked = mmr_rerank(scored, k, self.mmr_lambda, similarity=_similarity)

        # ---- 5) 组装推荐结果（含可解释理由）----
        index_of = {pos: i for i, pos in enumerate(candidates)}
        results: list[Recommendation] = []
        peer_stats, _ = self._social_context(model)
        for course, score in reranked:
            position = self._course_pos[course.course_id]
            column = index_of[position]
            component_values = {
                key: float(components[key][column]) for key in COMPONENT_KEYS
            }
            results.append(self._build_recommendation(model, course, score, component_values, peer_stats))
        return results

    def explain(self, model: LearnerModel, recommendation: Recommendation) -> list[str]:
        """重新生成某条推荐的可解释理由（按当前掌握度状态，可用于前端「为什么推荐我」）。

        Args:
            model: 学习者模型。
            recommendation: 待解释的推荐结果。

        Returns:
            中文理由列表；理由中的数值均为**当前**状态重新计算所得。
        """
        course = recommendation.course
        position = self._course_pos.get(course.course_id)
        peer_stats, _ = self._social_context(model)
        if position is None:
            return list(recommendation.reasons)
        component_values = {
            "content": self.content.score(model, course),
            "collaborative": self.collab.score(model.learner.learner_id, course.course_id),
            "quality": float(self._quality[position]),
            "readiness": model.readiness_for(course),
            "social": float(self._social_scores(model, np.array([position]))[0]),
        }
        return self._build_reasons(model, course, component_values, peer_stats)

    # ==================================================================
    # 召回
    # ==================================================================
    def _recall(
        self,
        pool: np.ndarray,
        content_scores: np.ndarray,
        cf_scores: np.ndarray,
        quality_scores: np.ndarray,
        k: int,
    ) -> list[int]:
        """三路召回取并集：内容 / 协同 / 口碑，每路取 ``k * RECOMMEND_FETCH_MULTIPLIER``。"""
        indices = np.flatnonzero(pool)
        if indices.size == 0:
            return []
        fetch = max(int(k) * int(RECOMMEND_FETCH_MULTIPLIER), int(k))
        if fetch >= indices.size:
            return [int(i) for i in indices]

        selected: set[int] = set()
        for source in (content_scores, cf_scores, quality_scores):
            values = source[indices]
            top = np.argpartition(-values, fetch - 1)[:fetch]
            selected.update(int(indices[i]) for i in top)
        return sorted(selected)

    # ==================================================================
    # 五路信号
    # ==================================================================
    def _build_quality(self) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """口碑分量：贝叶斯平滑评分（0.72）+ 对数热度（0.28）。

        贝叶斯平滑（IMDb 加权评分公式）::

            bayes = (rating * enroll + C * prior) / (enroll + C)      # C = 300, prior = 4.2

        作用是压制「3 人打了 5 分」的虚高评分，让样本量大的课程更可信；热度项取
        ``log1p(enroll) / log1p(max_enroll)`` 做 [0,1] 归一，避免头部课程绝对碾压。
        """
        n_courses = len(self.course_ids)
        quality = np.zeros(n_courses, dtype=np.float64)
        enroll_log = np.zeros(n_courses, dtype=np.float64)
        ratings = np.zeros(n_courses, dtype=np.float64)
        domains = np.empty(n_courses, dtype=object)

        for position, course_id in enumerate(self.course_ids):
            course = self.dataset.courses[course_id]
            ratings[position] = float(course.rating)
            enroll_log[position] = math.log1p(max(int(course.enroll_count), 0))
            domains[position] = course.domain

        max_enroll_log = float(enroll_log.max()) if n_courses else 1.0
        if max_enroll_log <= 0:
            max_enroll_log = 1.0

        domain_rating: dict[str, float] = {}
        for domain in sorted(set(domains.tolist())):
            mask = domains == domain
            domain_rating[str(domain)] = float(ratings[mask].mean()) if mask.any() else RATING_PRIOR

        for position, course_id in enumerate(self.course_ids):
            course = self.dataset.courses[course_id]
            enroll = float(max(int(course.enroll_count), 0))
            bayes = (float(course.rating) * enroll + RATING_SMOOTHING * RATING_PRIOR) / (
                enroll + RATING_SMOOTHING
            )
            rating_term = bayes / 5.0
            popularity_term = enroll_log[position] / max_enroll_log
            quality[position] = float(
                np.clip(
                    QUALITY_RATING_WEIGHT * rating_term
                    + QUALITY_POPULARITY_WEIGHT * popularity_term,
                    0.0,
                    1.0,
                )
            )
        return quality, enroll_log, domain_rating

    def _build_completion_prior(self) -> tuple[np.ndarray, float]:
        """课程完成率先验（全体学习者的完成率，向全局均值做贝叶斯平滑）。

        用于同伴样本不足时的社交分量回退值，也用于推荐理由里的「同类课程基准」。
        """
        n_courses = len(self.course_ids)
        totals = np.zeros(n_courses, dtype=np.float64)
        completions = np.zeros(n_courses, dtype=np.float64)
        n_inter = 0
        n_completed = 0
        for inter in self.dataset.interactions:
            position = self._course_pos.get(inter.course_id)
            n_inter += 1
            if inter.completed:
                n_completed += 1
                if position is not None:
                    completions[position] += 1.0
            if position is not None:
                totals[position] += 1.0
        population = float(n_completed / n_inter) if n_inter else 0.5
        prior = (completions + 2.0 * population) / (totals + 2.0)
        return np.clip(prior, 0.0, 1.0), population

    def _social_context(
        self, model: LearnerModel
    ) -> tuple[dict[str, tuple[int, int, float, float]], list[tuple[str, float]]]:
        """计算相似同伴集合与逐课程同伴完成统计（按模型版本缓存）。

        Returns:
            ``(stats, peers)``：``stats`` 为
            ``course_id -> (学过人数, 完成人数, 相似度权重和, 加权完成权重)``，
            ``peers`` 为 ``[(learner_id, similarity), ...]``。
        """
        key = (id(model), model.version)
        if self._social_cache is not None and self._social_cache[0] == key[0] and self._social_cache[1] == key[1]:
            return self._social_cache[2], self._social_cache[3]

        self._ensure_cf()
        peers = self.collab.similar_learners(model.learner.learner_id, SOCIAL_PEER_K)
        history = interactions_by_learner(self.dataset)
        stats: dict[str, tuple[int, int, float, float]] = {}
        if peers:
            raw: dict[str, list[float]] = {}
            for peer_id, similarity in peers:
                for inter in history.get(peer_id, ()):
                    entry = raw.setdefault(inter.course_id, [0.0, 0.0, 0.0, 0.0])
                    entry[0] += 1.0                                  # 学过人数
                    entry[1] += 1.0 if inter.completed else 0.0      # 完成人数
                    entry[2] += similarity                           # 相似度权重和
                    entry[3] += similarity if inter.completed else 0.0
            for course_id, entry in raw.items():
                stats[course_id] = (int(entry[0]), int(entry[1]), float(entry[2]), float(entry[3]))

        self._social_cache = (key[0], key[1], stats, peers)
        return stats, peers

    def _social_scores(self, model: LearnerModel, positions: np.ndarray) -> np.ndarray:
        """社交分量：相似同伴的贝叶斯平滑完成率。

        公式::

            conf      = w / (w + 1.5)                     # w = 学过该课的同伴相似度之和
            social    = conf * 同伴完成率 + (1 - conf) * 全体完成率先验

        收缩的作用是防止「只有 1 个相似同伴学过」时给出极端值。
        """
        stats, _ = self._social_context(model)
        result = np.ones(len(positions), dtype=np.float64) * 0.5
        for index, position in enumerate(positions):
            course_id = self.course_ids[int(position)]
            entry = stats.get(course_id)
            prior = float(self._completion_prior[int(position)])
            if entry is None or entry[2] <= 0.0:
                result[index] = prior
                continue
            _n_inter, _n_completed, weight, weighted_completed = entry
            conf = weight / (weight + SOCIAL_SHRINK)
            peer_rate = weighted_completed / weight
            result[index] = conf * peer_rate + (1.0 - conf) * prior
        return np.clip(result, 0.0, 1.0)

    # ==================================================================
    # 组装结果与理由
    # ==================================================================
    def _build_recommendation(
        self,
        model: LearnerModel,
        course: Course,
        score: float,
        component_values: dict[str, float],
        peer_stats: dict[str, tuple[int, int, float, float]],
    ) -> Recommendation:
        """把课程 + 分数 + 分量 + 理由组装成 :class:`Recommendation`。"""
        expected_gain = model.expected_gain(course)
        p_completion, risk_level = self._predict_completion(model, course)
        reasons = self._build_reasons(model, course, component_values, peer_stats)
        closure = model.target_closure
        in_closure = all(kp in closure for kp in course.kp_ids)
        return Recommendation(
            course=course,
            score=float(score),
            components=component_values,
            reasons=reasons,
            expected_gain=float(expected_gain),
            p_completion=p_completion,
            readiness=component_values["readiness"],
            style_fit=model.style_fit(course),
            time_fit=model.time_fit(course),
            risk_level=risk_level,
            kp_names=[self.graph.name_of(kp) for kp in course.kp_ids if kp in self.graph],
            in_goal_closure=in_closure,
        )

    def _predict_completion(
        self, model: LearnerModel, course: Course
    ) -> tuple[float | None, str | None]:
        """调用效果预测模块（若可用），任何异常都退化为 ``(None, None)``。"""
        if self.effect_predictor is None:
            return None, None
        try:
            prediction = self.effect_predictor.predict_completion(
                self.dataset, self.graph, model, course
            )
            p_completion = float(getattr(prediction, "p_completion", prediction))
            risk_level = getattr(prediction, "risk_level", None)
            return max(0.0, min(1.0, p_completion)), risk_level
        except Exception:  # noqa: BLE001 - 预测模块不可用时不得影响推荐主流程
            return None, None

    def _build_reasons(
        self,
        model: LearnerModel,
        course: Course,
        component_values: dict[str, float],
        peer_stats: dict[str, tuple[int, int, float, float]],
    ) -> list[str]:
        """生成 4~5 条具体、带数值的中文推荐理由（见模块 docstring 第二节）。"""
        reasons: list[str] = []
        required = model.required_mastery

        # ---- ① 目标缺口 ----
        gaps = model.course_gaps(course)
        if gaps:
            kp_id, mastery, gap = gaps[0]
            name = self.graph.name_of(kp_id)
            text = (
                f"填补目标缺口「{name}」（当前掌握度 {mastery:.2f}，"
                f"距目标掌握线 {required:.2f} 还差 {gap:.2f}）"
            )
            if len(gaps) > 1:
                others = "」「".join(self.graph.name_of(k) for k, _, _ in gaps[1:3])
                text += f"，同时覆盖「{others}」等 {len(gaps) - 1} 个缺口知识点"
            reasons.append(text)
        else:
            covered = [(kp, model.mastery_of(kp)) for kp in course.kp_ids if kp in self.graph]
            weak = [(kp, value) for kp, value in covered if value < required]
            if weak:
                kp_id, mastery = min(weak, key=lambda item: item[1])
                text = (
                    f"巩固目标路径上的「{self.graph.name_of(kp_id)}」"
                    f"（当前掌握度 {mastery:.2f}，距目标掌握线 {required:.2f} "
                    f"还差 {required - mastery:.2f}）"
                )
                if len(weak) > 1:
                    text += f"，本课共覆盖 {len(weak)} 个未达标知识点"
                reasons.append(text)
            else:
                names = [self.graph.name_of(kp) for kp in course.kp_ids[:2] if kp in self.graph]
                label = "」「".join(names) if names else course.title
                average = float(np.mean([value for _, value in covered])) if covered else 0.0
                reasons.append(
                    f"课程覆盖「{label}」，你的掌握度已达 {average:.2f}"
                    f"（≥ 目标掌握线 {required:.2f}），适合作为巩固与查漏补缺"
                )

        # ---- ② 先修就绪 ----
        ready, total, mean_mastery = model.readiness_detail(course)
        if total == 0:
            reasons.append(
                f"「{course.domain}」领域的基础课，无先修门槛（难度 {course.level}/5），可直接开始"
            )
        elif component_values["readiness"] >= 0.6:
            reasons.append(
                f"先修知识已就绪 {ready}/{total}（先修平均掌握度 {mean_mastery:.2f}），可顺利开始"
            )
        else:
            weak = [
                self.graph.name_of(kp)
                for kp in sorted(cached_course_prerequisites(self.dataset, self.graph).get(course.course_id, ()))
                if model.mastery_of(kp) < required
            ][:2]
            hint = "」「".join(weak) if weak else "前置知识点"
            reasons.append(
                f"先修仅就绪 {ready}/{total}（先修平均掌握度 {mean_mastery:.2f}），"
                f"建议先补「{hint}」再学本课"
            )

        # ---- ③ 相似同伴 ----
        entry = peer_stats.get(course.course_id)
        position = self._course_pos.get(course.course_id)
        prior_rate = (
            float(self._completion_prior[position]) if position is not None else self._population_rate
        )
        if entry is not None and entry[0] > 0:
            n_inter, n_completed = entry[0], entry[1]
            rate = n_completed / n_inter if n_inter else prior_rate
            reasons.append(
                f"与你背景相近的 {n_inter} 位学员学过本课，{n_completed} 人完成"
                f"（完成率 {rate:.0%}，全站基准 {prior_rate:.0%}）"
            )
        else:
            reasons.append(
                f"暂无足够相似同伴数据，本课全站完成率 {prior_rate:.0%}"
                f"（全站均值 {self._population_rate:.0%}）"
            )

        # ---- ④ 时间 / 风格匹配 ----
        hours = float(course.duration_hours)
        budget = max(float(model.learner.time_budget_hours_per_week), 0.5)
        style = model.style_fit(course)
        modality_name = MODALITY_NAMES.get(course.modality, course.modality)
        match_word = "一致" if model.learner.learning_style.prefer_modality == course.modality else "相近"
        reasons.append(
            f"预计 {hours:.1f} 小时，约占你每周 {budget:.1f} 小时预算的 {hours / budget:.0%}；"
            f"{modality_name}与你偏好的模态{match_word}（风格匹配度 {style:.2f}）"
        )

        # ---- ⑤ 口碑 ----
        domain_avg = self._domain_rating.get(course.domain, RATING_PRIOR)
        delta = float(course.rating) - domain_avg
        comparison = "高于" if delta >= 0 else "低于"
        reasons.append(
            f"评分 {float(course.rating):.1f}/5（{int(course.enroll_count):,} 人在学），"
            f"{comparison}「{course.domain}」领域均分 {domain_avg:.2f} 共 {abs(delta):.2f} 分，"
            f"口碑分项 {component_values['quality']:.2f}"
        )
        return reasons

    # ==================================================================
    # 诊断
    # ==================================================================
    def component_table(self, recommendations: Iterable[Recommendation]) -> list[dict[str, Any]]:
        """把推荐结果的分项得分整理成表格（便于报告与前端图表）。"""
        rows: list[dict[str, Any]] = []
        for rank, rec in enumerate(recommendations, start=1):
            row: dict[str, Any] = {
                "rank": rank,
                "course_id": rec.course.course_id,
                "title": rec.course.title,
                "score": round(float(rec.score), 4),
            }
            row.update({k: round(float(v), 4) for k, v in rec.components.items()})
            row["expected_gain"] = round(float(rec.expected_gain), 4)
            rows.append(row)
        return rows

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"HybridRecommender(courses={len(self.course_ids)}, "
            f"weights={self.weights}, mmr_lambda={self.mmr_lambda})"
        )
