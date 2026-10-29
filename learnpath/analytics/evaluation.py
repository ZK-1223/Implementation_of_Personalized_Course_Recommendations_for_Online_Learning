"""推荐算法离线评测：留出法 + 排序指标 + 多策略对比 + 消融实验。

对应报告「实验与结果分析」章节的全部数据来源，也是 ``docs/API.md`` 第 10 节
``/api/evaluation`` 的实现。

--------------------------------------------------------------------------------
一、评测协议（留出法 / Leave-Out）
--------------------------------------------------------------------------------
1. **正反馈定义**：``completed = True`` 或 ``progress >= 0.8`` 的交互视为正反馈
   （「学完了」或「快学完了」，都说明学习者认可这门课）。
2. **测试集切分**：对每位正反馈数 >= ``MIN_POSITIVES`` 的学习者，用固定随机种子
   （:data:`learnpath.config.EVAL_SEED`）**随机抽取 20%** 的正反馈移入测试集，
   其余交互留在训练集。固定种子保证结果可复现。
3. **训练侧重建**：用去掉测试项后的交互重建数据集，再在其上训练协同过滤（SVD）、
   构造学习者模型（BKT 历史回放）——**严格避免测试项泄漏到训练**。
4. **候选集**：对每位学习者，从「训练集中未交互过的课程」里排序取 Top-K，
   与测试集正反馈比对。
5. **K 取值**：:data:`learnpath.config.EVAL_K` = 10（即报告中的 P@10 / R@10 / NDCG@10）。

--------------------------------------------------------------------------------
二、指标定义
--------------------------------------------------------------------------------
设某学习者的推荐列表为 ``L``（长度 K），测试正反馈集合为 ``R``（|R| = n_R）::

    Precision@K = |L ∩ R| / K
    Recall@K    = |L ∩ R| / n_R
    DCG@K       = Σ_{i=1..K} 1[L_i ∈ R] / log2(i + 1)
    NDCG@K      = DCG@K / IDCG@K,   IDCG@K = Σ_{i=1..min(|R|,K)} 1 / log2(i + 1)
    AP          = Σ_{i=1..K} (1[L_i ∈ R] · Precision@i) / min(|R|, K)

对全体学习者取算术平均得到 Precision@K / Recall@K / NDCG@K / MAP。

另有三类**列表层面**的指标（每次推荐列表计算一次，再对学习者取平均）::

    Coverage  = |∪_u L_u| / |C|                              # 目录覆盖率，衡量长尾能力
    Diversity = 1 - mean_{i<j} sim(c_i, c_j)                  # 品类内平均不相似度
    Novelty   = mean_{c ∈ L} -log2(pop_c) / log2(|C|)         # 归一化新颖度
    HitRate   = |{u : |L_u ∩ R_u| > 0}| / |U|                 # 至少命中一次的用户占比

``sim`` 使用与混合推荐 MMR 完全相同的课程加权余弦相似度（见
:func:`learnpath.recommend.hybrid.course_similarity`），保证「多样性」在推荐与评测两侧
口径一致。``pop_c`` 取训练集上的交互频次占比，故 Novelty 刻画的是「推荐了多少冷门课」。

--------------------------------------------------------------------------------
三、对比策略
--------------------------------------------------------------------------------
=======================  =========================================================
策略                      说明
=======================  =========================================================
``Popularity 基线``       按训练集交互频次排序（最热门的课优先）——强基线
``Content 内容推荐``      :class:`learnpath.recommend.content.ContentRecommender`
``CF-SVD 协同过滤``        :class:`learnpath.recommend.collaborative.CollaborativeRecommender`
``Hybrid 混合推荐（本系统）``  :class:`learnpath.recommend.hybrid.HybridRecommender`
=======================  =========================================================

为什么必须打赢 Popularity：热门课程基线在新用户占比较高、兴趣分布集中的数据集上非常强
（它天然拟合了「大多数人都学这些课」），只有在 NDCG@10 上稳定超过它，才能证明知识图谱
约束 + 协同信号 + 多样性重排的组合确实提供了「因人而异」的增量价值。

--------------------------------------------------------------------------------
四、消融实验
--------------------------------------------------------------------------------
:meth:`OfflineEvaluator.ablation` 在**同一测试集**上逐个关闭模块，报告 NDCG@10 的变化量
（delta = 变体 − 完整系统，负值表示该模块有正贡献）：

1. 去掉知识图谱先修约束（先修过滤关闭 + readiness 权重归零）
2. 去掉 MMR 多样性重排（λ = 1.0）
3. 去掉协同过滤信号（collaborative 权重归零）
4. 去掉社交同伴信号（social 权重归零）
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np

from learnpath.config import (
    EVAL_K,
    EVAL_SEED,
    EVAL_TEST_RATIO,
    RECOMMEND_WEIGHTS,
)
from learnpath.data.schema import Dataset, Interaction
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import LearnerModel, interactions_by_learner
from learnpath.recommend.collaborative import CollaborativeRecommender
from learnpath.recommend.content import ContentRecommender
from learnpath.recommend.hybrid import HybridRecommender, build_course_features

__all__ = [
    "OfflineEvaluator",
    "StrategyResult",
    "build_evaluation",
    "EVALUATION_STRATEGY_NAMES",
]


#: 正反馈判定阈值：完成或进度 >= 该值。
POSITIVE_PROGRESS_THRESHOLD: Final[float] = 0.8

#: 参与评测的最少正反馈数（低于此值的学习者无法稳定切分，跳过）。
MIN_POSITIVES: Final[int] = 3

#: 四个对比策略的固定名称（顺序即 API 返回顺序）。
EVALUATION_STRATEGY_NAMES: Final[tuple[str, ...]] = (
    "Popularity 基线",
    "Content 内容推荐",
    "CF-SVD 协同过滤",
    "Hybrid 混合推荐（本系统）",
)

#: ``models/model_metrics.json`` 中的两个键（由 prediction 模块产出）。
EFFECT_MODEL_KEY: Final[str] = "effect_model"
GAIN_MODEL_KEY: Final[str] = "gain_model"

#: ``build_evaluation`` 的结果缓存，避免 API 每次请求都重跑评测。
_BUILD_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


# ===========================================================================
# 数据结构
# ===========================================================================


@dataclass(slots=True)
class StrategyResult:
    """单一推荐策略的离线指标集合（对应 API 中 ``recommenders`` 数组的元素）。

    Attributes:
        name: 策略名称（见 :data:`EVALUATION_STRATEGY_NAMES`）。
        precision_at_k: P@K，推荐列表命中比例。
        recall_at_k: R@K，测试正反馈被召回比例。
        ndcg_at_k: NDCG@K，考虑排序位置的归一化折损累计增益（核心指标）。
        map: MAP，平均精度均值。
        coverage: 目录覆盖率，推荐过的课程占全部课程的比例。
        diversity: 列表内平均不相似度（1 - 平均两两相似度）。
        novelty: 归一化新颖度（-log2 流行度的均值，按 log2(|C|) 归一）。
        hit_rate: 至少命中一次的学习者占比。
        n_learners: 参与评测的学习者数量（不进入 API 序列化）。
    """

    name: str
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map: float
    coverage: float
    diversity: float
    novelty: float
    hit_rate: float
    n_learners: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转为 ``docs/API.md`` 第 10 节 ``recommenders`` 元素的字典。"""
        return {
            "name": self.name,
            "precision_at_k": round(float(self.precision_at_k), 4),
            "recall_at_k": round(float(self.recall_at_k), 4),
            "ndcg_at_k": round(float(self.ndcg_at_k), 4),
            "map": round(float(self.map), 4),
            "coverage": round(float(self.coverage), 4),
            "diversity": round(float(self.diversity), 4),
            "novelty": round(float(self.novelty), 4),
            "hit_rate": round(float(self.hit_rate), 4),
        }


class _Accumulator:
    """单个策略的指标累加器（按学习者逐条累加）。"""

    __slots__ = (
        "name",
        "n",
        "precision",
        "recall",
        "ndcg",
        "map",
        "diversity",
        "novelty",
        "hits",
        "hit_users",
        "recommended",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.n = 0
        self.precision = 0.0
        self.recall = 0.0
        self.ndcg = 0.0
        self.map = 0.0
        self.diversity = 0.0
        self.novelty = 0.0
        self.hits = 0
        self.hit_users = 0
        self.recommended: set[str] = set()

    def add(
        self,
        precision: float,
        recall: float,
        ndcg: float,
        ap: float,
        diversity: float,
        novelty: float,
        hit: bool,
        course_ids: Iterable[str],
    ) -> None:
        self.n += 1
        self.precision += precision
        self.recall += recall
        self.ndcg += ndcg
        self.map += ap
        self.diversity += diversity
        self.novelty += novelty
        if hit:
            self.hits += 1
            self.hit_users += 1
        self.recommended.update(course_ids)

    def result(self, n_courses: int) -> StrategyResult:
        n = max(self.n, 1)
        return StrategyResult(
            name=self.name,
            precision_at_k=self.precision / n,
            recall_at_k=self.recall / n,
            ndcg_at_k=self.ndcg / n,
            map=self.map / n,
            coverage=len(self.recommended) / n_courses if n_courses else 0.0,
            diversity=self.diversity / n,
            novelty=self.novelty / n,
            hit_rate=self.hit_users / n,
            n_learners=self.n,
        )


# ===========================================================================
# 指标函数（独立导出便于单元测试）
# ===========================================================================


def dcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """折损累计增益 ``DCG@K``（二元相关性）。"""
    total = 0.0
    for rank, course_id in enumerate(recommended[:k], start=1):
        if course_id in relevant:
            total += 1.0 / math.log2(rank + 1)
    return total


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """归一化折损累计增益 ``NDCG@K``。``relevant`` 为空时返回 0。"""
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal <= 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / ideal


def average_precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """``AP@K``：命中位置越靠前得分越高。分母取 ``min(|R|, K)`` 保证落在 [0,1]。"""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, course_id in enumerate(recommended[:k], start=1):
        if course_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def list_diversity(recommended: Sequence[str], similarity: np.ndarray, position: Mapping[str, int]) -> float:
    """列表内平均不相似度 ``1 - mean(两两相似度)``。"""
    indices = [position[c] for c in recommended if c in position]
    if len(indices) < 2:
        return 0.0
    block = similarity[np.ix_(indices, indices)]
    upper = block[np.triu_indices(len(indices), k=1)]
    if upper.size == 0:
        return 0.0
    return float(1.0 - upper.mean())


def list_novelty(recommended: Sequence[str], popularity: Mapping[str, float], n_courses: int) -> float:
    """归一化新颖度：``mean(-log2(pop) / log2(|C|))``。流行度为 0 时用最小非零概率兜底。"""
    if not recommended or n_courses <= 1:
        return 0.0
    floor = 1.0 / (n_courses + 1)
    denominator = math.log2(n_courses)
    values = []
    for course_id in recommended:
        probability = max(float(popularity.get(course_id, 0.0)), floor)
        values.append(-math.log2(probability) / denominator)
    return float(sum(values) / len(values))


# ===========================================================================
# 评测器
# ===========================================================================


class OfflineEvaluator:
    """推荐算法的离线评测器（留出法 + 多策略对比 + 消融）。

    Args:
        dataset: 完整数据集。
        graph: 知识图谱。
        k: 评测截断长度，默认 :data:`learnpath.config.EVAL_K`（10）。
        test_ratio: 每位学习者正反馈中划入测试集的比例，默认 0.2。
        seed: 切分随机种子，默认 :data:`learnpath.config.EVAL_SEED`（42）。
        max_learners: 最多评测多少个学习者（None = 全部）。用于测试与快速试跑。
        min_positives: 参与评测所需的最少正反馈数。

    Example:
        >>> from learnpath.data.fixtures import build_tiny_dataset
        >>> from learnpath.knowledge.graph import KnowledgeGraph
        >>> ds = build_tiny_dataset()
        >>> graph = KnowledgeGraph.from_dataset(ds)
        >>> evaluator = OfflineEvaluator(ds, graph, k=5, max_learners=6)
        >>> data = evaluator.evaluate()
        >>> [row["name"] for row in data["recommenders"]][0]
        'Popularity 基线'
    """

    __slots__ = (
        "dataset",
        "graph",
        "k",
        "test_ratio",
        "seed",
        "max_learners",
        "min_positives",
        "_split_cache",
        "_content_cache",
        "_collab_cache",
        "_feature_cache",
        "_results_cache",
    )

    def __init__(
        self,
        dataset: Dataset,
        graph: KnowledgeGraph,
        k: int = EVAL_K,
        test_ratio: float = EVAL_TEST_RATIO,
        seed: int = EVAL_SEED,
        max_learners: int | None = None,
        min_positives: int = MIN_POSITIVES,
    ) -> None:
        self.dataset: Dataset = dataset
        self.graph: KnowledgeGraph = graph
        self.k: int = int(k)
        self.test_ratio: float = float(test_ratio)
        self.seed: int = int(seed)
        self.max_learners: int | None = max_learners
        self.min_positives: int = int(min_positives)

        self._split_cache: tuple[Dataset, dict[str, set[str]]] | None = None
        self._content_cache: tuple[Dataset, ContentRecommender] | None = None
        self._collab_cache: tuple[Dataset, CollaborativeRecommender] | None = None
        self._feature_cache: tuple[Dataset, np.ndarray, dict[str, int]] | None = None
        self._results_cache: dict[str, Any] | None = None

    # ==================================================================
    # 切分
    # ==================================================================
    def split(self) -> tuple[Dataset, dict[str, set[str]]]:
        """留出法切分，返回 ``(训练集, {learner_id: 测试正反馈 course_id 集合})``。"""
        if self._split_cache is not None:
            return self._split_cache

        grouped = interactions_by_learner(self.dataset)
        rng = random.Random(self.seed)
        test: dict[str, set[str]] = {}
        held_out: set[tuple[str, str]] = set()

        for learner_id in sorted(grouped):
            interactions = grouped[learner_id]
            positives = [
                inter
                for inter in interactions
                if inter.completed or inter.progress >= POSITIVE_PROGRESS_THRESHOLD
            ]
            if len(positives) < self.min_positives:
                continue
            n_test = max(1, int(round(len(positives) * self.test_ratio)))
            n_test = min(n_test, max(1, len(positives) - 1))
            sampled = rng.sample(positives, n_test)
            test[learner_id] = {inter.course_id for inter in sampled}
            held_out.update((inter.learner_id, inter.course_id) for inter in sampled)

        train_interactions: list[Interaction] = [
            inter
            for inter in self.dataset.interactions
            if (inter.learner_id, inter.course_id) not in held_out
        ]
        train = Dataset(
            knowledge_points=self.dataset.knowledge_points,
            kp_edges=self.dataset.kp_edges,
            courses=self.dataset.courses,
            learners=self.dataset.learners,
            goals=self.dataset.goals,
            interactions=train_interactions,
            meta={
                **dict(self.dataset.meta),
                "split": {
                    "protocol": "leave-out",
                    "test_ratio": self.test_ratio,
                    "seed": self.seed,
                    "n_test_items": len(held_out),
                },
            },
        )
        self._split_cache = (train, test)
        return self._split_cache

    # ==================================================================
    # 训练侧组件
    # ==================================================================
    def _content(self, train: Dataset) -> ContentRecommender:
        if self._content_cache is not None and self._content_cache[0] is train:
            return self._content_cache[1]
        recommender = ContentRecommender(train, self.graph)
        self._content_cache = (train, recommender)
        return recommender

    def _collab(self, train: Dataset) -> CollaborativeRecommender:
        if self._collab_cache is not None and self._collab_cache[0] is train:
            return self._collab_cache[1]
        recommender = CollaborativeRecommender(train).fit()
        self._collab_cache = (train, recommender)
        return recommender

    def _features(self, train: Dataset) -> tuple[np.ndarray, dict[str, int]]:
        """课程特征相似度矩阵与课程下标（用于 diversity 指标）。"""
        if self._feature_cache is not None and self._feature_cache[0] is train:
            return self._feature_cache[1], self._feature_cache[2]
        course_ids, features = build_course_features(train)
        similarity = features @ features.T
        position = {cid: i for i, cid in enumerate(course_ids)}
        self._feature_cache = (train, similarity, position)
        return similarity, position

    def _learners(self, test: Mapping[str, set[str]]) -> list[str]:
        learners = sorted(lid for lid, items in test.items() if items)
        if self.max_learners is not None:
            learners = learners[: int(self.max_learners)]
        return learners

    # ==================================================================
    # 主评测
    # ==================================================================
    def evaluate(self) -> dict[str, Any]:
        """执行四策略对比评测。

        Returns:
            ``{"recommenders": [StrategyResult.to_dict(), ...], "notes": [...], "k": 10,
            "n_learners": N, "n_courses": M}``。
        """
        if self._results_cache is not None:
            return self._results_cache

        train, test = self.split()
        learners = self._learners(test)
        if not learners:
            raise ValueError("没有满足条件的学习者（正反馈数量不足），无法评测")

        content = self._content(train)
        collab = self._collab(train)
        hybrid = HybridRecommender(
            train,
            self.graph,
            content_recommender=content,
            collaborative_recommender=collab,
        )
        similarity, position = self._features(train)
        n_courses = len(train.courses)
        popularity = self._popularity(train)
        popularity_array = self._popularity_array(train)

        accumulators = {
            name: _Accumulator(name) for name in EVALUATION_STRATEGY_NAMES
        }

        for learner_id in learners:
            relevant = test[learner_id]
            model = LearnerModel(train.learners[learner_id], train, self.graph)
            enrolled = model.courses_covered()
            unseen = np.array(
                [cid not in enrolled for cid in train.courses], dtype=bool
            )
            if unseen.sum() == 0:
                continue

            rankings: dict[str, list[str]] = {}
            rankings[EVALUATION_STRATEGY_NAMES[0]] = self._rank_by(
                train, unseen, popularity_array
            )
            rankings[EVALUATION_STRATEGY_NAMES[1]] = self._rank_by(
                train, unseen, content.score_all(model)
            )
            rankings[EVALUATION_STRATEGY_NAMES[2]] = self._rank_by(
                train, unseen, collab.score_all(learner_id)
            )
            hybrid_recs = hybrid.recommend(model, k=self.k, exclude_enrolled=True)
            rankings[EVALUATION_STRATEGY_NAMES[3]] = [r.course.course_id for r in hybrid_recs]

            for name, recommended in rankings.items():
                hits = len(set(recommended) & relevant)
                accumulators[name].add(
                    precision=hits / self.k if self.k else 0.0,
                    recall=hits / len(relevant) if relevant else 0.0,
                    ndcg=ndcg_at_k(recommended, relevant, self.k),
                    ap=average_precision_at_k(recommended, relevant, self.k),
                    diversity=list_diversity(recommended, similarity, position),
                    novelty=list_novelty(recommended, popularity, n_courses),
                    hit=hits > 0,
                    course_ids=recommended,
                )

        recommenders = [accumulators[name].result(n_courses).to_dict() for name in EVALUATION_STRATEGY_NAMES]
        self._results_cache = {
            "recommenders": recommenders,
            "notes": self._notes(train, test, learners),
            "k": self.k,
            "n_learners": len(learners),
            "n_courses": n_courses,
        }
        return self._results_cache

    def _rank_by(self, train: Dataset, unseen: np.ndarray, scores: np.ndarray) -> list[str]:
        """按分数对未见过的课程排序取 Top-K（分数相同则按课程 id 稳定排序）。"""
        indices = np.flatnonzero(unseen)
        if indices.size == 0:
            return []
        values = scores[indices]
        order = np.argsort(-values, kind="stable")[: self.k]
        course_ids = list(train.courses)
        return [course_ids[int(indices[i])] for i in order]

    @staticmethod
    def _popularity(train: Dataset) -> dict[str, float]:
        """训练集上的课程流行度分布 ``course_id -> 频次占比``。"""
        counts: dict[str, float] = {}
        for inter in train.interactions:
            counts[inter.course_id] = counts.get(inter.course_id, 0.0) + 1.0
        total = sum(counts.values()) or 1.0
        return {cid: value / total for cid, value in counts.items()}

    @staticmethod
    def _popularity_array(train: Dataset) -> np.ndarray:
        """训练集课程热度向量（用于 Popularity 基线的排序）。"""
        course_ids = list(train.courses)
        position = {cid: i for i, cid in enumerate(course_ids)}
        counts = np.zeros(len(course_ids), dtype=np.float64)
        for inter in train.interactions:
            index = position.get(inter.course_id)
            if index is not None:
                counts[index] += 1.0
        return counts

    def _notes(self, train: Dataset, test: Mapping[str, set[str]], learners: Sequence[str]) -> list[str]:
        """生成评测协议说明（写入 API 的 ``notes`` 字段）。"""
        n_test = sum(len(v) for v in test.values())
        return [
            "离线评测采用留一/留出混合协议：先按学习者分组，再随机抽取每位学习者 "
            f"{self.test_ratio:.0%} 的正反馈（completed=True 或 progress>=0.8）作为测试集，"
            f"训练集保留剩余交互（共 {len(train.interactions)} 条），严格避免测试项泄漏。",
            f"候选集为训练集中未与该学习者交互过的课程，截断长度 K={self.k}；"
            f"共评测 {len(learners)} 位学习者、{n_test} 条测试正反馈。",
            "NDCG@K 采用二元相关性（命中即 1），IDCG 按 min(|R|, K) 计算，"
            "因此 Recall@K < 1 时 NDCG@K 仍可反映排序质量。",
            "Diversity 用列表内平均两两不相似度（1 - 余弦相似度），与混合推荐 MMR 的相似度口径一致；"
            "Novelty 为 -log2(训练集流行度) 的均值并按 log2(|C|) 归一，越大代表越偏冷门。",
            "所有随机过程（测试集切分、SVD 初始化）均使用固定种子，结果可复现。",
        ]

    # ==================================================================
    # 消融实验
    # ==================================================================
    def ablation(self) -> list[dict[str, Any]]:
        """在同一测试集上关闭单个模块，报告 NDCG@10 的变化量。

        Returns:
            ``[{"variant": str, "ndcg_at_k": float, "delta": float,
            "precision_at_k": float, "diversity": float, "recall_at_k": float}, ...]``，
            第一条为「完整混合推荐（基线）」，``delta`` 为相对该基线的差。
        """
        train, test = self.split()
        learners = self._learners(test)
        if not learners:
            return []

        content = self._content(train)
        collab = self._collab(train)
        similarity, position = self._features(train)
        base_weights = RECOMMEND_WEIGHTS

        variants: list[tuple[str, dict[str, Any]]] = [
            ("完整混合推荐（基线）", {}),
            (
                "去掉知识图谱先修约束",
                {
                    "use_prerequisite_filter": False,
                    "only_goal_closure": False,
                    "weights": replace(base_weights, readiness=0.0),
                },
            ),
            ("去掉 MMR 多样性重排", {"mmr_lambda": 1.0}),
            ("去掉协同过滤信号", {"weights": replace(base_weights, collaborative=0.0)}),
            ("去掉社交同伴信号", {"weights": replace(base_weights, social=0.0)}),
        ]

        rows: list[dict[str, Any]] = []
        baseline_ndcg: float | None = None
        # 学习者模型与状态无关（recommend 不修改掌握度），因此 5 个变体共用一份，
        # 避免重复的 BKT 历史回放（720 学习者 × 5 变体 ≈ 节省 10 s）。
        models = [
            (learner_id, LearnerModel(train.learners[learner_id], train, self.graph))
            for learner_id in learners
        ]
        for variant_name, kwargs in variants:
            engine = HybridRecommender(
                train,
                self.graph,
                content_recommender=content,
                collaborative_recommender=collab,
                **kwargs,
            )
            accumulator = _Accumulator(variant_name)
            for learner_id, model in models:
                relevant = test[learner_id]
                recommended = [
                    rec.course.course_id
                    for rec in engine.recommend(model, k=self.k, exclude_enrolled=True)
                ]
                hits = len(set(recommended) & relevant)
                accumulator.add(
                    precision=hits / self.k if self.k else 0.0,
                    recall=hits / len(relevant) if relevant else 0.0,
                    ndcg=ndcg_at_k(recommended, relevant, self.k),
                    ap=average_precision_at_k(recommended, relevant, self.k),
                    diversity=list_diversity(recommended, similarity, position),
                    novelty=0.0,
                    hit=hits > 0,
                    course_ids=recommended,
                )
            result = accumulator.result(len(train.courses))
            if baseline_ndcg is None:
                baseline_ndcg = result.ndcg_at_k
            rows.append(
                {
                    "variant": variant_name,
                    "ndcg_at_k": round(float(result.ndcg_at_k), 4),
                    "delta": round(float(result.ndcg_at_k - baseline_ndcg), 4),
                    "precision_at_k": round(float(result.precision_at_k), 4),
                    "recall_at_k": round(float(result.recall_at_k), 4),
                    "diversity": round(float(result.diversity), 4),
                }
            )
        return rows

    # ==================================================================
    # 汇总
    # ==================================================================
    def run(self) -> dict[str, Any]:
        """执行评测 + 消融，返回 ``{"recommenders", "ablation", "notes", "k", ...}``。"""
        evaluation = self.evaluate()
        return {
            "recommenders": evaluation["recommenders"],
            "ablation": self.ablation(),
            "notes": list(evaluation["notes"]),
            "k": evaluation["k"],
            "n_learners": evaluation["n_learners"],
            "n_courses": evaluation["n_courses"],
        }

    def to_api_dict(self, effect_model: Mapping[str, Any] | None = None,
                    gain_model: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """按 ``docs/API.md`` 第 10 节结构输出（效果预测指标可外部注入）。"""
        data = self.run()
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "recommenders": data["recommenders"],
            "effect_model": None if effect_model is None else dict(effect_model),
            "gain_model": None if gain_model is None else dict(gain_model),
            "ablation": data["ablation"],
            "notes": data["notes"],
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"OfflineEvaluator(k={self.k}, test_ratio={self.test_ratio}, "
            f"seed={self.seed}, max_learners={self.max_learners})"
        )


# ===========================================================================
# 模块级 API 入口
# ===========================================================================


def _load_model_metrics(metrics_path: Path | None) -> dict[str, Any]:
    """读取 ``models/model_metrics.json``（缺失 / 损坏时返回空字典，绝不抛异常）。"""
    path = metrics_path
    if path is None:
        from learnpath.paths import MODELS_DIR, MODEL_METRICS_FILE

        path = MODELS_DIR / MODEL_METRICS_FILE
    try:
        if not Path(path).is_file():
            return {}
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def build_evaluation(
    dataset: Dataset,
    graph: KnowledgeGraph,
    metrics_path: str | Path | None = None,
    *,
    k: int = EVAL_K,
    max_learners: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """构造 ``/api/evaluation`` 的完整响应（推荐指标 + 消融 + 效果模型指标）。

    返回值**严格对应** ``docs/API.md`` 第 10 节的顶层结构::

        {
          "generated_at": "ISO8601",
          "recommenders": [ {...}, ... ],     # 4 个策略，顺序固定
          "effect_model": {...} | null,       # 从 models/model_metrics.json 读取
          "gain_model":   {...} | null,
          "ablation":     [ {"variant","ndcg_at_k","delta", ...}, ... ],
          "notes":        ["...", ...]
        }

    Args:
        dataset: 完整数据集。
        graph: 知识图谱。
        metrics_path: 效果预测指标文件路径；默认 ``models/model_metrics.json``。
            文件不存在时 ``effect_model`` / ``gain_model`` 返回 ``None``，
            **不影响** ``recommenders`` 与 ``ablation`` 的计算。
        k: 评测截断长度。
        max_learners: 最多评测的学习者数量（None = 全部）。
        force: 忽略缓存强制重新计算。

    Returns:
        可直接由 FastAPI ``return`` 的字典（所有值均为 JSON 可序列化类型）。
    """
    cache_key = (id(dataset), int(k), None if max_learners is None else int(max_learners))
    cached = _BUILD_CACHE.get(cache_key)
    if cached is not None and not force:
        payload = copy.deepcopy(cached)
        payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
        return payload

    evaluator = OfflineEvaluator(dataset, graph, k=k, max_learners=max_learners)
    data = evaluator.run()
    metrics = _load_model_metrics(metrics_path)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "recommenders": data["recommenders"],
        "effect_model": metrics.get(EFFECT_MODEL_KEY),
        "gain_model": metrics.get(GAIN_MODEL_KEY),
        "ablation": data["ablation"],
        "notes": list(data["notes"]),
    }
    _BUILD_CACHE[cache_key] = copy.deepcopy(payload)
    return payload
