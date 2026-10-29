"""推荐模块（实现要求第 1、7 条的推荐侧：混合推荐 + 可解释理由）。

对外契约：

    # learnpath/recommend/content.py
    class ContentRecommender:
        def __init__(self, dataset: Dataset, graph: KnowledgeGraph) -> None
        def score(self, model: LearnerModel, course: Course) -> float

    # learnpath/recommend/collaborative.py
    class CollaborativeRecommender:
        def __init__(self, dataset: Dataset, n_components: int = CF_SVD_COMPONENTS)
        def fit(self) -> None
        def score(self, learner_id: str, course_id: str) -> float
        def similar_learners(self, learner_id: str, k: int) -> list[tuple[str, float]]

    # learnpath/recommend/hybrid.py
    @dataclass
    class Recommendation:
        course: Course
        score: float
        components: dict[str, float]      # content / collaborative / quality / readiness / social
        reasons: list[str]                # 中文可解释理由
        expected_gain: float
        p_completion: float | None

    class HybridRecommender:
        def __init__(self, dataset: Dataset, graph: KnowledgeGraph,
                     effect_predictor=None) -> None
        def recommend(self, model: LearnerModel, k: int = 10,
                      exclude_enrolled: bool = True) -> list[Recommendation]
        def explain(self, model: LearnerModel, recommendation: Recommendation) -> list[str]

模块还应提供 ``mmr_rerank`` 以保证推荐多样性（参数见 :data:`learnpath.config.MMR_LAMBDA`）。
"""

from __future__ import annotations

from learnpath.recommend.collaborative import CollaborativeRecommender
from learnpath.recommend.content import ContentRecommender
from learnpath.recommend.hybrid import (
    COMPONENT_KEYS,
    HybridRecommender,
    Recommendation,
    build_course_features,
    course_similarity,
    mmr_rerank,
)

__all__ = [
    "ContentRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
    "Recommendation",
    "mmr_rerank",
    "course_similarity",
    "build_course_features",
    "COMPONENT_KEYS",
]
