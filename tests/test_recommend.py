"""推荐模块单元测试：内容推荐、协同过滤、混合推荐与 MMR 多样性重排。

覆盖范围：
    * :class:`ContentRecommender` 的分数值域、向量化一致性与缺口敏感性；
    * :class:`CollaborativeRecommender` 的 SVD 拟合、分数值域、隐向量相似度、冷启动；
    * :func:`mmr_rerank` 的相关性 / 多样性权衡；
    * :class:`HybridRecommender` 的五路分量、目标闭包约束、可解释理由、预测模块解耦。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from learnpath.data.fixtures import build_tiny_dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import LearnerModel
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


@pytest.fixture(scope="module")
def dataset():
    return build_tiny_dataset()


@pytest.fixture(scope="module")
def graph(dataset):
    return KnowledgeGraph.from_dataset(dataset)


@pytest.fixture(scope="module")
def content(dataset, graph):
    return ContentRecommender(dataset, graph)


@pytest.fixture(scope="module")
def collab(dataset):
    return CollaborativeRecommender(dataset).fit()


@pytest.fixture(scope="module")
def engine(dataset, graph):
    return HybridRecommender(dataset, graph)


@pytest.fixture(scope="module")
def model(dataset, graph):
    return LearnerModel(dataset.learners["L-0001"], dataset, graph)


# ===========================================================================
# 内容推荐
# ===========================================================================


class TestContentRecommender:
    """知识图谱内容推荐。"""

    def test_score_all_shape_and_range(self, content, model, dataset):
        scores = content.score_all(model)
        assert scores.shape == (len(dataset.courses),)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
        assert scores.std() > 0.0

    def test_score_matches_score_all(self, content, model, dataset):
        scores = content.score_all(model)
        for position, course_id in enumerate(content.course_ids):
            assert content.score(model, dataset.courses[course_id]) == pytest.approx(
                float(scores[position])
            )

    def test_unknown_course_scores_zero(self, content, model, dataset):
        ghost = dataset.courses[next(iter(dataset.courses))]
        payload = ghost.to_dict()
        payload["course_id"] = "C-NOT-EXIST"
        from learnpath.data.schema import Course

        unknown = Course.from_dict(payload)
        assert content.score(model, unknown) == 0.0

    def test_readiness_array_consistent_with_model(self, content, model, dataset):
        readiness = content.readiness_all(model)
        for position, course_id in enumerate(content.course_ids):
            assert readiness[position] == pytest.approx(
                model.readiness_for(dataset.courses[course_id]), abs=1e-12
            )

    def test_root_course_has_full_readiness(self, content, model, dataset):
        for position, course_id in enumerate(content.course_ids):
            course = dataset.courses[course_id]
            if not graph_prerequisites(model, course):
                assert content.readiness_all(model)[position] == pytest.approx(1.0)

    def test_mastering_target_kps_reduces_gain(self, dataset, graph, content):
        """目标知识点越熟练，课程带来的缺口收益越低。"""
        learner = dataset.learners["L-0001"]
        before = LearnerModel(learner, dataset, graph)
        after = LearnerModel(learner, dataset, graph)
        for kp_id in before.target_kps:
            after._mastery[after._index[kp_id]] = 0.95  # noqa: SLF001
        after._invalidate()  # noqa: SLF001
        assert content.gain_all(after).sum() < content.gain_all(before).sum()

    def test_components_keys(self, content, model):
        parts = content.components(model)
        assert set(parts) == {"gain_term", "level_match", "readiness", "score"}
        assert all(isinstance(v, np.ndarray) for v in parts.values())


def graph_prerequisites(model: LearnerModel, course) -> frozenset[str]:
    return model.graph.course_prerequisites(course)


# ===========================================================================
# 协同过滤
# ===========================================================================


class TestCollaborativeRecommender:
    """SVD 隐语义协同过滤。"""

    def test_fit_shapes(self, collab, dataset):
        assert collab.matrix_shape == (len(dataset.learners), len(dataset.courses))
        assert 0.0 < collab.matrix_density <= 1.0
        assert collab.fitted

    def test_score_in_unit_interval(self, collab, dataset):
        for learner_id in list(dataset.learners)[:4]:
            for course_id in list(dataset.courses)[:6]:
                value = collab.score(learner_id, course_id)
                assert 0.0 <= value <= 1.0

    def test_score_all_shape(self, collab, dataset):
        scores = collab.score_all(next(iter(dataset.learners)))
        assert scores.shape == (len(dataset.courses),)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_unknown_ids_return_zero(self, collab):
        assert collab.score("L-NOT-EXIST", "C-0001") == 0.0
        assert collab.score("L-0001", "C-NOT-EXIST") == 0.0

    def test_similar_learners_excludes_self_and_is_sorted(self, collab, dataset):
        learner_id = "L-0001"
        peers = collab.similar_learners(learner_id, 4)
        assert peers
        assert all(pid != learner_id for pid, _ in peers)
        similarities = [value for _, value in peers]
        assert similarities == sorted(similarities, reverse=True)
        assert all(0.0 < value <= 1.0 for value in similarities)
        assert len(peers) <= 4

    def test_similar_learners_zero_k(self, collab):
        assert collab.similar_learners("L-0001", 0) == []

    def test_similarity_matrix_shape(self, collab, dataset):
        matrix = collab.similarity_matrix()
        n = len(dataset.learners)
        assert matrix.shape == (n, n)
        assert np.allclose(np.diag(matrix), 1.0)
        assert np.allclose(matrix, matrix.T, atol=1e-9)

    def test_cold_start_learner_scores_zero(self, dataset):
        """没有任何交互的学习者，协同过滤应「弃权」（全部返回 0）。"""
        from learnpath.data.schema import Dataset

        cold_id = next(iter(dataset.learners))
        cold = Dataset(
            knowledge_points=dataset.knowledge_points,
            kp_edges=dataset.kp_edges,
            courses=dataset.courses,
            learners=dataset.learners,
            goals=dataset.goals,
            interactions=[i for i in dataset.interactions if i.learner_id != cold_id],
        )
        recommender = CollaborativeRecommender(cold).fit()
        assert float(recommender.score_all(cold_id).sum()) == pytest.approx(0.0)

    def test_fit_is_deterministic(self, dataset):
        first = CollaborativeRecommender(dataset).fit().score_all("L-0001")
        second = CollaborativeRecommender(dataset).fit().score_all("L-0001")
        assert np.allclose(first, second)

    def test_lazy_fit_on_first_use(self, dataset):
        recommender = CollaborativeRecommender(dataset)
        assert not recommender.fitted
        recommender.score("L-0001", "C-0001")
        assert recommender.fitted


# ===========================================================================
# 课程相似度与 MMR
# ===========================================================================


class TestCourseSimilarity:
    """课程特征与相似度。"""

    def test_feature_matrix_is_normalised(self, dataset):
        course_ids, matrix = build_course_features(dataset)
        assert len(course_ids) == len(dataset.courses)
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-9)

    def test_self_similarity_is_one(self, dataset):
        course = next(iter(dataset.courses.values()))
        assert course_similarity(course, course) == pytest.approx(1.0)

    def test_similarity_is_symmetric_and_bounded(self, dataset):
        courses = list(dataset.courses.values())
        for i in range(0, 10):
            for j in range(0, 10):
                a, b = courses[i], courses[j]
                value = course_similarity(a, b)
                assert 0.0 <= value <= 1.0
                assert value == pytest.approx(course_similarity(b, a))

    def test_same_kp_courses_are_more_similar(self, dataset):
        """知识点有重叠的两门课，其相似度必须高于知识点完全不重叠的两门课。"""
        from itertools import combinations

        courses = list(dataset.courses.values())
        overlapping = None
        disjoint = None
        for a, b in combinations(courses, 2):
            shared = set(a.kp_ids) & set(b.kp_ids)
            if shared and overlapping is None:
                overlapping = (a, b)
            if not shared and disjoint is None:
                disjoint = (a, b)
            if overlapping and disjoint:
                break
        if overlapping is None or disjoint is None:
            pytest.skip("夹具中没有可对比的课程组合")
        assert course_similarity(*overlapping) > course_similarity(*disjoint)

    def test_matrix_similarity_matches_scalar(self, dataset):
        course_ids, matrix = build_course_features(dataset)
        position = {cid: i for i, cid in enumerate(course_ids)}
        pair_sim = matrix @ matrix.T
        courses = list(dataset.courses.values())
        for i in range(5):
            for j in range(5):
                a, b = courses[i], courses[j]
                assert course_similarity(a, b) == pytest.approx(
                    float(pair_sim[position[a.course_id], position[b.course_id]]), abs=1e-9
                )


class TestMMRRerank:
    """MMR 多样性重排。"""

    def _candidates(self, dataset, n=12):
        courses = list(dataset.courses.values())[:n]
        return [(course, 1.0 - 0.02 * i) for i, course in enumerate(courses)]

    def test_returns_k_items(self, dataset):
        candidates = self._candidates(dataset)
        result = mmr_rerank(candidates, 5, 0.75)
        assert len(result) == 5
        assert all(isinstance(item, tuple) for item in result)

    def test_fewer_candidates_than_k(self, dataset):
        candidates = self._candidates(dataset, n=3)
        assert len(mmr_rerank(candidates, 10, 0.75)) == 3

    def test_empty_input(self):
        assert mmr_rerank([], 5, 0.75) == []
        assert mmr_rerank([], 0, 0.75) == []

    def test_lambda_one_preserves_relevance_order(self, dataset):
        candidates = self._candidates(dataset, n=8)
        result = mmr_rerank(candidates, 8, 1.0)
        assert [item[1] for item in result] == [item[1] for item in candidates]

    def test_lambda_zero_increases_diversity(self, dataset):
        """λ = 0 时优先选与已选最不相似的课程，列表内平均相似度必须下降。"""

        def mean_similarity(items) -> float:
            values = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    values.append(course_similarity(items[i][0], items[j][0]))
            return sum(values) / len(values) if values else 0.0

        candidates = self._candidates(dataset, n=10)
        greedy = mmr_rerank(candidates, 6, 1.0)
        diversified = mmr_rerank(candidates, 6, 0.0)
        assert mean_similarity(diversified) <= mean_similarity(greedy) + 1e-9


# ===========================================================================
# 混合推荐
# ===========================================================================


class TestHybridRecommender:
    """多路召回 + 加权融合 + MMR + 可解释理由。"""

    def test_returns_at_most_k(self, engine, model):
        results = engine.recommend(model, k=5)
        assert 0 < len(results) <= 5

    def test_all_recommendations_inside_goal_closure(self, engine, model):
        """主推荐列表必须全部落在目标先修闭包内。"""
        for recommendation in engine.recommend(model, k=10):
            assert recommendation.in_goal_closure
            assert set(recommendation.course.kp_ids) <= set(model.target_closure)

    def test_exclude_enrolled(self, engine, model):
        results = engine.recommend(model, k=10, exclude_enrolled=True)
        covered = model.courses_covered()
        assert all(r.course.course_id not in covered for r in results)

    def test_components_keys_and_range(self, engine, model):
        for recommendation in engine.recommend(model, k=5):
            assert tuple(recommendation.components) == COMPONENT_KEYS
            for value in recommendation.components.values():
                assert 0.0 <= value <= 1.0
            assert 0.0 <= recommendation.score <= 1.0

    def test_score_equals_weighted_sum(self, engine, model):
        weights = engine.weights
        mapping = {
            "content": weights.content,
            "collaborative": weights.collaborative,
            "quality": weights.quality,
            "readiness": weights.readiness,
            "social": weights.social,
        }
        total = sum(mapping.values())
        for recommendation in engine.recommend(model, k=5):
            expected = sum(
                mapping[key] / total * recommendation.components[key] for key in COMPONENT_KEYS
            )
            assert recommendation.score == pytest.approx(expected, abs=1e-9)

    def test_reasons_are_specific(self, engine, model):
        """理由必须具体：不得出现空洞话术，且至少两条带数值。"""
        banned = ("推荐给你", "为你推荐", "值得一学", "优质课程")
        for recommendation in engine.recommend(model, k=5):
            assert recommendation.reasons
            assert len(recommendation.reasons) >= 4
            for reason in recommendation.reasons:
                assert isinstance(reason, str) and len(reason) >= 12
                for phrase in banned:
                    assert phrase not in reason
            numeric = [r for r in recommendation.reasons if any(ch.isdigit() for ch in r)]
            assert len(numeric) >= 3
            assert any("「" in r for r in recommendation.reasons)

    def test_reasons_cover_required_dimensions(self, engine, model):
        text = " ".join(engine.recommend(model, k=3)[0].reasons)
        assert "目标" in text  # 目标缺口
        assert ("先修" in text) or ("无先修门槛" in text)  # 先修就绪
        assert "学员" in text  # 相似同伴
        assert "小时" in text  # 时间匹配
        assert "评分" in text  # 口碑

    def test_expected_gain_non_negative(self, engine, model):
        for recommendation in engine.recommend(model, k=5):
            assert recommendation.expected_gain >= 0.0

    def test_p_completion_none_without_predictor(self, engine, model):
        for recommendation in engine.recommend(model, k=3):
            assert recommendation.p_completion is None
            assert recommendation.risk_level is None

    def test_broken_predictor_is_isolated(self, dataset, graph, model):
        """预测模块抛异常时，推荐必须照常返回（只把 p_completion 置空）。"""

        class Boom:
            def predict_completion(self, *args, **kwargs):
                raise RuntimeError("模型加载失败")

        engine = HybridRecommender(dataset, graph, effect_predictor=Boom())
        results = engine.recommend(model, k=3)
        assert results
        assert all(r.p_completion is None for r in results)

    def test_predictor_result_is_used(self, dataset, graph, model):
        class Fake:
            def predict_completion(self, dataset_, graph_, model_, course_):
                return SimpleNamespace(p_completion=0.72, risk_level="中")

        engine = HybridRecommender(dataset, graph, effect_predictor=Fake())
        results = engine.recommend(model, k=3)
        assert results[0].p_completion == pytest.approx(0.72)
        assert results[0].risk_level == "中"

    def test_deterministic(self, engine, model):
        first = [r.course.course_id for r in engine.recommend(model, k=8)]
        second = [r.course.course_id for r in engine.recommend(model, k=8)]
        assert first == second

    def test_explain_returns_reasons(self, engine, model):
        recommendation = engine.recommend(model, k=1)[0]
        reasons = engine.explain(model, recommendation)
        assert isinstance(reasons, list)
        assert reasons and all(isinstance(r, str) for r in reasons)

    def test_k_zero_returns_empty(self, engine, model):
        assert engine.recommend(model, k=0) == []

    def test_to_dict_contract(self, engine, model):
        payload = engine.recommend(model, k=1)[0].to_dict(rank=1)
        for key in (
            "rank",
            "course_id",
            "title",
            "domain",
            "level",
            "duration_hours",
            "modality",
            "modality_name",
            "rating",
            "enroll_count",
            "price",
            "kp_names",
            "score",
            "components",
            "reasons",
            "expected_gain",
            "p_completion",
            "readiness",
            "style_fit",
            "time_fit",
            "in_goal_closure",
        ):
            assert key in payload
        assert payload["rank"] == 1
        assert set(payload["components"]) == set(COMPONENT_KEYS)

    def test_goal_courses_are_subset_of_catalog(self, engine, dataset):
        ids = engine.goal_courses("L-0001")
        assert ids
        assert set(ids) <= set(dataset.courses)
        closure = engine.graph.prerequisite_closure(
            dataset.goals[dataset.learners["L-0001"].goal_id].target_kp_ids
        )
        for course_id in ids:
            assert set(dataset.courses[course_id].kp_ids) <= set(closure)

    def test_recommend_is_fast(self, engine, model):
        engine.recommend(model, k=10)  # 预热缓存
        start = time.perf_counter()
        engine.recommend(model, k=10)
        assert time.perf_counter() - start < 1.0

    def test_state_change_affects_recommendation(self, dataset, graph):
        """学习反馈写入掌握度后，推荐结果应随之变化或至少收益下降。"""
        model = LearnerModel(dataset.learners["L-0002"], dataset, graph)
        engine = HybridRecommender(dataset, graph)
        results = engine.recommend(model, k=5)
        if not results:
            pytest.skip("没有可推荐课程")
        target = results[0]
        before_gain = model.expected_gain(target.course)
        model.apply_outcome(target.course, 1.0, 1.0)
        after_gain = model.expected_gain(target.course)
        assert after_gain < before_gain

    def test_ablation_switches(self, dataset, graph, model):
        """关闭 KG 约束后允许推荐闭包外课程（消融开关生效）。"""
        engine = HybridRecommender(
            dataset, graph, use_prerequisite_filter=False, only_goal_closure=False
        )
        results = engine.recommend(model, k=10)
        assert results
        assert any(not r.in_goal_closure for r in results)


class TestRecommendationDataclass:
    """推荐结果数据结构的默认值契约。"""

    def test_defaults(self, dataset):
        course = next(iter(dataset.courses.values()))
        recommendation = Recommendation(
            course=course,
            score=0.5,
            components={key: 0.5 for key in COMPONENT_KEYS},
            reasons=["缺口「X」还差 0.3"],
            expected_gain=0.2,
        )
        assert recommendation.p_completion is None
        assert recommendation.in_goal_closure is True
        assert recommendation.course_id == course.course_id
        assert recommendation.to_dict()["p_completion"] is None
