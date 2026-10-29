"""学习者建模单元测试：BKT 知识追踪 + 多维学习者画像。

覆盖范围：
    * :class:`MasteryTracker` 的观测更新、学习转移、能力调制、遗忘衰减；
    * :class:`LearnerModel` 的先验初始化、历史回放、先修就绪、风格 / 时间匹配、
      收益估计的无副作用性、状态更新的原地语义、画像快照契约。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from learnpath.config import (
    BKT,
    DOMAINS,
    MASTERY_DECAY_PER_WEEK,
    MASTERY_GAP,
    MODALITIES,
)
from learnpath.data.fixtures import build_tiny_dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.bkt import MasteryTracker
from learnpath.learner.profile import LearnerModel, ProfileSnapshot


@pytest.fixture(scope="module")
def dataset():
    return build_tiny_dataset()


@pytest.fixture(scope="module")
def graph(dataset):
    return KnowledgeGraph.from_dataset(dataset)


@pytest.fixture(scope="module")
def model(dataset, graph):
    return LearnerModel(dataset.learners["L-0001"], dataset, graph)


# ===========================================================================
# BKT
# ===========================================================================


class TestMasteryTracker:
    """贝叶斯知识追踪四参数模型。"""

    def test_update_stays_in_unit_interval(self):
        tracker = MasteryTracker()
        for prior in (0.0, 0.05, 0.3, 0.5, 0.9, 0.999):
            for score in (0.0, 0.2, 0.5, 0.8, 1.0):
                value = tracker.update(prior, score, 1.0, 0.5)
                assert 0.0 <= value <= 0.999
                assert isinstance(value, float)

    def test_update_monotone_in_observed_score(self):
        """观测分数越高，掌握度提升越多（单调性）。"""
        tracker = MasteryTracker()
        values = [tracker.update(0.3, s, 1.0, 0.5) for s in (0.0, 0.3, 0.6, 0.9, 1.0)]
        assert values == sorted(values)
        assert values[-1] > values[0]

    def test_update_monotone_in_prior(self):
        tracker = MasteryTracker()
        values = [tracker.update(p, 0.8, 1.0, 0.5) for p in (0.0, 0.2, 0.4, 0.6, 0.8)]
        assert values == sorted(values)

    def test_ability_increases_learning_rate(self):
        """能力越强学习率越高（能力调制项 0.55 + 0.90 * ability）。"""
        tracker = MasteryTracker()
        low = tracker.update(0.2, 0.5, 1.0, 0.1)
        high = tracker.update(0.2, 0.5, 1.0, 0.95)
        assert high > low
        assert tracker.effective_learning_rate(1.0, 0.0) < tracker.effective_learning_rate(1.0, 0.5)
        assert tracker.effective_learning_rate(1.0, 0.5) < tracker.effective_learning_rate(1.0, 1.0)

    def test_kp_weight_increases_gain(self):
        """覆盖强度越高，单次学习的知识摄入越多。"""
        tracker = MasteryTracker()
        values = [tracker.update(0.25, 0.8, w, 0.5) for w in (0.2, 0.5, 0.8, 1.0)]
        assert values == sorted(values)

    def test_posterior_is_pure_observation_update(self):
        """posterior 不含学习转移，因此不大于 update 的结果。"""
        tracker = MasteryTracker()
        post = tracker.posterior(0.3, 0.9)
        next_value = tracker.update(0.3, 0.9, 1.0, 0.5)
        assert 0.3 < post < 1.0
        assert next_value > post

    def test_observation_likelihood_matches_definition(self):
        tracker = MasteryTracker()
        like_mastered, like_unmastered = tracker.observation_likelihood(1.0)
        assert like_mastered == pytest.approx(1.0 - BKT.p_slip)
        assert like_unmastered == pytest.approx(BKT.p_guess)

    def test_decay_matches_anchored_exponential_formula(self):
        """遗忘以先验基线为锚点：只衰减「超出先验」的那部分。

        公式：``decayed = p_init + (m - p_init) * γ^Δw``。
        以先验为锚点而非衰减到 0，是因为遗忘抹去的是"已学到"的部分，
        而"学习前的基线常识"不应被时间抹平（否则长期未复习的知识点会被压到近 0，
        全库掌握度均值塌陷、知识缺口被人为放大）。
        """
        tracker = MasteryTracker()
        floor = tracker.p_init
        for weeks in (0.5, 4.0, 12.0, 52.0):
            expected = floor + (0.8 - floor) * MASTERY_DECAY_PER_WEEK**weeks
            assert tracker.decay(0.8, weeks) == pytest.approx(expected, rel=1e-9)

    def test_decay_is_monotone_and_bounded(self):
        tracker = MasteryTracker()
        values = [tracker.decay(0.6, w) for w in (0.0, 1.0, 4.0, 12.0, 52.0)]
        assert values == sorted(values, reverse=True)
        assert values[0] == pytest.approx(0.6)
        assert all(0.0 <= v <= 1.0 for v in values)
        assert tracker.decay(0.6, -3.0) == pytest.approx(0.6)
        # 长期闲置也不会低于先验基线
        assert tracker.decay(0.6, 500.0) >= tracker.p_init - 1e-9

    def test_decay_factor_consistency(self):
        tracker = MasteryTracker()
        factor = tracker.decay_factor(6.0)
        floor = tracker.p_init
        assert tracker.decay(0.7, 6.0) == pytest.approx(floor + (0.7 - floor) * factor)
        # 低于先验基线时不衰减（基线之下没有"已学到"的部分可遗忘）
        assert tracker.decay(floor / 2, 6.0) == pytest.approx(floor / 2)


# ===========================================================================
# 学习者模型
# ===========================================================================


class TestLearnerModelInitialisation:
    """先验初始化与历史回放。"""

    def test_covers_every_knowledge_point(self, model, dataset):
        vector = model.mastery_vector()
        assert set(vector) == set(dataset.knowledge_points)
        assert len(model.mastery_array()) == len(dataset.knowledge_points)

    def test_no_all_zero_prior(self, model):
        """未给出的知识点按先修就绪度推断，不应全为 0。"""
        values = np.array(list(model.mastery_vector().values()))
        assert (values > 0.0).all()
        assert values.mean() > 0.02
        assert values.max() > 0.05

    def test_explicit_initial_mastery_is_respected(self, dataset, graph):
        """没有历史记录时，``initial_mastery`` 必须被原样采用（只做概率截断）。"""
        learner = dataset.learners["L-0001"]
        if not learner.initial_mastery:
            pytest.skip("该夹具学习者没有显式初始掌握度")
        from learnpath.data.schema import Dataset

        empty = Dataset(
            knowledge_points=dataset.knowledge_points,
            kp_edges=dataset.kp_edges,
            courses=dataset.courses,
            learners=dataset.learners,
            goals=dataset.goals,
            interactions=[
                i for i in dataset.interactions if i.learner_id != learner.learner_id
            ],
        )
        model = LearnerModel(learner, empty, graph)
        for kp_id, value in learner.initial_mastery.items():
            assert model.mastery_of(kp_id) == pytest.approx(value)

    def test_replay_can_decay_stale_knowledge(self, dataset, graph):
        """跨越多月的历史回放会按遗忘曲线衰减，因此掌握度允许低于最初自评值。"""
        learner = dataset.learners["L-0001"]
        if not learner.initial_mastery:
            pytest.skip("该夹具学习者没有显式初始掌握度")
        model = LearnerModel(learner, dataset, graph)
        declined = [
            kp_id
            for kp_id, value in learner.initial_mastery.items()
            if model.mastery_of(kp_id) < value
        ]
        assert declined or model.history_size() == 0

    def test_history_replay_raises_mastery_of_covered_kps(self, dataset, graph):
        learner = dataset.learners["L-0001"]
        covered = graph
        model = LearnerModel(learner, dataset, graph)
        covered_ids = model.courses_covered()
        assert covered_ids, "夹具学习者应至少有一条交互记录"
        # 学过的课程覆盖的知识点，其掌握度应高于「同领域未学过」知识点的均值
        touched = {
            kp for cid in covered_ids for kp in dataset.courses[cid].kp_ids
        }
        untouched = set(dataset.knowledge_points) - touched
        if not untouched:
            pytest.skip("该学习者已覆盖全部知识点，无法对比")
        touched_mean = float(np.mean([model.mastery_of(k) for k in touched]))
        untouched_mean = float(np.mean([model.mastery_of(k) for k in untouched]))
        assert touched_mean > untouched_mean

    def test_mastery_values_are_probabilities(self, model):
        assert all(0.0 <= v <= 1.0 for v in model.mastery_vector().values())

    def test_mastery_of_unknown_kp_is_zero(self, model):
        assert model.mastery_of("KP-NOT-EXIST") == 0.0


class TestLearnerModelQueries:
    """缺口 / 领域 / 覆盖等查询契约。"""

    def test_knowledge_gap_subset_of_target_closure(self, model):
        gaps = model.knowledge_gap()
        closure = model.graph.prerequisite_closure(model.target_kps)
        assert set(gaps) <= set(closure)
        for kp_id, gap in gaps.items():
            assert model.mastery_of(kp_id) < MASTERY_GAP
            assert gap == pytest.approx(model.required_mastery - model.mastery_of(kp_id))
            assert gap > 0

    def test_domain_mastery_covers_all_domains(self, model):
        domain_mastery = model.domain_mastery()
        assert set(domain_mastery) <= set(DOMAINS)
        assert all(0.0 <= v <= 1.0 for v in domain_mastery.values())

    def test_courses_covered_matches_history(self, model, dataset):
        expected = {i.course_id for i in dataset.interactions_of(model.learner.learner_id)}
        assert model.courses_covered() == expected

    def test_course_gaps_only_within_closure(self, model):
        for course in list(dataset_courses(model))[:10]:
            for kp_id, mastery, gap in model.course_gaps(course):
                assert kp_id in model.target_closure
                assert gap == pytest.approx(model.required_mastery - mastery)
                assert gap > 0


def dataset_courses(model: LearnerModel):
    return model.dataset.courses.values()


class TestFitScores:
    """先修就绪度、风格匹配、时间可承受度。"""

    def test_readiness_in_unit_interval(self, model):
        for course in list(model.dataset.courses.values())[:15]:
            value = model.readiness_for(course)
            assert 0.0 <= value <= 1.0

    def test_course_without_prerequisite_is_fully_ready(self, model):
        for course in model.dataset.courses.values():
            if not model.graph.course_prerequisites(course):
                assert model.readiness_for(course) == pytest.approx(1.0)
                break

    def test_readiness_improves_when_prerequisites_mastered(self, dataset, graph):
        """把先修知识点掌握度拉满后，就绪度必须提升。"""
        model = LearnerModel(dataset.learners["L-0002"], dataset, graph)
        target = None
        for course in dataset.courses.values():
            prereqs = graph.course_prerequisites(course)
            if prereqs:
                target = (course, prereqs)
                break
        assert target is not None
        course, prereqs = target
        before = model.readiness_for(course)
        for kp_id in prereqs:
            index = model._index[kp_id]  # noqa: SLF001 - 白盒测试
            model._mastery[index] = 0.95  # noqa: SLF001
        model._invalidate()  # noqa: SLF001
        after = model.readiness_for(course)
        assert after > before
        assert after > 0.9

    def test_style_fit_in_unit_interval(self, model):
        for course in list(model.dataset.courses.values())[:15]:
            assert 0.0 <= model.style_fit(course) <= 1.0

    def test_style_fit_prefers_matching_modality(self, dataset, graph):
        model = LearnerModel(dataset.learners["L-0003"], dataset, graph)
        preferred = model.learner.learning_style.prefer_modality
        same = [c for c in dataset.courses.values() if c.modality == preferred]
        diff = [c for c in dataset.courses.values() if c.modality not in (preferred, "mixed")]
        if not same or not diff:
            pytest.skip("夹具中没有可对比的模态组合")
        # 用同一理论比例对比，隔离模态的影响
        course_same = same[0]
        course_diff = min(
            diff, key=lambda c: abs(c.theory_ratio - course_same.theory_ratio)
        )
        assert model.style_fit(course_same) >= model.style_fit(course_diff)

    def test_time_fit_in_unit_interval_and_prefers_short_courses(self, dataset, graph):
        model = LearnerModel(dataset.learners["L-0004"], dataset, graph)
        courses = sorted(dataset.courses.values(), key=lambda c: c.duration_hours)
        short, long = courses[0], courses[-1]
        assert 0.0 <= model.time_fit(short) <= 1.0
        assert 0.0 <= model.time_fit(long) <= 1.0
        assert model.time_fit(short) >= model.time_fit(long)

    def test_style_vector_sums_to_one(self, model):
        vector = model.style_vector()
        assert vector["theory"] + vector["practice"] == pytest.approx(1.0)
        modality_sum = sum(vector[m] for m in MODALITIES)
        assert modality_sum == pytest.approx(1.0)


class TestExpectedGainAndUpdates:
    """收益估计与状态更新。"""

    def test_expected_gain_is_non_negative_and_finite(self, model):
        for course in list(model.dataset.courses.values())[:15]:
            value = model.expected_gain(course)
            assert value >= 0.0
            assert math.isfinite(value)
            assert value < 2.0

    def test_expected_gain_has_no_side_effect(self, model):
        course = next(iter(model.dataset.courses.values()))
        before_vector = model.mastery_vector()
        before_version = model.version
        model.expected_gain(course)
        model.readiness_for(course)
        model.style_fit(course)
        model.time_fit(course)
        assert model.version == before_version
        assert model.mastery_vector() == before_vector

    def test_expected_gain_decreases_with_mastery(self, dataset, graph):
        """已掌握的知识点边际收益递减。"""
        learner = dataset.learners["L-0005"]
        course = list(dataset.courses.values())[0]
        low_model = LearnerModel(learner, dataset, graph)
        high_model = LearnerModel(learner, dataset, graph)
        for kp_id in course.kp_ids:
            high_model._mastery[high_model._index[kp_id]] = 0.95  # noqa: SLF001
        high_model._invalidate()  # noqa: SLF001
        assert high_model.expected_gain(course) < low_model.expected_gain(course)

    def test_apply_outcome_mutates_in_place_and_returns_delta(self, dataset, graph):
        model = LearnerModel(dataset.learners["L-0006"], dataset, graph)
        course = list(dataset.courses.values())[0]
        before = {kp: model.mastery_of(kp) for kp in course.kp_ids}
        version_before = model.version

        deltas = model.apply_outcome(course, score=0.9, effort_ratio=1.0)

        assert set(deltas) == set(course.kp_ids)
        assert model.version == version_before + 1
        for kp_id, delta in deltas.items():
            assert delta > 0.0
            assert model.mastery_of(kp_id) == pytest.approx(before[kp_id] + delta)
            assert model.mastery_of(kp_id) > before[kp_id]

    def test_apply_outcome_high_score_beats_low_score(self, dataset, graph):
        course = list(dataset.courses.values())[0]
        good = LearnerModel(dataset.learners["L-0007"], dataset, graph)
        bad = LearnerModel(dataset.learners["L-0007"], dataset, graph)
        good_delta = sum(good.apply_outcome(course, 1.0, 1.0).values())
        bad_delta = sum(bad.apply_outcome(course, 0.0, 1.0).values())
        assert good_delta > bad_delta

    def test_apply_decay_reduces_mastery(self, model):
        before = model.mastery_vector()
        version_before = model.version
        model.apply_decay(8.0)
        after = model.mastery_vector()
        assert model.version == version_before + 1
        assert all(after[k] < before[k] for k in before)
        factor = MASTERY_DECAY_PER_WEEK**8.0
        assert after[next(iter(after))] == pytest.approx(before[next(iter(before))] * factor)


class TestProfileSnapshot:
    """画像快照字段契约。"""

    def test_snapshot_fields(self, model):
        snapshot = model.snapshot()
        assert isinstance(snapshot, ProfileSnapshot)
        assert snapshot.learner_id == model.learner.learner_id
        assert set(snapshot.mastery) == set(model.dataset.knowledge_points)
        assert isinstance(snapshot.domain_mastery, dict)
        assert list(snapshot.target_kps) == list(model.target_kps)
        assert isinstance(snapshot.gaps, dict)
        assert 0.0 <= snapshot.readiness <= 1.0
        assert set(snapshot.style_vector) >= {"theory", "practice"}
        assert snapshot.time_budget == pytest.approx(model.learner.time_budget_hours_per_week)
        assert snapshot.total_available_hours == pytest.approx(
            model.learner.total_available_hours
        )
        assert snapshot.gap_count == len(snapshot.gaps)
        assert 0 <= snapshot.mastered_count <= len(snapshot.mastery)
        assert 0.0 <= snapshot.completion_rate <= 1.0
        assert 0.0 <= snapshot.avg_mastery <= 1.0

    def test_snapshot_is_json_serialisable(self, model):
        import json

        payload = model.snapshot().to_dict()
        text = json.dumps(payload, ensure_ascii=False)
        assert "learner_id" in text
        assert all(isinstance(v, (int, float)) for v in payload["mastery"].values())

    def test_snapshot_readiness_is_goal_level(self, model):
        snapshot = model.snapshot()
        assert 0.0 <= snapshot.readiness <= 1.0

    def test_snapshot_reflects_state_change(self, dataset, graph):
        model = LearnerModel(dataset.learners["L-0008"], dataset, graph)
        before = model.snapshot()
        course = next(
            (c for c in dataset.courses.values() if model.expected_gain(c) > 0.05), None
        )
        if course is None:
            pytest.skip("夹具中没有可提升掌握度的课程")
        model.apply_outcome(course, 1.0, 1.0)
        after = model.snapshot()
        assert after.avg_mastery > before.avg_mastery
        assert after.gap_count <= before.gap_count
