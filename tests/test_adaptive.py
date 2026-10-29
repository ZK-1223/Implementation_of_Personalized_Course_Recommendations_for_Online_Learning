"""自适应调整引擎的单元测试（实现要求 ⑥）。

覆盖四类触发规则（未达标补强 / 已掌握跳过 / 进度落后 / 进度超前）、
「原地更新 model」、「路径重排」与「进度报告」。
"""

from __future__ import annotations

import pytest

from learnpath.adaptive.engine import AdaptiveEngine, AdaptationResult, FeedbackOutcome
from learnpath.config import ADAPTIVE
from learnpath.data.fixtures import build_tiny_dataset
from learnpath.data.schema import Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.bkt import MasteryTracker
from learnpath.learner.profile import LearnerModel
from learnpath.planning.planner import PathPlanner


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_tiny_dataset()


@pytest.fixture(scope="module")
def graph(dataset: Dataset) -> KnowledgeGraph:
    return KnowledgeGraph.from_dataset(dataset)


@pytest.fixture(scope="module")
def planner(dataset: Dataset, graph: KnowledgeGraph) -> PathPlanner:
    return PathPlanner(dataset, graph)


@pytest.fixture(scope="module")
def engine(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
) -> AdaptiveEngine:
    return AdaptiveEngine(dataset, graph, planner, MasteryTracker())


def _pick_learner_with_path(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
) -> tuple[LearnerModel, list]:
    """找一个能规划出多步路径的学习者（tiny 数据集中不是每个人都有路径）。"""
    for learner_id in sorted(dataset.learners):
        model = LearnerModel(dataset.learners[learner_id], dataset, graph)
        path = planner.plan(model, "balanced")
        if len(path.steps) >= 2:
            return model, path.steps
    pytest.skip("tiny 数据集上找不到多步路径")


# ============================================================
# 掌握度更新（原地）
# ============================================================


def test_apply_feedback_updates_model_in_place(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    engine: AdaptiveEngine,
) -> None:
    """``apply_feedback`` 必须**原地**提升传入 model 的掌握度。"""
    model, steps = _pick_learner_with_path(dataset, graph, planner)
    course = steps[0].course
    kp_id = course.kp_ids[0]
    before = float(model.mastery_of(kp_id))

    result = engine.apply_feedback(
        model,
        FeedbackOutcome(
            learner_id=model.learner.learner_id,
            course_id=course.course_id,
            progress=1.0,
            quiz_score=0.95,
            rating=5.0,
            hours_spent=float(course.duration_hours),
        ),
    )
    after = float(model.mastery_of(kp_id))
    assert after > before, "BKT 更新后掌握度应上升"
    assert after == pytest.approx(result.mastery_updates[kp_id], abs=1e-9)
    assert result.delta[kp_id] == pytest.approx(after - before, abs=1e-9)
    assert isinstance(result, AdaptationResult)
    assert result.actions, "至少要有一条中文动作说明"
    assert result.reason


def test_low_score_triggers_remediation(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    engine: AdaptiveEngine,
) -> None:
    """测验分低于 ``low_quiz_score`` → 判定未掌握并插入补强课。"""
    model, steps = _pick_learner_with_path(dataset, graph, planner)
    course = steps[0].course
    result = engine.apply_feedback(
        model,
        FeedbackOutcome(
            learner_id=model.learner.learner_id,
            course_id=course.course_id,
            progress=0.5,
            quiz_score=0.30,
            rating=2.0,
            hours_spent=float(course.duration_hours) * 0.5,
        ),
    )
    assert result.score == pytest.approx(0.30)
    assert "未掌握" in result.triggers
    assert any("低于阈值" in action for action in result.actions)
    assert any("掌握度" in action for action in result.actions)
    assert result.replanned is True
    assert result.new_path is not None


def test_high_score_triggers_skip(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    engine: AdaptiveEngine,
) -> None:
    """测验分高于 ``high_quiz_score`` → 判定已掌握并跳过后续冗余课程。"""
    model, steps = _pick_learner_with_path(dataset, graph, planner)
    course = steps[0].course
    result = engine.apply_feedback(
        model,
        FeedbackOutcome(
            learner_id=model.learner.learner_id,
            course_id=course.course_id,
            progress=1.0,
            quiz_score=0.99,
            rating=5.0,
            hours_spent=float(course.duration_hours),
        ),
    )
    assert result.score == pytest.approx(0.99)
    assert any(trigger in result.triggers for trigger in ("已掌握跳过", "已掌握"))
    assert any("高于阈值" in action for action in result.actions)
    assert result.replanned is True


def test_no_quiz_falls_back_to_progress_score(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    engine: AdaptiveEngine,
) -> None:
    """没有测验分时按进度折算（0 → 0.20，1 → 0.90），不会误判成优秀。"""
    model, steps = _pick_learner_with_path(dataset, graph, planner)
    course = steps[0].course
    outcome = FeedbackOutcome(
        learner_id=model.learner.learner_id,
        course_id=course.course_id,
        progress=1.0,
        quiz_score=None,
        hours_spent=float(course.duration_hours),
    )
    score, source = outcome.observed_score()
    assert score == pytest.approx(0.90)
    assert "进度" in source
    result = engine.apply_feedback(model, outcome)
    assert result.score == pytest.approx(0.90)
    assert "已掌握跳过" not in result.triggers


def test_result_to_dict_shape(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    engine: AdaptiveEngine,
) -> None:
    """``AdaptationResult.to_dict`` 必须覆盖 ``docs/API.md`` §5 的字段。"""
    model, steps = _pick_learner_with_path(dataset, graph, planner)
    course = steps[0].course
    result = engine.apply_feedback(
        model,
        FeedbackOutcome(
            learner_id=model.learner.learner_id,
            course_id=course.course_id,
            progress=0.8,
            quiz_score=0.65,
            rating=4.0,
            hours_spent=float(course.duration_hours) * 0.8,
        ),
    )
    payload = result.to_dict()
    for key in (
        "learner_id",
        "course_id",
        "mastery_updates",
        "delta",
        "actions",
        "replanned",
        "reason",
    ):
        assert key in payload
    assert isinstance(payload["actions"], list)
    for value in payload["mastery_updates"].values():
        assert 0.0 <= value <= 1.0


def test_unknown_course_raises(
    dataset: Dataset,
    graph: KnowledgeGraph,
    engine: AdaptiveEngine,
) -> None:
    """未知课程 id 必须抛 ``KeyError``，而不是静默忽略。"""
    model = LearnerModel(dataset.learners[sorted(dataset.learners)[0]], dataset, graph)
    with pytest.raises(KeyError):
        engine.apply_feedback(
            model,
            FeedbackOutcome(
                learner_id=model.learner.learner_id,
                course_id="C-NOT-EXIST",
                progress=1.0,
                quiz_score=0.8,
                hours_spent=1.0,
            ),
        )


# ============================================================
# 进度报告
# ============================================================


def test_progress_report(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    engine: AdaptiveEngine,
) -> None:
    """进度报告字段齐全、进度值落在 [0,1]、状态与阈值一致。"""
    model, _steps = _pick_learner_with_path(dataset, graph, planner)
    path = planner.plan(model, "balanced")
    report = engine.progress_report(model, path)
    for key in (
        "learner_id",
        "status",
        "ratio",
        "actual_progress",
        "plan_progress",
        "planned_hours",
        "actual_hours",
        "path_total_hours",
        "done_courses",
        "remaining_courses",
        "elapsed_weeks",
        "next_course",
        "retention_estimate",
        "weekly",
    ):
        assert key in report
    assert 0.0 <= report["actual_progress"] <= 1.0
    assert 0.0 <= report["plan_progress"] <= 1.0
    assert report["status"] in {"落后", "正常", "超前"}
    if report["ratio"] < ADAPTIVE.behind_schedule:
        assert report["status"] == "落后"
    elif report["ratio"] > ADAPTIVE.ahead_schedule:
        assert report["status"] == "超前"
    else:
        assert report["status"] == "正常"
    assert 0.0 <= report["retention_estimate"] <= 1.0
    assert isinstance(report["weekly"], list)


def test_progress_report_empty_path(
    dataset: Dataset,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    engine: AdaptiveEngine,
) -> None:
    """空路径不应崩溃，应返回中性进度。"""
    model = LearnerModel(dataset.learners[sorted(dataset.learners)[0]], dataset, graph)
    empty = planner.rebuild(model, [], "balanced")
    report = engine.progress_report(model, empty)
    assert report["ratio"] == pytest.approx(1.0)
    assert report["path_total_hours"] == pytest.approx(0.0)
