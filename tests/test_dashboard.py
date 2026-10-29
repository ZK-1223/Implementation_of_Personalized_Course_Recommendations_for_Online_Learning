"""学习分析仪表板的单元测试（实现要求 ⑧）。

只验证「数据结构 + 真实数据聚合」，不依赖训练好的效果预测模型：
``completion_forecast`` 在模型缺失时会退化为解析式估计（``source="analytic"``），
这也是接口契约允许的行为。
"""

from __future__ import annotations

import pytest

from learnpath.analytics.dashboard import (
    build_dashboard,
    difficulty_distribution,
    domain_radar,
    gap_table,
    history_activity,
    learning_streak,
    mastery_curve,
    modality_mix,
    time_allocation,
)
from learnpath.config import DOMAINS, MASTERY_GAP, MASTERY_MASTERED, MODALITIES
from learnpath.data.fixtures import build_tiny_dataset
from learnpath.data.schema import Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import LearnerModel
from learnpath.planning.planner import PathPlanner


REQUIRED_KEYS = (
    "kpi",
    "mastery_curve",
    "domain_radar",
    "difficulty_distribution",
    "modality_mix",
    "time_allocation",
    "completion_forecast",
    "gap_table",
    "history_activity",
)


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
def model(dataset: Dataset, graph: KnowledgeGraph) -> LearnerModel:
    return LearnerModel(dataset.learners[sorted(dataset.learners)[0]], dataset, graph)


@pytest.fixture(scope="module")
def path(planner: PathPlanner, model: LearnerModel):
    return planner.plan(model, "balanced")


@pytest.fixture(scope="module")
def dashboard(dataset: Dataset, graph: KnowledgeGraph, model: LearnerModel, path) -> dict:
    return build_dashboard(dataset, graph, model, path)


# ============================================================
# 结构
# ============================================================


def test_dashboard_has_all_sections(dashboard: dict, model: LearnerModel) -> None:
    """九个字段必须齐全，且 ``learner_id`` 正确。"""
    assert dashboard["learner_id"] == model.learner.learner_id
    for key in REQUIRED_KEYS:
        assert key in dashboard, f"缺少字段 {key}"


def test_kpi_fields(dashboard: dict, dataset: Dataset, model: LearnerModel, path) -> None:
    """KPI 字段完整、取值合法、且三个「已掌握」口径自洽。"""
    kpi = dashboard["kpi"]
    for key in (
        "overall_progress",
        "mastered_kps",
        "total_target_kps",
        "avg_mastery",
        "readiness",
        "planned_hours",
        "available_hours",
        "risk_courses",
        "learning_streak_weeks",
    ):
        assert key in kpi
    assert 0.0 <= kpi["overall_progress"] <= 1.0
    assert 0.0 <= kpi["avg_mastery"] <= 1.0
    assert 0.0 <= kpi["readiness"] <= 1.0
    assert kpi["mastered_kps"] <= kpi["closure_kps"]
    assert kpi["mastered_kps_all"] >= kpi["mastered_kps_in_closure"]
    # 与 snapshot().mastered_count 完全一致（lead 要求两处口径统一）
    assert kpi["mastered_kps_all"] == model.snapshot().mastered_count
    assert kpi["planned_hours"] == pytest.approx(path.total_hours, abs=0.01)
    assert kpi["available_hours"] == pytest.approx(
        model.learner.time_budget_hours_per_week * model.learner.deadline_weeks, abs=0.01
    )
    assert kpi["risk_courses"] >= 0
    assert kpi["learning_streak_weeks"] >= 0


def test_mastery_curve_is_weekly_and_monotone(
    dashboard: dict,
    path,
    model: LearnerModel,
) -> None:
    """掌握度曲线：第 0 周为当前状态，之后按周采样且单调不减。"""
    curve = dashboard["mastery_curve"]
    assert curve, "掌握度曲线不能为空"
    assert curve[0]["week"] == 0
    assert curve[0]["avg_mastery"] == pytest.approx(
        sum(float(model.mastery_of(kp)) for kp in model.target_closure)
        / len(model.target_closure),
        abs=1e-3,
    )
    assert [point["week"] for point in curve] == sorted(point["week"] for point in curve)
    if path.steps:
        # 曲线覆盖 W0..W(路径总周数)：共 weeks+1 个点
        max_week = max(
            max(step.week for step in path.steps),
            int(getattr(path, "weeks", 0) or 0),
        )
        assert len(curve) == max_week + 1, (
            f"路径跨 {max_week} 周，曲线却有 {len(curve)} 个点"
        )
        assert curve[-1]["week"] == max_week
    values = [point["avg_mastery"] for point in curve]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1] + 1e-9, "掌握度推进曲线必须单调不减"
    mastered = [point["mastered"] for point in curve]
    assert mastered == sorted(mastered), "已掌握知识点数也应单调不减"
    for point in curve:
        assert 0.0 <= point["avg_mastery"] <= 1.0


def test_domain_radar(dashboard: dict) -> None:
    """领域雷达：覆盖全部 7 个领域，掌握度与目标线都在 [0,1]。"""
    radar = dashboard["domain_radar"]
    assert len(radar) == len(DOMAINS)
    assert {row["domain"] for row in radar} == set(DOMAINS)
    for row in radar:
        assert set(row) == {"domain", "mastery", "target"}
        assert 0.0 <= row["mastery"] <= 1.0
        assert 0.0 <= row["target"] <= 1.0


def test_difficulty_distribution(dashboard: dict, path) -> None:
    """难度分布：五个等级齐全，计数之和等于路径课程数。"""
    rows = dashboard["difficulty_distribution"]
    assert [row["level"] for row in rows] == [1, 2, 3, 4, 5]
    assert sum(row["count"] for row in rows) == len(path.steps)


def test_modality_mix(dashboard: dict, path) -> None:
    """模态配比：四种模态齐全，课时之和等于路径总时长。"""
    rows = dashboard["modality_mix"]
    assert {row["modality"] for row in rows} == set(MODALITIES)
    for row in rows:
        assert row["name"]
        assert row["hours"] >= 0.0
    total = sum(row["hours"] for row in rows)
    assert total == pytest.approx(path.total_hours, abs=0.02)


def test_time_allocation(dashboard: dict, path, model: LearnerModel) -> None:
    """每周时间分配：周次与周计划一致，``planned`` 等于每周预算。"""
    rows = dashboard["time_allocation"]
    assert len(rows) == len(path.weekly_plan)
    budget = model.learner.time_budget_hours_per_week
    for row, plan_row in zip(rows, path.weekly_plan):
        assert row["week"] == plan_row["week"]
        assert row["hours"] == pytest.approx(plan_row["hours"], abs=0.01)
        assert row["planned"] == pytest.approx(budget, abs=0.01)


def test_completion_forecast(dashboard: dict, path) -> None:
    """完成预测：与路径等长（或为空），概率与风险等级合法。"""
    rows = dashboard["completion_forecast"]
    if not path.steps:
        assert rows == []
        return
    assert len(rows) == len(path.steps)
    for row in rows:
        assert set(row) >= {"order", "course_id", "course_title", "p_completion", "risk_level"}
        assert 0.0 <= row["p_completion"] <= 1.0
        assert row["risk_level"] in {"低", "中", "高"}
    assert [row["order"] for row in rows] == [step.order for step in path.steps]


def test_gap_table(dashboard: dict, model: LearnerModel) -> None:
    """缺口表：按缺口降序、掌握度低于阈值、字段完整。"""
    rows = dashboard["gap_table"]
    gaps = model.knowledge_gap()
    assert len(rows) == min(len(gaps), 12)
    values = [row["gap"] for row in rows]
    assert values == sorted(values, reverse=True)
    for row in rows:
        assert set(row) >= {"kp_id", "name", "mastery", "difficulty", "domain", "in_path"}
        assert row["mastery"] < MASTERY_GAP
        assert row["mastery"] + row["gap"] == pytest.approx(
            float(model.required_mastery), abs=1e-6
        )
        assert 1 <= row["difficulty"] <= 5
        assert isinstance(row["in_path"], bool)


def test_history_activity_is_real_aggregation(
    dashboard: dict,
    dataset: Dataset,
    model: LearnerModel,
) -> None:
    """历史活动必须是**真实交互记录的周聚合**（总时长与记录逐条相加一致）。"""
    rows = dashboard["history_activity"]
    records = [r for r in dataset.interactions if r.learner_id == model.learner.learner_id]
    if not records:
        assert rows == []
        return
    expected_hours = sum(float(r.hours_spent) for r in records)
    expected_completed = sum(1 for r in records if r.completed)
    assert sum(row["hours"] for row in rows) == pytest.approx(expected_hours, abs=0.05)
    assert sum(row["completed"] for row in rows) == expected_completed
    assert [row["week"] for row in rows] == sorted(row["week"] for row in rows)
    assert rows[0]["week"] == 1


# ============================================================
# 无路径 / 外部预测 / 独立函数
# ============================================================


def test_dashboard_without_path(dataset: Dataset, graph: KnowledgeGraph, model: LearnerModel, planner: PathPlanner) -> None:
    """路径为空时不应崩溃，退化为历史数据视图。"""
    empty = planner.rebuild(model, [], "balanced")
    payload = build_dashboard(dataset, graph, model, empty)
    for key in REQUIRED_KEYS:
        assert key in payload
    assert payload["completion_forecast"] == []
    assert payload["time_allocation"] == []
    assert payload["meta"]["has_path"] is False
    # 难度分布退化为历史课程
    assert sum(row["count"] for row in payload["difficulty_distribution"]) >= 0


def test_dashboard_with_external_predictions(
    dataset: Dataset,
    graph: KnowledgeGraph,
    model: LearnerModel,
    path,
) -> None:
    """外部传入预测序列时应被直接采用（``source="provided"``）。"""
    predictions = [(0.9, "低") for _ in path.steps]
    payload = build_dashboard(dataset, graph, model, path, predictions=predictions)
    rows = payload["completion_forecast"]
    assert len(rows) == len(path.steps)
    for row in rows:
        assert row["p_completion"] == pytest.approx(0.9)
        assert row["risk_level"] == "低"
        assert row["source"] == "provided"
    assert payload["meta"]["forecast_source"] == "provided"


def test_standalone_helpers(dataset: Dataset, graph: KnowledgeGraph, model: LearnerModel, path) -> None:
    """各独立聚合函数的入参、出参形状正确。"""
    records = [r for r in dataset.interactions if r.learner_id == model.learner.learner_id]
    assert mastery_curve(graph, model, path)[0]["week"] == 0
    assert len(domain_radar(model, model.goal)) == len(DOMAINS)
    assert len(difficulty_distribution(dataset, path, records)) == 5
    assert len(modality_mix(dataset, path, records)) == len(MODALITIES)
    assert isinstance(time_allocation(model, path), list)
    assert isinstance(gap_table(graph, model, path), list)
    assert isinstance(history_activity(dataset, records), list)
    assert learning_streak(dataset, records) >= 0
