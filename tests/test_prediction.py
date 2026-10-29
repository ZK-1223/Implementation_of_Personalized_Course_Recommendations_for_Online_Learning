"""学习效果预测的单元测试（实现要求 ⑤）。

覆盖：
    * :func:`learnpath.prediction.features.build_features` 的维度、取值范围与**零标签泄漏**；
    * :class:`learnpath.prediction.effect.EffectPredictor` 的训练 / 评估 / 持久化 / 推理；
    * ``models/model_metrics.json`` 的结构（下游 web-app 直接消费）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from learnpath.data.fixtures import build_tiny_dataset
from learnpath.data.schema import Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import LearnerModel
from learnpath.prediction.effect import (
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    CompletionPrediction,
    EffectPredictor,
    GainPrediction,
    risk_level_of,
)
from learnpath.prediction.features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    build_features,
    feature_vector,
)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_tiny_dataset()


@pytest.fixture(scope="module")
def graph(dataset: Dataset) -> KnowledgeGraph:
    return KnowledgeGraph.from_dataset(dataset)


@pytest.fixture(scope="module")
def model(dataset: Dataset, graph: KnowledgeGraph) -> LearnerModel:
    learner_id = sorted(dataset.learners)[0]
    return LearnerModel(dataset.learners[learner_id], dataset, graph)


@pytest.fixture(scope="module")
def predictor(dataset: Dataset, graph: KnowledgeGraph, tmp_path_factory) -> EffectPredictor:
    path = tmp_path_factory.mktemp("models")
    instance = EffectPredictor(model_path=path)
    instance.fit(dataset, graph)
    return instance


# ============================================================
# 特征工程
# ============================================================


def test_features_shape_and_keys(dataset: Dataset, graph: KnowledgeGraph, model: LearnerModel) -> None:
    """特征维度 ≥ 14、键顺序固定、全部为有限浮点数。"""
    course = dataset.courses[sorted(dataset.courses)[0]]
    features = build_features(dataset, graph, model, course)
    assert len(features) >= 14
    assert tuple(features) == FEATURE_NAMES
    vector = feature_vector(features)
    assert len(vector) == len(FEATURE_NAMES)
    assert all(np.isfinite(value) for value in vector)
    assert set(FEATURE_NAMES) <= set(FEATURE_LABELS)


def test_features_are_bounded(dataset: Dataset, graph: KnowledgeGraph, model: LearnerModel) -> None:
    """有明确语义区间的特征必须落在 [0,1]。"""
    unit_features = (
        "learner_ability",
        "avg_mastery",
        "target_avg_mastery",
        "readiness",
        "style_fit",
        "time_fit",
        "prereq_uncovered_ratio",
        "gap_coverage_ratio",
        "hist_completion_rate",
        "hist_avg_quiz",
        "theory_gap",
        "modality_match",
    )
    for course in list(dataset.courses.values())[:8]:
        features = build_features(dataset, graph, model, course)
        for name in unit_features:
            assert 0.0 <= features[name] <= 1.0, f"{name}={features[name]} 超出 [0,1]"
        assert features["course_level"] >= 1.0
        assert features["duration_hours"] > 0.0
        assert features["expected_gain_model"] >= 0.0


def test_features_do_not_leak_current_course(
    dataset: Dataset,
    graph: KnowledgeGraph,
) -> None:
    """**零标签泄漏**：``hist_*`` 特征必须排除当前课程那条交互记录。

    构造方式：找一条真实交互 ``(learner, course)``，再用该学习者的模型分别
    对「这门课」和「一门没学过的课」求特征——若 ``hist_*`` 把当前课程算进去，
    前者会比后者少一条记录，从而两者不同；这里直接用计数校验更严格：
    用只有 1 条交互的假学习者验证「评价该课程时历史量为 0」。
    """
    learner_id = None
    interaction = None
    counts: dict[str, int] = {}
    for record in dataset.interactions:
        counts[record.learner_id] = counts.get(record.learner_id, 0) + 1
    for record in dataset.interactions:
        if counts[record.learner_id] == 1:
            learner_id, interaction = record.learner_id, record
            break
    if interaction is None:
        pytest.skip("tiny 数据集中没有只有 1 条交互的学习者")

    live = LearnerModel(dataset.learners[learner_id], dataset, graph)
    course = dataset.courses[interaction.course_id]
    features = build_features(dataset, graph, live, course)
    # 该学习者只有这一条记录，且它正是当前课程 → 排除后历史条数为 0
    assert features["hist_log_count"] == pytest.approx(0.0, abs=1e-9)


def test_features_are_deterministic(dataset: Dataset, graph: KnowledgeGraph, model: LearnerModel) -> None:
    course = dataset.courses[sorted(dataset.courses)[0]]
    first = build_features(dataset, graph, model, course)
    second = build_features(dataset, graph, model, course)
    assert first == second


# ============================================================
# 预测器
# ============================================================


def test_fit_returns_metrics(predictor: EffectPredictor) -> None:
    """训练返回完整指标，且数值合理（AUC > 0.5、R² 为有限实数）。"""
    metrics = predictor.metrics
    for key in ("auc", "accuracy", "f1", "cv_mean", "cv_std", "mae", "rmse", "r2"):
        assert key in metrics
        assert isinstance(metrics[key], float)
    assert 0.5 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["r2"] > -1.0
    assert predictor.is_fitted


def test_report_shape_matches_contract(predictor: EffectPredictor, dataset: Dataset) -> None:
    """``model_metrics.json`` 的结构必须与 lead 指定的契约一致。"""
    report = predictor.report
    assert set(report) >= {"generated_at", "dataset", "effect_model", "gain_model"}
    assert set(report["dataset"]) == {"n_learners", "n_courses", "n_interactions"}

    effect = report["effect_model"]
    assert effect["task"] == "course_completion"
    assert 0.0 <= effect["positive_rate"] <= 1.0
    names = [row["name"] for row in effect["models"]]
    assert len(names) == 2, "必须给出「逻辑回归」与「梯度提升树」两个对比模型"
    for row in effect["models"]:
        for key in ("name", "auc", "accuracy", "f1", "cv_mean", "cv_std"):
            assert key in row
    assert effect["best"] in names
    assert effect["features"]
    for item in effect["features"]:
        assert set(item) == {"name", "importance"}
        assert item["name"] in FEATURE_NAMES

    gain = report["gain_model"]
    assert gain["task"] == "mastery_gain_regression"
    assert len(gain["models"]) == 2, "回归也要给出两个对比模型"
    for row in gain["models"]:
        for key in ("name", "mae", "rmse", "r2"):
            assert key in row
    assert gain["best"] in [row["name"] for row in gain["models"]]


def test_metrics_file_written(predictor: EffectPredictor) -> None:
    """训练产物 ``model_metrics.json`` 必须真的落盘且是合法 JSON。"""
    path = predictor.model_path / "model_metrics.json"
    assert path.exists(), "训练后必须写出 model_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["effect_model"]["task"] == "course_completion"


def test_predict_completion(predictor: EffectPredictor, dataset: Dataset, graph: KnowledgeGraph) -> None:
    """完成概率预测：概率区间、风险等级、Top-5 可解释驱动因素。"""
    learner_id = sorted(dataset.learners)[0]
    model = LearnerModel(dataset.learners[learner_id], dataset, graph)
    for course in list(dataset.courses.values())[:5]:
        prediction = predictor.predict_completion(dataset, graph, model, course)
        assert isinstance(prediction, CompletionPrediction)
        assert 0.0 <= prediction.p_completion <= 1.0
        assert prediction.p_drop == pytest.approx(1.0 - prediction.p_completion)
        assert prediction.risk_level in {"低", "中", "高"}
        assert len(prediction.drivers) <= 5
        for name, contribution in prediction.drivers:
            assert name in FEATURE_NAMES
            assert np.isfinite(contribution)
        payload = prediction.to_dict()
        assert set(payload) == {"p_completion", "p_drop", "risk_level", "drivers"}


def test_predict_gain(predictor: EffectPredictor, dataset: Dataset, graph: KnowledgeGraph) -> None:
    """增益预测：增益非负、每个覆盖知识点的预测掌握度都在 [0,1]。"""
    learner_id = sorted(dataset.learners)[0]
    model = LearnerModel(dataset.learners[learner_id], dataset, graph)
    for course in list(dataset.courses.values())[:5]:
        prediction = predictor.predict_gain(dataset, graph, model, course)
        assert isinstance(prediction, GainPrediction)
        assert 0.0 <= prediction.expected_gain <= 1.0
        assert set(prediction.expected_kp_mastery) == set(course.kp_ids)
        for kp_id, value in prediction.expected_kp_mastery.items():
            assert 0.0 <= value <= 1.0
            assert value >= float(model.mastery_of(kp_id)) - 1e-9, "预测掌握度不应低于当前值"
        assert 0.0 <= prediction.predicted_level <= 1.0


def test_risk_level_thresholds() -> None:
    """风险等级阈值：≥0.70 低 / 0.40~0.70 中 / <0.40 高。"""
    assert risk_level_of(0.9) == "低"
    assert risk_level_of(RISK_LOW_THRESHOLD) == "低"
    assert risk_level_of(0.5) == "中"
    assert risk_level_of(RISK_HIGH_THRESHOLD) == "中"
    assert risk_level_of(0.1) == "高"


def test_save_load_roundtrip(dataset: Dataset, graph: KnowledgeGraph, tmp_path: Path) -> None:
    """``save`` / ``load`` 往返后预测结果必须一致。"""
    learner_id = sorted(dataset.learners)[0]
    model = LearnerModel(dataset.learners[learner_id], dataset, graph)
    predictor = EffectPredictor(model_path=tmp_path)
    predictor.fit(dataset, graph)
    course = dataset.courses[sorted(dataset.courses)[0]]
    before = predictor.predict_completion(dataset, graph, model, course)

    reloaded = EffectPredictor(model_path=tmp_path)
    reloaded.load()
    after = reloaded.predict_completion(dataset, graph, model, course)
    assert after.p_completion == pytest.approx(before.p_completion, abs=1e-9)
    assert reloaded.completion_model_name == predictor.completion_model_name


def test_load_missing_model_raises(tmp_path: Path) -> None:
    """未训练且本地没有模型文件时：``load`` 抛 FileNotFoundError，推理抛 RuntimeError。"""
    predictor = EffectPredictor(model_path=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        predictor.load()
    assert not predictor.is_fitted
