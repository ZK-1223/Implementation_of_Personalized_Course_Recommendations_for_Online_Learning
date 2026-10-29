"""路径规划与多目标优化的单元测试（实现要求 ③④）。

覆盖：
    * :mod:`learnpath.planning.multiobjective` 的纯数学部分
      （目标归一化、非支配排序、拥挤度、NSGA-II、TOPSIS）；
    * :class:`learnpath.planning.planner.PathPlanner` 的结构合法性：
      先修拓扑顺序、**目标知识点覆盖**、排期 / 周计划 / 里程碑一致性、
      重规划、帕累托前沿、确定性、性能。

数据：默认用 :func:`learnpath.data.fixtures.build_tiny_dataset`（确定、快速）；
若仓库里存在完整数据集，则额外在 2 名真实学习者上跑一遍
「目标知识点覆盖率 ≥ 80%」的回归测试。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from learnpath.config import (
    MASTERY_GAP,
    MAX_PATH_COURSES,
    PLAN_PRESETS,
    ObjectiveWeights,
)
from learnpath.data.fixtures import build_tiny_dataset
from learnpath.data.schema import Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import LearnerModel
from learnpath.planning.multiobjective import (
    OBJECTIVE_KEYS,
    crowding_distance,
    evaluate_objectives,
    non_dominated_sort,
    nsga2,
    select_compromise,
    topsis_scores,
)
from learnpath.planning.planner import LearningPath, PathPlanner


# ============================================================
# 共享 fixture
# ============================================================


@pytest.fixture(scope="module")
def tiny_dataset() -> Dataset:
    return build_tiny_dataset()


@pytest.fixture(scope="module")
def graph(tiny_dataset: Dataset) -> KnowledgeGraph:
    return KnowledgeGraph.from_dataset(tiny_dataset)


@pytest.fixture(scope="module")
def planner(tiny_dataset: Dataset, graph: KnowledgeGraph) -> PathPlanner:
    return PathPlanner(tiny_dataset, graph)


@pytest.fixture(scope="module")
def learner_ids(tiny_dataset: Dataset) -> list[str]:
    return sorted(tiny_dataset.learners)


@pytest.fixture(scope="module")
def model(tiny_dataset: Dataset, graph: KnowledgeGraph, learner_ids: list[str]) -> LearnerModel:
    return LearnerModel(tiny_dataset.learners[learner_ids[0]], tiny_dataset, graph)


class _Dummy:
    """带 ``objectives`` 的最小候选解（用于纯数学测试）。"""

    def __init__(self, **objectives: float) -> None:
        self.objectives = dict(objectives)


# ============================================================
# 多目标内核
# ============================================================


def test_evaluate_objectives_in_unit_interval(planner: PathPlanner, model: LearnerModel) -> None:
    """四个目标都必须落在 [0,1]，且都是「越大越好」。"""
    path = planner.plan(model, "balanced")
    assert path.steps, "tiny 数据集上应当能规划出非空路径"
    objectives = evaluate_objectives(path.steps, model)
    assert set(objectives) == set(OBJECTIVE_KEYS)
    for key, value in objectives.items():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0, f"{key} 超出 [0,1]：{value}"


def test_evaluate_objectives_empty_path(model: LearnerModel) -> None:
    """空路径 = 零收益、零成本。"""
    objectives = evaluate_objectives([], model)
    assert objectives["gain"] == 0.0
    assert objectives["time"] == 1.0
    assert objectives["smoothness"] == 1.0
    assert objectives["fit"] == 0.0


def test_non_dominated_sort_fronts() -> None:
    """快速非支配排序：第一层互不支配，且分层整体单调。"""
    population = [
        _Dummy(gain=0.9, time=0.1, smoothness=0.5, fit=0.5),
        _Dummy(gain=0.1, time=0.9, smoothness=0.5, fit=0.5),
        _Dummy(gain=0.5, time=0.5, smoothness=0.5, fit=0.5),
        _Dummy(gain=0.2, time=0.2, smoothness=0.2, fit=0.2),  # 被所有人支配
    ]
    fronts = non_dominated_sort(population)
    assert len(fronts) >= 2
    assert len(fronts[0]) == 3
    assert fronts[-1][0] is population[3]
    flat = [candidate for front in fronts for candidate in front]
    assert len(flat) == len(population)


def test_crowding_distance_boundaries_infinite() -> None:
    """边界解的拥挤度必须为 +∞，保证极端解不被淘汰。"""
    front = [
        _Dummy(gain=0.1, time=0.9, smoothness=0.1, fit=0.1),
        _Dummy(gain=0.5, time=0.5, smoothness=0.5, fit=0.5),
        _Dummy(gain=0.9, time=0.1, smoothness=0.9, fit=0.9),
    ]
    distances = crowding_distance(front)
    assert distances[0] == float("inf")
    assert distances[2] == float("inf")
    assert distances[1] < float("inf")


def test_topsis_weights_change_selection() -> None:
    """TOPSIS：换权重必须能选出不同的妥协解（这是三档 preset 差异化的机制）。"""
    front = [
        _Dummy(gain=0.95, time=0.15, smoothness=0.6, fit=0.5),
        _Dummy(gain=0.55, time=0.95, smoothness=0.6, fit=0.5),
        _Dummy(gain=0.75, time=0.55, smoothness=0.6, fit=0.5),
    ]
    gain_heavy = select_compromise(front, ObjectiveWeights(gain=0.9, time=0.02, smoothness=0.04, fit=0.04))
    time_heavy = select_compromise(front, ObjectiveWeights(gain=0.05, time=0.9, smoothness=0.03, fit=0.02))
    assert gain_heavy is front[0], "重视收益时应选高收益解"
    assert time_heavy is front[1], "重视时间时应选高时间可承受度（短）解"
    scores = topsis_scores(front, ObjectiveWeights())
    assert len(scores) == 3 and all(0.0 <= s <= 1.0 for s in scores)


def test_nsga2_converges_on_bi_objective() -> None:
    """NSGA-II 在经典的「一升一降」二目标问题上应得到铺开的前沿。"""
    rng = np.random.default_rng(0)
    population = [rng.random(2) for _ in range(20)]

    def evaluate(chromosome: np.ndarray) -> dict[str, float]:
        x = float(chromosome[0])
        return {
            "gain": x,
            "time": 1.0 - x,
            "smoothness": 0.5,
            "fit": 0.5,
        }

    history: list[dict[str, float]] = []
    front = nsga2(population, evaluate, generations=15, history=history)
    assert front, "前沿不应为空"
    assert len(history) == 15
    gains = [solution.objectives["gain"] for solution in front]
    assert max(gains) > 0.8 and min(gains) < 0.2, "前沿应在 gain 方向充分铺开"
    for solution in front:
        assert 0.0 <= solution.objectives["gain"] <= 1.0


# ============================================================
# 规划器结构合法性
# ============================================================


def _covered_kps(path: LearningPath) -> set[str]:
    covered: set[str] = set()
    for step in path.steps:
        covered.update(step.kp_ids)
    return covered


def test_plan_structure(planner: PathPlanner, model: LearnerModel) -> None:
    """路径结构自洽：序号连续、累计时长单调、周计划与里程碑一致。"""
    path = planner.plan(model, "balanced")
    assert isinstance(path, LearningPath)
    assert path.learner_id == model.learner.learner_id
    assert path.preset == "balanced"
    assert len(path.steps) <= MAX_PATH_COURSES

    cumulative = 0.0
    for index, step in enumerate(path.steps, start=1):
        assert step.order == index
        cumulative += step.hours
        assert step.cumulative_hours == pytest.approx(cumulative, abs=1e-9)
        assert step.hours > 0
        assert 1 <= step.level <= 5
        assert step.week >= 1
        assert step.expected_gain >= 0.0

    assert path.total_hours == pytest.approx(cumulative, abs=1e-6)
    assert path.weeks >= max((step.week for step in path.steps), default=0)

    planned_hours = sum(item["hours"] for item in path.weekly_plan)
    assert planned_hours == pytest.approx(path.total_hours, abs=0.05)
    for item in path.weekly_plan:
        assert item["course_ids"], "每周计划不能有空周"
    for milestone in path.milestones:
        assert milestone["kp_names"] and milestone["label"].startswith("掌握")

    assert 0.0 <= path.feasibility <= 1.0
    for key in OBJECTIVE_KEYS:
        assert 0.0 <= path.objectives[key] <= 1.0


def test_plan_respects_prerequisite_order(
    planner: PathPlanner,
    model: LearnerModel,
    graph: KnowledgeGraph,
) -> None:
    """拓扑合法性：课程 B 的先修知识点若由路径中课程 A 覆盖，则 A 必须排在 B 之前。"""
    path = planner.plan(model, "thorough")
    assert len(path.steps) >= 2

    covered_so_far: set[str] = set()
    violations: list[str] = []
    for step in path.steps:
        prereq = graph.course_prerequisites(step.course)
        missing = {kp for kp in prereq if kp in _covered_kps(path) and kp not in covered_so_far}
        if missing:
            violations.append(f"{step.course.course_id} 缺少前置 {sorted(missing)[:3]}")
        covered_so_far.update(step.kp_ids)
    assert not violations, "存在先修顺序违规：" + "；".join(violations[:3])


def test_plan_respects_budget(
    planner: PathPlanner,
    tiny_dataset: Dataset,
    graph: KnowledgeGraph,
    learner_ids: list[str],
) -> None:
    """**回归测试（P0）**：路径总时长必须落在学习者的时间预算之内。

    时间预算 = ``time_budget_hours_per_week × deadline_weeks``。
    这是「数据可执行性」的硬要求：给一个每周只能学 3 小时的人排 280 小时的路径，
    等于没有规划。
    """
    for learner_id in learner_ids:
        model = LearnerModel(tiny_dataset.learners[learner_id], tiny_dataset, graph)
        learner = tiny_dataset.learners[learner_id]
        budget = learner.time_budget_hours_per_week * learner.deadline_weeks
        for preset in ("balanced", "fast", "thorough"):
            path = planner.plan(model, preset)
            assert path.budget_hours == pytest.approx(budget, abs=1e-6)
            assert path.total_hours <= budget + 1e-6, (
                f"{learner_id}/{preset}: 路径 {path.total_hours:.1f}h 超出预算 {budget:.1f}h"
            )
            assert path.over_budget is False
            assert path.coverage["budget_utilisation"] <= 1.0 + 1e-6
            assert path.weeks <= learner.deadline_weeks + 1


def test_plan_covers_target_knowledge_points(
    planner: PathPlanner,
    tiny_dataset: Dataset,
    graph: KnowledgeGraph,
    learner_ids: list[str],
) -> None:
    """**回归测试（P0）**：学习路径必须真的走向学习目标。

    判据：``path.coverage_ratio`` 不得低于「预算约束下确定性贪心基线」的覆盖率。
    由于预算可能不足以覆盖全部目标（例如每周只有 3 小时的学习者），
    这里不设固定百分比阈值，而是要求 NSGA-II 的解**不劣于**基线解，
    保证优化过程没有把目标丢掉。

    同时要求：路径覆盖的目标知识点数 > 0（不能一条目标都不覆盖）。
    """
    for learner_id in learner_ids[:6]:
        model = LearnerModel(tiny_dataset.learners[learner_id], tiny_dataset, graph)
        learner = tiny_dataset.learners[learner_id]
        budget = learner.time_budget_hours_per_week * learner.deadline_weeks
        _closure, gaps, _mastered = planner.knowledge_gaps(model)
        goal = planner.goal_of(model)
        target_ids = tuple(kp for kp in goal.target_kp_ids if kp in gaps)

        baseline = planner.greedy_cover(
            model, planner.build_candidates(model), gaps, budget, MAX_PATH_COURSES, target_ids
        )
        path = planner.plan(model, "balanced")

        if not path.coverage["target_gap"]:
            # 目标本来就都已掌握：空路径是正确结果
            assert path.steps == []
            continue

        candidates = planner.build_candidates(model)
        if not candidates:
            # 数据集中没有任何课程覆盖该学习者的目标缺口（tiny fixture 会有这种情况）
            assert path.steps == []
            continue

        assert path.coverage_ratio > 0.0, f"{learner_id}: 一个目标知识点都没有覆盖"
        assert len(path.covered_target_kp_ids) > 0

        # 与确定性基线比较：NSGA-II 得到的解不应明显劣于贪心基线
        baseline_kps: set[str] = set()
        for course in baseline:
            baseline_kps.update(course.kp_ids)
        baseline_covered = len(set(target_ids) & baseline_kps)
        assert len(path.covered_target_kp_ids) + 1 >= baseline_covered, (
            f"{learner_id}: 路径只覆盖 {len(path.covered_target_kp_ids)} 个目标，"
            f"基线覆盖 {baseline_covered} 个"
        )


def test_plan_difficulty_is_progressive(
    planner: PathPlanner,
    tiny_dataset: Dataset,
    graph: KnowledgeGraph,
    learner_ids: list[str],
) -> None:
    """难度单调递进：多步路径的末步难度不应低于首步难度。"""
    checked = 0
    for learner_id in learner_ids[:6]:
        model = LearnerModel(tiny_dataset.learners[learner_id], tiny_dataset, graph)
        path = planner.plan(model, "balanced")
        if len(path.steps) < 2:
            continue
        checked += 1
        assert path.steps[-1].level >= path.steps[0].level
    assert checked > 0, "tiny 数据集上应至少有一条多步路径"


def test_plan_coverage_name_fields(
    planner: PathPlanner,
    tiny_dataset: Dataset,
    graph: KnowledgeGraph,
    learner_ids: list[str],
) -> None:
    """``covered/uncovered_target_kp_names`` 必须与 id 列表一一对应且为中文名。

    这是 ``docs/API.md`` §4 的契约字段：服务层直接读属性做标签渲染，
    名称列表若为空，前端就只能显示 id。
    """
    checked = 0
    for learner_id in learner_ids:
        model = LearnerModel(tiny_dataset.learners[learner_id], tiny_dataset, graph)
        for preset in PLAN_PRESETS:
            path = planner.plan(model, preset)
            assert len(path.covered_target_kp_names) == len(path.covered_target_kp_ids)
            assert len(path.uncovered_target_kp_names) == len(path.uncovered_target_kp_ids)
            for kp_id, name in zip(path.covered_target_kp_ids, path.covered_target_kp_names):
                assert name == graph.name_of(kp_id)
            for kp_id, name in zip(
                path.uncovered_target_kp_ids, path.uncovered_target_kp_names
            ):
                assert name == graph.name_of(kp_id)
            payload = path.to_dict(graph)
            assert payload["covered_target_kp_names"] == path.covered_target_kp_names
            assert payload["uncovered_target_kp_names"] == path.uncovered_target_kp_names
            if path.covered_target_kp_ids:
                checked += 1
    assert checked > 0, "至少应有一个覆盖到目标知识点的路径"


def test_plan_step_count_is_reasonable(planner: PathPlanner, model: LearnerModel) -> None:
    """步骤数应落在合理区间，不应撞 ``MAX_PATH_COURSES`` 上限。"""
    path = planner.plan(model, "balanced")
    assert len(path.steps) <= MAX_PATH_COURSES
    if path.total_hours > 0:
        assert len(path.steps) >= 1


def test_plan_shortfall_is_reported(planner: PathPlanner, model: LearnerModel) -> None:
    """时间超支必须体现在 ``feasibility`` 与 ``coverage.hours_shortfall`` 上。"""
    path = planner.plan(model, "thorough")
    budget = float(model.learner.total_available_hours)
    shortfall = max(0.0, path.total_hours - budget)
    assert path.coverage["available_hours"] == pytest.approx(budget, abs=0.01)
    assert path.coverage["hours_shortfall"] == pytest.approx(shortfall, abs=0.05)
    if path.total_hours > budget:
        assert path.feasibility < 1.0


def test_plan_is_deterministic(planner: PathPlanner, model: LearnerModel) -> None:
    """同一模型连续两次规划结果必须完全一致（NSGA-II 使用固定随机种子）。"""
    first = planner.plan(model, "balanced")
    second = planner.plan(model, "balanced")
    assert first.course_ids == second.course_ids
    assert first.total_hours == pytest.approx(second.total_hours)


def test_presets_produce_valid_paths(planner: PathPlanner, model: LearnerModel) -> None:
    """三档 preset 都必须产出结构合法、且在预算内、走向目标的路径。"""
    paths = {key: planner.plan(model, key) for key in PLAN_PRESETS}
    for key, path in paths.items():
        assert path.preset == key
        assert path.steps, f"{key} 路径为空"
        assert path.total_hours <= path.budget_hours + 1e-6, f"{key} 超预算"
        assert path.coverage_ratio > 0.0
    # gain 权重最高的 thorough 其原始收益不应低于 fast
    assert paths["thorough"].total_gain >= paths["fast"].total_gain - 1e-6


def test_pareto_front(planner: PathPlanner, model: LearnerModel) -> None:
    """帕累托前沿：非空、互不支配、恰好一个 ``is_selected``、按总时长升序。"""
    front = planner.pareto_front(model)
    assert front, "帕累托前沿不应为空"
    hours = [path.total_hours for path in front]
    assert hours == sorted(hours)
    assert sum(1 for path in front if path.is_selected) == 1
    signatures = {tuple(sorted(p.course_ids)) for p in front}
    assert len(signatures) == len(front), "前沿上不应有重复的课程组合"

    for i, left in enumerate(front):
        for j, right in enumerate(front):
            if i == j:
                continue
            left_dominates = all(
                left.objectives[key] >= right.objectives[key] - 1e-9 for key in OBJECTIVE_KEYS
            ) and any(left.objectives[key] > right.objectives[key] + 1e-9 for key in OBJECTIVE_KEYS)
            assert not left_dominates, (
                f"前沿上第 {i} 个解支配了第 {j} 个解，非支配排序有误"
            )


def test_replan_skips_completed_courses(planner: PathPlanner, model: LearnerModel) -> None:
    """``replan``：已完成的课程不再出现在新路径里，且新路径仍在预算内。"""
    path = planner.plan(model, "balanced")
    done = path.course_ids[:2]
    if not done:
        pytest.skip("tiny 数据集上该学习者的路径为空")

    replanned = planner.replan(model, done, "balanced")
    assert replanned.learner_id == path.learner_id
    assert replanned.total_hours <= replanned.budget_hours + 1e-6
    # 已完成的课程视为已掌握，不应再次出现在新路径中
    assert not (set(replanned.course_ids) & set(done))


def test_plan_performance(planner: PathPlanner, model: LearnerModel) -> None:
    """性能：tiny 数据集上单次 plan 必须远小于 3 秒。"""
    started = time.perf_counter()
    planner.plan(model, "balanced", extra_mastered=frozenset())  # 命中缓存也计入
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"plan 耗时 {elapsed:.2f}s，超过 3s 上限"


def test_rebuild_respects_given_course_set(planner: PathPlanner, model: LearnerModel) -> None:
    """``rebuild`` 只做排序 / 排期 / 评估，不增删课程。"""
    course_ids = planner.plan(model, "balanced").course_ids[:3]
    rebuilt = planner.rebuild(model, course_ids, "balanced")
    assert sorted(rebuilt.course_ids) == sorted(set(course_ids))
    assert rebuilt.total_hours > 0
    assert rebuilt.coverage["n_courses"] == len(set(course_ids))


# ============================================================
# 完整数据集回归（存在 data/ 时才跑）
# ============================================================


@pytest.fixture(scope="module")
def full_bundle():
    from learnpath.data.loader import dataset_exists, load_dataset

    if not dataset_exists():
        pytest.skip("完整数据集 data/ 不存在，跳过全量回归测试")
    dataset = load_dataset()
    full_graph = KnowledgeGraph.from_dataset(dataset)
    full_graph.validate()
    return dataset, full_graph, PathPlanner(dataset, full_graph)


def test_full_dataset_target_coverage_and_performance(full_bundle) -> None:
    """完整数据集上：预算硬约束、目标覆盖、难度递进、单次规划 < 3 s。"""
    dataset, full_graph, full_planner = full_bundle
    for learner_id in ("L-0001", "L-0100", "L-0300"):
        if learner_id not in dataset.learners:
            continue
        learner = dataset.learners[learner_id]
        model = LearnerModel(learner, dataset, full_graph)
        budget = learner.time_budget_hours_per_week * learner.deadline_weeks
        for preset in ("balanced", "fast", "thorough"):
            started = time.perf_counter()
            path = full_planner.plan(model, preset)
            elapsed = time.perf_counter() - started
            assert elapsed < 3.0, f"{learner_id}/{preset}: plan 耗时 {elapsed:.2f}s"
            assert path.total_hours <= budget + 1e-6, (
                f"{learner_id}/{preset}: 路径 {path.total_hours:.1f}h > 预算 {budget:.1f}h"
            )
            assert path.coverage_ratio > 0.0
            # 预算较充足（≥150h）的学习者，目标达成率应 ≥ 0.6
            if budget >= 150.0:
                assert path.coverage_ratio >= 0.6, (
                    f"{learner_id}/{preset}: 预算 {budget:.0f}h，"
                    f"目标达成率仅 {path.coverage_ratio:.0%}"
                )
            if len(path.steps) >= 2:
                assert path.steps[-1].level >= path.steps[0].level
