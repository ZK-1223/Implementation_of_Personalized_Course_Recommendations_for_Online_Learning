"""学习分析仪表板（实现要求 ⑧「学习进度可视化与个性化学习报告」）。

入口
----
:func:`build_dashboard`
    ``build_dashboard(dataset, graph, model, path, recommendations=None, predictions=None)``
    返回的字典**严格等于** ``docs/API.md`` 第 7 节定义的九个字段：

    ``kpi / mastery_curve / domain_radar / difficulty_distribution / modality_mix /
    time_allocation / completion_forecast / gap_table / history_activity``

数据来源（全部是真实计算，无随机数、无占位值）
----------------------------------------------
==========================  ==============================================================
字段                         计算方式
==========================  ==============================================================
``kpi``                      KPI 卡片 9 项，见 :func:`_build_kpi`
``mastery_curve``            **基于路径逐步推进**：从当前掌握度出发，按路径顺序逐门课把
                             ``expected_gain`` 按覆盖权重分摊到知识点上，每步累加后统计
                             目标闭包平均掌握度与「已掌握知识点数」
``domain_radar``             ``LearnerModel.domain_mastery()`` 的真实领域均值 + 目标线
``difficulty_distribution``  路径课程的难度分布（无路径时退化为该学习者的历史课程）
``modality_mix``             路径各模态的课时占比（无路径时退化为历史）
``time_allocation``          路径 ``weekly_plan`` 的每周课时 vs 每周预算
``completion_forecast``      每门路径课程的完成概率与风险等级（来自 ``EffectPredictor``；
                             模型不可用时用解析式兜底，见 :func:`_analytic_completion`）
``gap_table``                ``LearnerModel.knowledge_gap()`` 的缺口明细 + 是否在路径内
``history_activity``         按**周**聚合该学习者的真实交互记录（投入时长 / 完成门数）
==========================  ==============================================================

``mastery_curve`` 与 ``history_activity`` 都刻意区分了「历史真实数据」与「路径预测」：
周次 0 是当前状态，之后每一周来自路径推进；``history_activity`` 则是过去真实的
交互聚合，两者共同构成「过去怎么样 / 未来会怎么样」的对照。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Iterable, Mapping, Sequence

from learnpath.config import (
    MASTERY_GAP,
    MASTERY_MASTERED,
    MIN_LEVEL,
    MAX_LEVEL,
    MODALITY_NAMES,
    MODALITIES,
    DOMAINS,
)
from learnpath.data.schema import Course, Dataset, Interaction
from learnpath.knowledge.graph import KnowledgeGraph

#: 风险等级的判定阈值（与 :mod:`learnpath.prediction.effect` 保持一致）
RISK_LOW_THRESHOLD: Final[float] = 0.70
RISK_HIGH_THRESHOLD: Final[float] = 0.40

#: 缺口表最多返回的知识点条数
GAP_TABLE_LIMIT: Final[int] = 12

#: 完成预测最多返回的课程条数
FORECAST_LIMIT: Final[int] = 40

#: 解析式完成概率的兜底权重（模型不可用时使用）
ANALYTIC_BASE: Final[float] = 0.12
ANALYTIC_READINESS: Final[float] = 0.40
ANALYTIC_TIME_FIT: Final[float] = 0.18
ANALYTIC_HISTORY: Final[float] = 0.30

_EPS: Final[float] = 1e-9


# ============================================================
# 入口
# ============================================================


def build_dashboard(
    dataset: Dataset,
    graph: KnowledgeGraph,
    model: Any,
    path: Any,
    recommendations: Sequence[Any] | None = None,
    predictions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """构建 ``GET /api/analytics/{learner_id}`` 的完整响应体。

    Args:
        dataset: 全量数据集。
        graph: 知识点图。
        model: :class:`~learnpath.learner.profile.LearnerModel`。
        path: :class:`~learnpath.planning.planner.LearningPath`（可为 ``None``，
            此时难度分布 / 模态 / 时间分配退化为历史数据）。
        recommendations: 可选，推荐结果列表（保留给前端做「推荐 vs 路径」对照，
            本函数不消费其内容）。
        predictions: 可选，与 ``path.steps`` **等长**的完成预测序列。元素可以是
            ``CompletionPrediction`` / ``dict`` / ``(p_completion, risk_level)`` 元组。
            为 ``None`` 时本函数会尝试加载 ``models/`` 下已训练好的
            :class:`~learnpath.prediction.effect.EffectPredictor`；仍不可用时
            退化为**解析式估计**（:func:`_analytic_completion`），并在
            ``completion_forecast[*].source`` 标注 ``"analytic"``。

    Returns:
        含 ``learner_id`` 与九个字段的字典，可直接 JSON 序列化。
    """
    learner = getattr(model, "learner", None)
    learner_id = str(getattr(learner, "learner_id", ""))
    goal = getattr(model, "goal", None)
    if goal is None:  # pragma: no cover - 契约保证 LearnerModel.goal 存在
        goal = dataset.goals[dataset.learners[learner_id].goal_id]

    target_kps: tuple[str, ...] = tuple(getattr(model, "target_kps", goal.target_kp_ids))
    steps = list(getattr(path, "steps", []) or [])
    closure = graph.prerequisite_closure(target_kps)

    records = [r for r in dataset.interactions if r.learner_id == learner_id]
    forecast, forecast_source = _completion_forecast(
        dataset, graph, model, steps, predictions
    )

    return {
        "learner_id": learner_id,
        "kpi": _build_kpi(dataset, graph, model, path, goal, closure, target_kps, records, forecast),
        "mastery_curve": mastery_curve(graph, model, path, closure, target_kps),
        "domain_radar": domain_radar(model, goal),
        "difficulty_distribution": difficulty_distribution(dataset, path, records),
        "modality_mix": modality_mix(dataset, path, records),
        "time_allocation": time_allocation(model, path),
        "completion_forecast": forecast,
        "gap_table": gap_table(graph, model, path),
        "history_activity": history_activity(dataset, records),
        "meta": {
            "has_path": bool(steps),
            "n_records": len(records),
            "forecast_source": forecast_source,
            "n_target_kps": len(target_kps),
            "n_closure": len(closure),
        },
    }


# ============================================================
# KPI
# ============================================================


def _build_kpi(
    dataset: Dataset,
    graph: KnowledgeGraph,
    model: Any,
    path: Any,
    goal: Any,
    closure: frozenset[str],
    target_kps: Sequence[str],
    records: Sequence[Interaction],
    forecast: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """KPI 卡片（``docs/API.md`` §7 ``kpi``）。

    定义：
        * ``overall_progress``：路径已完成课程数 / 路径总课程数；
          无路径时退化为该学习者的历史完成率。
        * ``mastered_kps``：**目标先修闭包内**掌握度 ≥ ``MASTERY_MASTERED(0.85)``
          的知识点个数；``total_target_kps``：目标知识点总数。
          （两者口径不同：分子统计「闭包内」，分母是「目标知识点」，
          因此 ``mastered_kps`` 可以大于 ``total_target_kps``。
          为了便于交叉核对，另外显式给出
          ``mastered_kps_in_closure``（同分子）与
          ``mastered_kps_all``（**全部 420 个知识点**中已掌握的数量，
          与 ``snapshot().mastered_count`` 完全一致）。）
        * ``avg_mastery``：目标先修闭包内的平均掌握度。
        * ``readiness``：``LearnerModel.snapshot().readiness``（目标就绪度）。
        * ``planned_hours``：路径总时长；``available_hours``：到截止日的总可用时长。
        * ``risk_courses``：完成概率低于 ``RISK_HIGH_THRESHOLD`` 的路径课程数。
        * ``learning_streak_weeks``：从最近一次学习往前连续有记录的周数。
    """
    learner = model.learner
    total_target = max(1, len(target_kps))
    mastered_in_closure = sum(
        1 for kp_id in closure if float(model.mastery_of(kp_id)) >= MASTERY_MASTERED
    )
    closure_values = [float(model.mastery_of(kp_id)) for kp_id in closure]
    avg_mastery = sum(closure_values) / len(closure_values) if closure_values else 0.0

    # 与 snapshot().mastered_count 对齐的口径：全部知识点中已掌握的数量
    try:
        mastered_all = int(model.snapshot().mastered_count)
    except Exception:  # pragma: no cover
        mastered_all = sum(
            1
            for kp_id in dataset.knowledge_points
            if float(model.mastery_of(kp_id)) >= MASTERY_MASTERED
        )

    steps = list(getattr(path, "steps", []) or [])
    done_ids = {r.course_id for r in records if r.completed}
    if steps:
        done = sum(1 for step in steps if step.course.course_id in done_ids)
        overall_progress = done / len(steps)
    else:
        overall_progress = (sum(1 for r in records if r.completed) / len(records)) if records else 0.0

    readiness = 0.0
    try:
        readiness = float(model.snapshot().readiness)
    except Exception:  # pragma: no cover - 快照异常不影响仪表板可用性
        readiness = 0.0

    return {
        "overall_progress": round(float(overall_progress), 3),
        "mastered_kps": int(mastered_in_closure),
        "total_target_kps": int(total_target),
        "mastered_kps_in_closure": int(mastered_in_closure),
        "mastered_kps_all": int(mastered_all),
        "closure_kps": int(len(closure)),
        "avg_mastery": round(float(avg_mastery), 3),
        "readiness": round(readiness, 3),
        "planned_hours": round(float(getattr(path, "total_hours", 0.0) or 0.0), 2),
        "available_hours": round(float(getattr(learner, "total_available_hours", 0.0) or 0.0), 2),
        "risk_courses": int(
            sum(1 for item in forecast if float(item.get("p_completion", 1.0)) < RISK_HIGH_THRESHOLD)
        ),
        "learning_streak_weeks": learning_streak(dataset, records),
        "weekly_budget": round(float(getattr(learner, "time_budget_hours_per_week", 0.0) or 0.0), 2),
        "deadline_weeks": int(getattr(learner, "deadline_weeks", 0) or 0),
        "gap_count": int(len(model.knowledge_gap())),
        "required_mastery": round(float(getattr(goal, "required_mastery", 0.8) or 0.8), 3),
    }


# ============================================================
# 掌握度推进曲线
# ============================================================


def mastery_curve(
    graph: KnowledgeGraph,
    model: Any,
    path: Any,
    closure: frozenset[str] | None = None,
    target_kps: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """掌握度推进曲线（第 0 周为当前状态，之后逐周推进）。

    推进算法（真实计算，非拟合）：

    1. 取目标先修闭包 ``K`` 的当前掌握度 ``m_k``；
    2. 按路径顺序逐门课程处理：把该课程的 ``expected_gain`` 按
       ``KPCoverage.weight`` 归一化后**分摊**到它覆盖的、且属于 ``K`` 的知识点上，
       得到 ``m_k ← min(1, m_k + gain · w_k / Σw)``；
    3. 每处理完一门课，记录一次 ``(week, avg_mastery, mastered)``，
       ``avg_mastery = mean_{k∈K} m_k``，``mastered = |{k∈T : m_k ≥ 0.85}|``；
    4. 同一周内多门课只保留该周末的状态（曲线按周采样）。

    Args:
        graph: 知识点图。
        model: 学习者模型。
        path: 学习路径（``None`` 或空路径时只返回第 0 周）。
        closure: 目标闭包（缺省由 ``model.target_kps`` 推出）。
        target_kps: 目标知识点（用于统计 mastered 数）。

    Returns:
        ``[{"week": int, "avg_mastery": float, "mastered": int}]``。
    """
    if closure is None:
        closure = graph.prerequisite_closure(
            tuple(getattr(model, "target_kps", getattr(model.goal, "target_kp_ids", ())))
        )
    if target_kps is None:
        target_kps = tuple(getattr(model, "target_kps", ()))

    mastery = {kp_id: float(model.mastery_of(kp_id)) for kp_id in closure}

    def _snapshot(week: int) -> dict[str, Any]:
        values = list(mastery.values())
        avg = sum(values) / len(values) if values else 0.0
        return {
            "week": int(week),
            "avg_mastery": round(float(avg), 3),
            "mastered": int(
                sum(1 for kp_id in target_kps if mastery.get(kp_id, 0.0) >= MASTERY_MASTERED)
            ),
        }

    curve = [_snapshot(0)]
    steps = list(getattr(path, "steps", []) or [])
    if not steps:
        return curve

    # 逐门课推进掌握度，记录每门课结束时的状态
    state_by_week: dict[int, dict[str, Any]] = {}
    last_week = 0
    for step in steps:
        hits = [cov for cov in step.course.kp_coverage if cov.kp_id in mastery]
        total_weight = sum(float(cov.weight) for cov in hits)
        if total_weight > _EPS:
            gain = float(step.expected_gain)
            for cov in hits:
                share = gain * float(cov.weight) / total_weight
                mastery[cov.kp_id] = min(1.0, mastery[cov.kp_id] + share)
        week = int(step.week)
        last_week = max(last_week, week)
        state_by_week[week] = _snapshot(week)

    # 周数取「最后一门课的起始周」与「路径总周数」的较大者：
    # 一门课可能跨越多个周（例如 14 小时的课跨 5 周），此时路径总周数
    # 会大于最后一门课的起始周，曲线要一直画到路径结束。
    total_weeks = int(getattr(path, "weeks", 0) or 0)
    last_week = max(last_week, total_weeks)

    # 按周采样：W0..WN 共 N+1 个点。没有课程的周沿用**上一周已记录的状态**
    # （不能用最终状态回填，否则曲线会先高后低，失去「单调推进」的语义）。
    for week in range(1, last_week + 1):
        if week in state_by_week:
            curve.append(state_by_week[week])
        else:
            previous = curve[-1]
            curve.append(
                {
                    "week": int(week),
                    "avg_mastery": float(previous["avg_mastery"]),
                    "mastered": int(previous["mastered"]),
                }
            )
    return curve


# ============================================================
# 领域雷达
# ============================================================


def domain_radar(model: Any, goal: Any) -> list[dict[str, Any]]:
    """领域掌握度雷达（``domain -> mastery`` 来自 ``LearnerModel.domain_mastery()``）。

    ``target`` 取目标的 ``required_mastery``（默认 0.80），作为雷达图的参考环。
    """
    try:
        mastery_map = dict(model.domain_mastery())
    except Exception:  # pragma: no cover - 兜底为全 0
        mastery_map = {}
    target = float(getattr(goal, "required_mastery", 0.8) or 0.8)
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        value = float(mastery_map.get(domain, 0.0))
        rows.append(
            {
                "domain": domain,
                "mastery": round(value, 3),
                "target": round(target, 3),
            }
        )
    return rows


# ============================================================
# 难度 / 模态 / 时间分配
# ============================================================


def difficulty_distribution(
    dataset: Dataset,
    path: Any,
    records: Sequence[Interaction],
) -> list[dict[str, Any]]:
    """难度分布：路径课程的难度直方图；无路径时用该学习者的历史课程。"""
    steps = list(getattr(path, "steps", []) or [])
    counter = {level: 0 for level in range(MIN_LEVEL, MAX_LEVEL + 1)}
    if steps:
        for step in steps:
            level = max(MIN_LEVEL, min(MAX_LEVEL, int(step.level)))
            counter[level] += 1
    else:
        for record in records:
            course = dataset.courses.get(record.course_id)
            if course is None:
                continue
            level = max(MIN_LEVEL, min(MAX_LEVEL, int(course.level)))
            counter[level] += 1
    return [{"level": level, "count": counter[level]} for level in sorted(counter)]


def modality_mix(
    dataset: Dataset,
    path: Any,
    records: Sequence[Interaction],
) -> list[dict[str, Any]]:
    """模态配比：路径各模态的课时总量；无路径时用历史投入时长。"""
    hours = {modality: 0.0 for modality in MODALITIES}
    steps = list(getattr(path, "steps", []) or [])
    if steps:
        for step in steps:
            modality = step.course.modality
            hours[modality] = hours.get(modality, 0.0) + float(step.hours)
    else:
        for record in records:
            course = dataset.courses.get(record.course_id)
            if course is None:
                continue
            hours[course.modality] = hours.get(course.modality, 0.0) + float(record.hours_spent)
    return [
        {
            "modality": modality,
            "name": MODALITY_NAMES.get(modality, modality),
            "hours": round(float(hours.get(modality, 0.0)), 2),
        }
        for modality in MODALITIES
    ]


def time_allocation(model: Any, path: Any) -> list[dict[str, Any]]:
    """每周时间分配：路径 ``weekly_plan`` 的课时 vs 学习者每周预算。"""
    weekly_budget = float(getattr(model.learner, "time_budget_hours_per_week", 0.0) or 0.0)
    plan = list(getattr(path, "weekly_plan", []) or [])
    return [
        {
            "week": int(item["week"]),
            "hours": round(float(item["hours"]), 2),
            "planned": round(weekly_budget, 2),
        }
        for item in plan
    ]


# ============================================================
# 完成预测
# ============================================================


def _completion_forecast(
    dataset: Dataset,
    graph: KnowledgeGraph,
    model: Any,
    steps: Sequence[Any],
    predictions: Sequence[Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    """生成 ``completion_forecast`` 并返回数据来源标记。"""
    if not steps:
        return [], "empty"

    if predictions is not None:
        rows = _forecast_from_predictions(steps, predictions)
        if rows:
            return rows, "provided"

    predictor = _load_predictor()
    if predictor is not None:
        rows: list[dict[str, Any]] = []
        for step in steps[:FORECAST_LIMIT]:
            try:
                result = predictor.predict_completion(dataset, graph, model, step.course)
            except Exception:  # pragma: no cover - 单门课失败不影响整表
                result = None
            if result is None:
                rows.append(_analytic_forecast_row(model, step))
            else:
                rows.append(
                    {
                        "order": int(step.order),
                        "course_id": step.course.course_id,
                        "course_title": step.course.title,
                        "p_completion": round(float(result.p_completion), 3),
                        "risk_level": result.risk_level,
                        "source": "model",
                    }
                )
        return rows, "model"

    return [_analytic_forecast_row(model, step) for step in steps[:FORECAST_LIMIT]], "analytic"


def _load_predictor() -> Any:
    """尝试加载 ``models/`` 下的效果预测模型；不可用返回 ``None``。"""
    try:
        from learnpath.prediction.effect import EffectPredictor

        predictor = EffectPredictor()
        predictor.load()
        return predictor
    except Exception:  # pragma: no cover - 模型缺失 / 版本不匹配时的兜底
        return None


def _analytic_forecast_row(model: Any, step: Any) -> dict[str, Any]:
    """解析式完成概率兜底（模型不可用时使用）。

    ``p = clip(0.12 + 0.40·readiness + 0.18·time_fit + 0.30·历史完成率, 0, 1)``

    这三个信号与效果预测模型的输入同源（先修就绪度、时间可承受度、历史完成率），
    只是没有经过训练校准，因此返回的行会标注 ``source="analytic"``，
    前端应据此提示「预测模型未就绪」。
    """
    p = _analytic_completion(model, step.course)
    return {
        "order": int(step.order),
        "course_id": step.course.course_id,
        "course_title": step.course.title,
        "p_completion": round(float(p), 3),
        "risk_level": _risk_level_of(p),
        "source": "analytic",
    }


def _analytic_completion(model: Any, course: Course) -> float:
    """解析式完成概率（见 :func:`_analytic_forecast_row`）。"""
    readiness = float(model.readiness_for(course))
    time_fit = float(model.time_fit(course))
    try:
        history = float(model.snapshot().completion_rate)
    except Exception:  # pragma: no cover
        history = 0.45
    value = (
        ANALYTIC_BASE
        + ANALYTIC_READINESS * readiness
        + ANALYTIC_TIME_FIT * time_fit
        + ANALYTIC_HISTORY * history
    )
    return max(0.0, min(1.0, value))


def _risk_level_of(p_completion: float) -> str:
    if p_completion >= RISK_LOW_THRESHOLD:
        return "低"
    if p_completion >= RISK_HIGH_THRESHOLD:
        return "中"
    return "高"


def _forecast_from_predictions(
    steps: Sequence[Any],
    predictions: Sequence[Any],
) -> list[dict[str, Any]]:
    """把外部传入的预测序列规整成 ``completion_forecast`` 行。"""
    rows: list[dict[str, Any]] = []
    for step, prediction in zip(steps, predictions):
        p: float | None = None
        risk = ""
        if isinstance(prediction, (tuple, list)) and len(prediction) >= 2:
            p = float(prediction[0])
            risk = str(prediction[1])
        elif isinstance(prediction, Mapping):
            p = float(prediction.get("p_completion", 0.0))
            risk = str(prediction.get("risk_level", ""))
        else:
            value = getattr(prediction, "p_completion", None)
            if value is not None:
                p = float(value)
                risk = str(getattr(prediction, "risk_level", ""))
        if p is None:
            continue
        p = max(0.0, min(1.0, p))
        rows.append(
            {
                "order": int(step.order),
                "course_id": step.course.course_id,
                "course_title": step.course.title,
                "p_completion": round(p, 3),
                "risk_level": risk or _risk_level_of(p),
                "source": "provided",
            }
        )
    return rows


# ============================================================
# 缺口表 / 历史活动
# ============================================================


def gap_table(graph: KnowledgeGraph, model: Any, path: Any) -> list[dict[str, Any]]:
    """知识缺口表：按缺口幅度降序取前 ``GAP_TABLE_LIMIT`` 条，并标注是否在路径内。"""
    gaps = dict(model.knowledge_gap())
    if not gaps:
        return []
    in_path = set(getattr(path, "kp_ids", []) or [])
    rows: list[dict[str, Any]] = []
    for kp_id, gap_value in sorted(gaps.items(), key=lambda item: (-item[1], item[0]))[
        :GAP_TABLE_LIMIT
    ]:
        if kp_id not in graph:
            continue
        kp = graph.node(kp_id)
        rows.append(
            {
                "kp_id": kp_id,
                "name": kp.name,
                "mastery": round(float(model.mastery_of(kp_id)), 3),
                "gap": round(float(gap_value), 3),
                "difficulty": int(kp.difficulty),
                "domain": kp.domain,
                "in_path": kp_id in in_path,
                "depth": int(graph.depth(kp_id)),
            }
        )
    return rows


def history_activity(
    dataset: Dataset,
    records: Sequence[Interaction],
) -> list[dict[str, Any]]:
    """历史活动按周聚合（真实数据）。

    周次定义：以该学习者**最早一条交互**为第 1 周起点，
    ``week = ⌊(t - t_min).days / 7⌋ + 1``。每周输出实际投入时长与完成课程数。
    """
    stamps = [(r, _parse_ts(r.timestamp)) for r in records]
    valid = [(r, ts) for r, ts in stamps if ts is not None]
    if not valid:
        return []
    origin = min(ts for _r, ts in valid)

    buckets: dict[int, dict[str, Any]] = {}
    for record, ts in valid:
        week = int((ts - origin).days // 7) + 1
        bucket = buckets.setdefault(week, {"week": week, "hours": 0.0, "completed": 0})
        bucket["hours"] += float(record.hours_spent)
        if record.completed:
            bucket["completed"] += 1

    return [
        {
            "week": int(buckets[week]["week"]),
            "hours": round(float(buckets[week]["hours"]), 2),
            "completed": int(buckets[week]["completed"]),
        }
        for week in sorted(buckets)
    ]


def learning_streak(dataset: Dataset, records: Sequence[Interaction]) -> int:
    """连续学习周数：从**最近一次**学习所在的周开始，往前数连续有记录的周数。"""
    stamps = [ts for ts in (_parse_ts(r.timestamp) for r in records) if ts is not None]
    if not stamps:
        return 0
    origin = min(stamps)
    weeks = sorted({int((ts - origin).days // 7) + 1 for ts in stamps})
    latest = weeks[-1]
    streak = 1
    cursor = latest
    present = set(weeks)
    while cursor - 1 in present:
        streak += 1
        cursor -= 1
    return int(streak)


def _parse_ts(text: str) -> datetime | None:
    """解析 ISO 8601 时间戳，失败返回 ``None``。"""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


__all__ = [
    "build_dashboard",
    "mastery_curve",
    "domain_radar",
    "difficulty_distribution",
    "modality_mix",
    "time_allocation",
    "gap_table",
    "history_activity",
    "learning_streak",
    "RISK_LOW_THRESHOLD",
    "RISK_HIGH_THRESHOLD",
]
