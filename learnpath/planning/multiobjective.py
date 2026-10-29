"""多目标优化内核：目标归一化、非支配排序、拥挤度、NSGA-II 与 TOPSIS 妥协解选择。

本模块对应实现要求 ④「多目标优化：平衡学习效果、时间成本、难度梯度」，
是纯数学层，**不依赖任何学习器实现**（只通过鸭子类型访问 ``model`` 的公开方法），
因此可以被单元测试用 stub 单独验证。

四个目标（全部归一化到 ``[0, 1]``，且**统一为「越大越好」**）：

======================  ==================================================  ============
目标                    含义                                                方向
======================  ==================================================  ============
``gain``                路径累计预期掌握度增益 / 学习者全部知识缺口总量        最大化
``time``                时间可承受度 = 1 - 归一化时间成本                    最大化
``smoothness``          难度梯度平滑度（相邻难度差 + 能力错配惩罚）           最大化
``fit``                 平均 ``style_fit × time_fit``（风格 × 时间匹配）      最大化
======================  ==================================================  ============

之所以把「时间成本最小化」翻转为「可承受度最大化」，是为了让四个目标方向一致，
从而可以直接套用标准 NSGA-II 的支配关系定义：

    a 支配 b  ⟺  ∀j: a_j ≥ b_j  且  ∃j: a_j > b_j

------------------------------------------------------------------
一、目标归一化（:func:`evaluate_objectives`）
------------------------------------------------------------------
令路径步骤集合为 ``S``，学习者能力 ``A ∈ [0,1]``，总可用时长 ``B``（小时），
知识缺口集合 ``G``（``model.knowledge_gap()``，值为「待提升幅度」）。

1) **gain（学习收益）**::

       raw_gain = Σ_{s∈S} max(0, expected_gain(s))
       gain     = clip(raw_gain / max(ε, Σ_{kp∈G} gap(kp)), 0, 1)

   分母取「学习者全部知识缺口总量」，因此 ``gain`` 具有明确语义：
   **这条路径能把学习者的缺口补上多大比例**，天然落在 ``[0,1]`` 且随路径单调递增。

2) **time（时间可承受度）**::

       cost        = Σ_{s∈S} hours(s)
       norm_cost   = cost / (B + cost)          # 软归一化
       time        = 1 - norm_cost = B / (B + cost)

   ``norm_cost`` 是「成本占（预算 + 成本）的比例」，是严格单调的成本归一化，
   且不会像 ``clip(cost/B, 0, 1)`` 那样在超预算后**饱和到常数**——
   饱和会让所有超预算方案的目标值完全相同，使拥挤度距离退化为 0，
   进而让 NSGA-II 在超预算区域失去选择压力。

3) **smoothness（难度梯度平滑度）**::

       n           = |S|
       diffs       = [level(s_{i+1}) - level(s_i)]
       ss          = Σ diffs²
       norm_ss     = ss / max(1, (n-1) · (MAX_LEVEL - MIN_LEVEL)²)
       ability_lvl = MIN_LEVEL + (MAX_LEVEL - MIN_LEVEL) · A
       excess      = Σ_s max(0, level(s) - ability_lvl - TOL)
       norm_excess = excess / (n · (MAX_LEVEL - MIN_LEVEL))
       smoothness  = clip(1 - norm_ss - JUMP_PENALTY · norm_excess, 0, 1)

   第一项惩罚「相邻步骤难度跳变」，第二项惩罚「难度超出学习者能力上界」
   （``TOL = 0.75`` 级缓冲，允许略高于当前水平的适度挑战）。

4) **fit（风格匹配）**::

       fit = mean_{s∈S} [ style_fit(s) × time_fit(s) ]

------------------------------------------------------------------
二、NSGA-II（:func:`nsga2`）
------------------------------------------------------------------
决策向量 ``x ∈ [0,1]^m``（``m`` = 候选课程数），``x_i`` 表示「第 i 门候选课的入选倾向」，
由 :func:`decode_threshold` 这类解码器映射成真实的课程集合，从而把
「课程组合选择」这一离散组合优化问题嵌入连续空间，才能使用实数编码的
SBX 交叉与多项式变异。

算法步骤（与 Deb 2002 原始文献一致）：

1. 初始种群 ``P_0``（``N = NSGA_POPULATION`` 个随机决策向量）；
2. 每代：
   a. **二元锦标赛选择**（比较 ``(rank, -crowding)`` 字典序）；
   b. **模拟二进制交叉 SBX**（``eta_c = 20``，分布指数越大子代越接近父代；
      以 ``NSGA_CROSSOVER_RATE = 0.85`` 的概率执行），越界值裁剪回 ``[0,1]``；
   c. **多项式变异**（``eta_m = 20``，以 ``NSGA_MUTATION_RATE = 0.20`` 的概率
      对个体执行，个体内部每个决策分量的变异概率为 ``1/m``）；
   d. **精英保留**：父代 ∪ 子代（``2N`` 个）快速非支配排序分层，
      逐层填充直到超过 ``N``，最后一层按**拥挤度距离降序**截断。
3. 终止后返回第 1 层（帕累托前沿）。

复杂度：非支配排序 ``O(M·N²)``（``M`` = 目标数 = 4），拥挤度 ``O(M·N·logN)``；
本实现把支配关系用 NumPy 广播向量化，``N = 60`` 时单代约 0.1 ms 量级。

------------------------------------------------------------------
三、TOPSIS 妥协解选择（:func:`select_compromise`）
------------------------------------------------------------------
帕累托前沿上每个解都「互不支配」，需要按决策者偏好（``ObjectiveWeights``）挑一个：

1. 构造决策矩阵 ``X = [x_ij]``（``i`` 个方案 × ``j`` 个准则）；
2. **线性极差归一化** ``r_ij = (x_ij - min_i x_ij) / (max_i x_ij - min_i x_ij)``，
   把每个准则在前沿上拉伸到 ``[0,1]``
   （不用经典向量归一化的原因见 :func:`topsis_scores` 的 docstring：
   本问题 ``time`` 准则取值集中，向量归一化会让它被 ``gain`` 淹没）；
3. **加权** ``v_ij = w_j · r_ij``（``w`` 来自 ``PLAN_PRESETS`` 的 preset 权重）；
4. **正理想解** ``A⁺ = (max_i v_i1, …, max_i v_iM)``，
   **负理想解** ``A⁻ = (min_i v_i1, …, min_i v_iM)``
   （本系统中四个准则统一为效益型，故 ``A⁺`` 取列最大、``A⁻`` 取列最小）；
5. **距离** ``S_i⁺ = ‖v_i - A⁺‖₂``，``S_i⁻ = ‖v_i - A⁻‖₂``；
6. **相对贴近度** ``C_i = S_i⁻ / (S_i⁺ + S_i⁻) ∈ [0,1]``，取 ``argmax C_i`` 为妥协解。

``C_i`` 越接近 1 表示该方案在「多目标上都不吃亏」的意义下越优，
这正是「均衡 / 速通 / 夯实」三种预设通过**同一套准则、不同权重**产生差异化路径的机制。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from learnpath.config import (
    MAX_LEVEL,
    MIN_LEVEL,
    NSGA_CROSSOVER_RATE,
    NSGA_GENERATIONS,
    NSGA_MUTATION_RATE,
    NSGA_POPULATION,
    EVAL_SEED,
    ObjectiveWeights,
)

# ============================================================
# 常量与超参数
# ============================================================

#: 四个目标的固定顺序（构造决策矩阵 / 输出字典时都用它，保证可复现）
OBJECTIVE_KEYS: Final[tuple[str, ...]] = ("gain", "time", "smoothness", "fit")

#: 难度错配惩罚系数：把「难度超出能力上界」的归一化量折算成平滑度扣分
SMOOTHNESS_JUMP_PENALTY: Final[float] = 0.50

#: 允许路径难度略高于学习者当前能力上界的缓冲（单位：难度级）
ABILITY_LEVEL_TOLERANCE: Final[float] = 0.75

#: 无法从 model 读到时间预算时使用的缺省每周时长（小时）
DEFAULT_WEEKLY_BUDGET: Final[float] = 6.0

#: 无法从 model 读到截止周数时使用的缺省值
DEFAULT_DEADLINE_WEEKS: Final[int] = 12

#: 全局数值保护
_EPS: Final[float] = 1e-9


def clamp01(value: float) -> float:
    """把数值裁剪到 ``[0, 1]``（``NaN`` 归一为 0.0）。"""
    if value != value:  # NaN
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


# ============================================================
# 学习者上下文：目标归一化所需的标量
# ============================================================


@dataclass(frozen=True, slots=True)
class LearnerContext:
    """多目标评估所需的「和路径无关」的学习者标量集合。

    单独抽出来是为了让本模块保持纯数学：只依赖 ``LearnerModel`` 的公开契约
    （``snapshot()`` / ``knowledge_gap()``），并对实现细节做防御性 ``getattr``。

    Attributes:
        ability: 学习者综合能力 ``[0,1]``，用于难度错配惩罚。
        weekly_budget: 每周可投入小时数。
        total_budget: 到截止日期为止的总可用小时数。
        gap_total: 全部知识缺口幅度之和（``gain`` 的归一化分母）。
        gap_count: 缺口知识点个数。
    """

    ability: float = 0.5
    weekly_budget: float = DEFAULT_WEEKLY_BUDGET
    total_budget: float = DEFAULT_WEEKLY_BUDGET * DEFAULT_DEADLINE_WEEKS
    gap_total: float = 0.0
    gap_count: int = 0


def learner_context(model: Any) -> LearnerContext:
    """从 :class:`~learnpath.learner.profile.LearnerModel` 抽取评估上下文。

    对学习者对象的访问做三级回退：
        1. ``model.learner``（最直接的属性，:class:`LearnerModel` 构造时注入）；
        2. ``model.snapshot()``（契约方法，返回 ``ProfileSnapshot``）；
        3. 配置缺省值（``DEFAULT_WEEKLY_BUDGET × DEFAULT_DEADLINE_WEEKS``）。

    这样即使 ``LearnerModel`` 的内部属性名变化，本模块也不会崩。
    """
    learner = getattr(model, "learner", None)

    ability = 0.5
    weekly: float | None = None
    total: float | None = None

    if learner is not None:
        ability = float(getattr(learner, "ability", 0.5) or 0.5)
        weekly = _positive_or_none(getattr(learner, "time_budget_hours_per_week", None))
        total = _positive_or_none(getattr(learner, "total_available_hours", None))
        deadline = getattr(learner, "deadline_weeks", None)
    else:
        deadline = None

    if weekly is None or total is None:
        snapshot = None
        try:
            snapshot = model.snapshot()
        except Exception:  # pragma: no cover - 防御性回退
            snapshot = None
        if snapshot is not None:
            if weekly is None:
                weekly = _positive_or_none(getattr(snapshot, "time_budget", None))
            if deadline is None:
                deadline = getattr(snapshot, "deadline_weeks", None)

    weekly = weekly or DEFAULT_WEEKLY_BUDGET
    if total is None:
        try:
            weeks = int(deadline) if deadline else DEFAULT_DEADLINE_WEEKS
        except (TypeError, ValueError):
            weeks = DEFAULT_DEADLINE_WEEKS
        total = weekly * max(1, weeks)

    gap: Mapping[str, float] = {}
    try:
        gap = model.knowledge_gap() or {}
    except Exception:  # pragma: no cover - 防御性回退
        gap = {}
    gap_values = [max(0.0, float(v)) for v in gap.values()]

    return LearnerContext(
        ability=clamp01(ability),
        weekly_budget=float(weekly),
        total_budget=float(total),
        gap_total=float(sum(gap_values)),
        gap_count=len(gap_values),
    )


def _positive_or_none(value: Any) -> float | None:
    """把可能是 ``None`` / 非正数的数值转成 ``float``，否则返回 ``None``。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


# ============================================================
# 目标函数
# ============================================================


def evaluate_objectives(
    steps: Sequence[Any],
    model: Any,
    *,
    cache: MutableMapping[str, tuple[float, float]] | None = None,
    ctx: LearnerContext | None = None,
    reference_gain: float | None = None,
) -> dict[str, float]:
    """计算一条路径的四个归一化目标值。

    Args:
        steps: 路径步骤序列。元素需具备 ``course`` / ``hours`` / ``level`` /
            ``expected_gain`` 属性（:class:`~learnpath.planning.planner.PathStep`
            天然满足；NSGA-II 内部用轻量视图对象以便高频调用）。
        model: :class:`~learnpath.learner.profile.LearnerModel`，需要
            ``knowledge_gap()`` / ``style_fit(course)`` / ``time_fit(course)``。
        cache: 可选的 ``course_id -> (style_fit, time_fit)`` 缓存。NSGA-II 单次
            规划会评估上千个个体，同一门课的 ``style_fit/time_fit`` 会被重复计算，
            传入共享缓存可把 ``fit`` 目标的计算量从 ``O(评估次数·步骤数)``
            降到 ``O(候选课程数)``。
        ctx: 可选的预计算 :class:`LearnerContext`。搜索过程中学习者状态不变，
            外部算一次传入可省掉每次评估的 ``knowledge_gap()`` 全量遍历。
        reference_gain: ``gain`` 目标的归一化基准——「在时间预算内、把每个缺口知识点
            都覆盖一次所能取得的增益上界」（由
            :meth:`learnpath.planning.planner.PathPlanner.reference_gain` 预估）。
            为 ``None`` 时退化为「学习者全部知识缺口总量」。

    Returns:
        ``{"gain": float, "time": float, "smoothness": float, "fit": float}``，
        四个值都在 ``[0,1]``，且**都是越大越好**。

    Note:
        四个目标的归一化公式见模块 docstring 第一节；路径为空时返回
        ``gain=0, time=1, smoothness=1, fit=0``（零成本但零收益的退化解）。
    """
    if ctx is None:
        ctx = learner_context(model)
    items = list(steps)
    n = len(items)
    if n == 0:
        return {"gain": 0.0, "time": 1.0, "smoothness": 1.0, "fit": 0.0}

    hours: list[float] = []
    levels: list[int] = []
    raw_gain = 0.0
    fit_sum = 0.0

    for step in items:
        course = step.course
        hours.append(max(0.0, float(step.hours)))
        levels.append(int(step.level))
        raw_gain += max(0.0, float(step.expected_gain))

        key = getattr(course, "course_id", "")
        pair = cache.get(key) if cache is not None else None
        if pair is None:
            style = clamp01(_safe_float(model.style_fit(course), 0.5))
            time_fit = clamp01(_safe_float(model.time_fit(course), 0.5))
            pair = (style, time_fit)
            if cache is not None and key:
                cache[key] = pair
        fit_sum += pair[0] * pair[1]

    # ---- 1) gain ----
    if reference_gain is not None and reference_gain > _EPS:
        denominator = float(reference_gain)
    elif ctx.gap_total > _EPS:
        denominator = ctx.gap_total
    else:
        denominator = max(raw_gain, _EPS)
    gain = clamp01(raw_gain / denominator)

    # ---- 2) time ----
    cost = sum(hours)
    budget = ctx.total_budget if ctx.total_budget > _EPS else DEFAULT_WEEKLY_BUDGET
    time_score = clamp01(budget / (budget + cost)) if cost > 0 else 1.0

    # ---- 3) smoothness ----
    if n >= 2:
        span = float(MAX_LEVEL - MIN_LEVEL)
        squared = 0.0
        for i in range(n - 1):
            diff = float(levels[i + 1] - levels[i])
            squared += diff * diff
        norm_ss = squared / max(1.0, (n - 1) * span * span)
    else:
        norm_ss = 0.0

    span = float(MAX_LEVEL - MIN_LEVEL)
    ability_level = MIN_LEVEL + span * ctx.ability
    excess = 0.0
    for level in levels:
        over = float(level) - ability_level - ABILITY_LEVEL_TOLERANCE
        if over > 0:
            excess += over
    norm_excess = excess / (n * span) if span > 0 else 0.0
    smoothness = clamp01(1.0 - norm_ss - SMOOTHNESS_JUMP_PENALTY * norm_excess)

    # ---- 4) fit ----
    fit = clamp01(fit_sum / n)

    return {"gain": gain, "time": time_score, "smoothness": smoothness, "fit": fit}


def _safe_float(value: Any, default: float) -> float:
    """把 ``model`` 返回值安全转成 ``float``。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# 帕累托工具：目标矩阵 / 非支配排序 / 拥挤度
# ============================================================


def objective_matrix(candidates: Sequence[Any]) -> np.ndarray:
    """把候选解的 ``.objectives`` 字典拼成 ``(n, M)`` 决策矩阵（列序固定）。"""
    if not candidates:
        return np.zeros((0, len(OBJECTIVE_KEYS)), dtype=float)
    rows = []
    for candidate in candidates:
        objectives = candidate.objectives
        rows.append([float(objectives.get(key, 0.0)) for key in OBJECTIVE_KEYS])
    return np.asarray(rows, dtype=float)


def domination_matrix(values: np.ndarray) -> np.ndarray:
    """返回布尔支配矩阵 ``D[i, j] = 解 i 支配解 j``（向量化实现）。

    支配定义：``i`` 支配 ``j`` ⟺ 在全部 ``M`` 个目标上 ``i ≥ j`` 且至少一个目标 ``i > j``。
    """
    if values.size == 0:
        return np.zeros((0, 0), dtype=bool)
    ge = (values[:, None, :] >= values[None, :, :]).all(axis=2)
    gt = (values[:, None, :] > values[None, :, :]).any(axis=2)
    result = np.logical_and(ge, gt)
    np.fill_diagonal(result, False)
    return result


def non_dominated_sort(candidates: Sequence[Any]) -> list[list[Any]]:
    """快速非支配排序（Fast Non-dominated Sort, Deb 2002）。

    Args:
        candidates: 具备 ``.objectives: Mapping[str, float]`` 的解序列
            （``LearningPath`` / ``ParetoSolution`` / 任何带 objectives 的对象）。

    Returns:
        分层列表 ``[F0, F1, …]``，``F0`` 为帕累托第一前沿（互不支配且不被任何解支配），
        层内顺序保持输入顺序（便于结果可复现）。
    """
    if not candidates:
        return []
    values = objective_matrix(candidates)
    return [[candidates[int(i)] for i in layer] for layer in _nds_indices(values)]


def _nds_indices(values: np.ndarray) -> list[np.ndarray]:
    """非支配排序的「下标版」内核（NSGA-II 主循环用它，避免反复装箱对象）。"""
    n = int(values.shape[0])
    if n == 0:
        return []
    dom = domination_matrix(values)
    # counts[j] = 支配 j 的解的个数
    counts = dom.sum(axis=0).astype(np.int64)
    remaining = np.ones(n, dtype=bool)
    layers: list[np.ndarray] = []
    while remaining.any():
        layer = np.where(remaining & (counts == 0))[0]
        if layer.size == 0:  # pragma: no cover - 数值退化保护（理论上不可达）
            layer = np.where(remaining)[0]
        layers.append(layer)
        remaining[layer] = False
        counts -= dom[layer].sum(axis=0)
    return layers


def _crowding_values(values: np.ndarray) -> np.ndarray:
    """拥挤度距离的向量化内核（输入是同一前沿的目标矩阵）。"""
    n = int(values.shape[0])
    distance = np.zeros(n, dtype=float)
    if n == 0:
        return distance
    if n <= 2:
        distance[:] = math.inf
        return distance
    for m in range(values.shape[1]):
        order = np.argsort(values[:, m], kind="stable")
        column = values[order, m]
        span = float(column[-1] - column[0])
        distance[order[0]] = math.inf
        distance[order[-1]] = math.inf
        if span <= _EPS:
            continue
        distance[order[1:-1]] += (column[2:] - column[:-2]) / span
    return distance


def crowding_distance(front: Sequence[Any]) -> list[float]:
    """计算同一前沿内每个解的拥挤度距离。

    算法（对每个目标 ``m`` 独立执行）：

    1. 按目标 ``m`` 升序排序；
    2. 边界解（最小 / 最大）距离记为 ``+∞``，保证极端解永远被保留；
    3. 中间解累加 ``(f_{m}(i+1) - f_{m}(i-1)) / (f_m^max - f_m^min)``。

    距离越大表示该解周围越稀疏，应优先保留以维持前沿多样性。
    前沿规模 ≤ 2 时全部返回 ``+∞``。
    """
    if not front:
        return []
    return [float(value) for value in _crowding_values(objective_matrix(front))]


# ============================================================
# NSGA-II
# ============================================================


@dataclass(slots=True)
class ParetoSolution:
    """NSGA-II 前沿上的一个解。

    Attributes:
        chromosome: 长度 = 候选课程数的实数决策向量，含义为「入选倾向」。
        objectives: 四个归一化目标值。
        payload: 由 ``decode`` 解码出的问题对象（规划模块里是课程列表），
            只在最终前沿上解码，避免搜索过程中做无用解码。
        rank: 非支配层级（前沿解恒为 0）。
    """

    chromosome: np.ndarray
    objectives: dict[str, float]
    payload: Any = None
    rank: int = 0


def _tournament_select(
    ranks: Sequence[int],
    crowding: Sequence[float],
    rng: np.random.Generator,
    size: int = 2,
) -> int:
    """二元（或 k 元）锦标赛选择：``(rank 越小越好, crowding 越大越好)``。"""
    n = len(ranks)
    contenders = rng.integers(0, n, size=size)
    best = int(contenders[0])
    for raw in contenders[1:]:
        other = int(raw)
        if ranks[other] < ranks[best]:
            best = other
        elif ranks[other] == ranks[best] and crowding[other] > crowding[best]:
            best = other
    return best


def simulated_binary_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """模拟二进制交叉（SBX）。

    Args:
        parent1: 父代决策向量（取值在 ``[0,1]``）。
        parent2: 另一个父代。
        eta: 分布指数 ``eta_c``，越大子代越靠拢父代（默认 20）。
        rng: NumPy 随机数生成器。

    Returns:
        两个子代向量（已裁剪回 ``[0,1]``）。

    公式::

        u ~ U(0,1)
        β = (2u)^(1/(η+1))            , u ≤ 0.5
        β = (1/(2(1-u)))^(1/(η+1))    , u > 0.5
        c1 = ½[(1+β)p1 + (1-β)p2]
        c2 = ½[(1-β)p1 + (1+β)p2]

    边界处理采用裁剪（clipping）而非 Deb 的 bound-aware β 修正，
    因为本问题的决策空间本身就是「入选倾向」这一软约束，裁剪不会引入偏差。
    """
    u = rng.random(parent1.shape[0])
    power = 1.0 / (eta + 1.0)
    beta = np.where(
        u <= 0.5,
        np.power(np.maximum(2.0 * u, _EPS), power),
        np.power(np.maximum(1.0 / np.maximum(2.0 * (1.0 - u), _EPS), _EPS), power),
    )
    child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
    child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)
    return np.clip(child1, 0.0, 1.0), np.clip(child2, 0.0, 1.0)


def polynomial_mutation(
    individual: np.ndarray,
    eta: float,
    per_gene_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """多项式变异（Polynomial Mutation）。

    Args:
        individual: 待变异决策向量。
        eta: 分布指数 ``eta_m``（默认 20，越大扰动越小）。
        per_gene_rate: 每个决策分量的变异概率（通常取 ``1/m``）。
        rng: NumPy 随机数生成器。

    Returns:
        变异后的新向量（已裁剪回 ``[0,1]``）。

    公式::

        u ~ U(0,1)
        δ = (2u)^(1/(η+1)) - 1           , u < 0.5
        δ = 1 - (2(1-u))^(1/(η+1))       , u ≥ 0.5
        x' = clip(x + δ, 0, 1)
    """
    mask = rng.random(individual.shape[0]) < per_gene_rate
    if not mask.any():
        return individual.copy()
    u = rng.random(individual.shape[0])
    power = 1.0 / (eta + 1.0)
    delta = np.where(
        u < 0.5,
        np.power(np.maximum(2.0 * u, _EPS), power) - 1.0,
        1.0 - np.power(np.maximum(2.0 * (1.0 - u), _EPS), power),
    )
    mutated = individual + np.where(mask, delta, 0.0)
    return np.clip(mutated, 0.0, 1.0)


def nsga2(
    population: Sequence[np.ndarray],
    evaluate: Callable[[np.ndarray], Mapping[str, float]],
    generations: int = NSGA_GENERATIONS,
    crossover_rate: float = NSGA_CROSSOVER_RATE,
    mutation_rate: float = NSGA_MUTATION_RATE,
    *,
    decode: Callable[[np.ndarray], Any] | None = None,
    tournament_size: int = 2,
    eta_c: float = 20.0,
    eta_m: float = 20.0,
    seed: int = EVAL_SEED,
    history: list[dict[str, float]] | None = None,
) -> list[ParetoSolution]:
    """NSGA-II 主循环，返回帕累托第一前沿。

    Args:
        population: 初始种群（每个元素是 ``[0,1]^m`` 的决策向量）。建议把
            「分层贪心集合覆盖」得到的启发式解也放进去（种子解），保证前沿上
            至少有一个「可解释的好解」。
        evaluate: 目标函数，输入决策向量、输出 ``{gain, time, smoothness, fit}``。
            内部会按决策向量的字节串做缓存，重复个体不会重复评估。
        generations: 迭代代数（``NSGA_GENERATIONS = 40``）。
        crossover_rate: 交叉概率（``NSGA_CROSSOVER_RATE = 0.85``）。
        mutation_rate: 个体级变异概率（``NSGA_MUTATION_RATE = 0.20``）；
            个体内部每个分量的变异概率取 ``1.0 / m``。
        decode: 可选的解码器，只对**最终前沿**逐个调用，把决策向量还原成
            问题对象（规划模块里是课程列表）放进 :attr:`ParetoSolution.payload`。
        tournament_size: 锦标赛规模（默认 2）。
        eta_c: SBX 分布指数（默认 20）。
        eta_m: 多项式变异分布指数（默认 20）。
        seed: 随机种子（默认 ``EVAL_SEED``，保证可复现）。
        history: 可选的外部列表，每代追加一条
            ``{"generation", "front_size", "best_gain", "best_time", …}``，
            供报告画收敛曲线。

    Returns:
        帕累托第一前沿（已按 ``(total_hours 升序)`` 无关、按目标值去重），
        列表元素为 :class:`ParetoSolution`。

    Raises:
        ValueError: 初始种群为空。
    """
    if not population:
        raise ValueError("NSGA-II 需要非空初始种群")

    rng = np.random.default_rng(seed)
    size = len(population)

    cache: dict[bytes, dict[str, float]] = {}

    def _evaluate(chromosome: np.ndarray) -> dict[str, float]:
        key = chromosome.tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached
        raw = evaluate(chromosome)
        objectives = {name: float(raw.get(name, 0.0)) for name in OBJECTIVE_KEYS}
        cache[key] = objectives
        return objectives

    def _matrices(objs: Sequence[Mapping[str, float]]) -> np.ndarray:
        return np.asarray(
            [[float(o[key]) for key in OBJECTIVE_KEYS] for o in objs], dtype=float
        )

    def _rank_and_crowd(
        objs: Sequence[Mapping[str, float]],
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """一次完成非支配分层 + 层级 + 拥挤度（避免重复排序）。"""
        values = _matrices(objs)
        layers = _nds_indices(values)
        n = len(objs)
        ranks = np.zeros(n, dtype=np.int64)
        crowd = np.zeros(n, dtype=float)
        for level, layer in enumerate(layers):
            ranks[layer] = level
            crowd[layer] = _crowding_values(values[layer])
        return layers, ranks, crowd

    pop = [np.asarray(ind, dtype=float).copy() for ind in population]
    objectives: list[dict[str, float]] = [_evaluate(ind) for ind in pop]
    layers, ranks, crowd = _rank_and_crowd(objectives)

    for generation in range(max(0, int(generations))):
        # ---------- 1) 生成子代：锦标赛选择 → SBX → 多项式变异 ----------
        gene_rate = 1.0 / max(1, pop[0].shape[0])
        offspring: list[np.ndarray] = []
        while len(offspring) < size:
            i1 = _tournament_select(ranks, crowd, rng, tournament_size)
            i2 = _tournament_select(ranks, crowd, rng, tournament_size)
            parent1, parent2 = pop[i1], pop[i2]
            if rng.random() < crossover_rate:
                child1, child2 = simulated_binary_crossover(parent1, parent2, eta_c, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            for child in (child1, child2):
                if rng.random() < mutation_rate:
                    child = polynomial_mutation(child, eta_m, gene_rate, rng)
                offspring.append(child)
                if len(offspring) >= size:
                    break
        offspring = offspring[:size]

        # ---------- 2) 精英保留：父代 ∪ 子代 → 排序 → 按层 + 拥挤度截断 ----------
        combined = pop + offspring
        combined_objs = objectives + [_evaluate(ind) for ind in offspring]
        layers, combined_ranks, combined_crowd = _rank_and_crowd(combined_objs)

        selected_idx: list[int] = []
        for layer in layers:
            if len(selected_idx) + int(layer.size) <= size:
                selected_idx.extend(int(i) for i in layer)
                continue
            # 最后一层：按拥挤度降序填充（并列时按下标，保证确定性）
            order = sorted(
                (int(i) for i in layer),
                key=lambda i: (
                    -(0.0 if combined_crowd[i] != combined_crowd[i] else combined_crowd[i]),
                    i,
                ),
            )
            selected_idx.extend(order[: size - len(selected_idx)])
            break

        pop = [combined[i] for i in selected_idx]
        objectives = [dict(combined_objs[i]) for i in selected_idx]
        layers, ranks, crowd = _rank_and_crowd(objectives)

        if history is not None:
            gains = [o["gain"] for o in objectives]
            times = [o["time"] for o in objectives]
            history.append(
                {
                    "generation": generation + 1,
                    "front_size": float(layers[0].size) if layers else 0.0,
                    "best_gain": float(max(gains, default=0.0)),
                    "mean_gain": float(np.mean(gains)),
                    "mean_time": float(np.mean(times)),
                }
            )

    # ---------- 3) 输出帕累托前沿（按目标值去重） ----------
    proxies = [
        ParetoSolution(chromosome=pop[i], objectives=dict(objectives[i]))
        for i in range(len(pop))
    ]
    first_front = layers[0] if layers else np.arange(0, dtype=np.int64)

    seen: set[tuple[float, ...]] = set()
    unique: list[ParetoSolution] = []
    for raw_index in first_front:
        solution = proxies[int(raw_index)]
        signature = tuple(round(solution.objectives[key], 6) for key in OBJECTIVE_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        if decode is not None:
            solution.payload = decode(solution.chromosome)
        unique.append(solution)

    unique.sort(key=lambda sol: (sol.objectives["time"], -sol.objectives["gain"]))
    return unique


# ============================================================
# TOPSIS 妥协解选择
# ============================================================


def topsis_scores(front: Sequence[Any], weights: ObjectiveWeights) -> list[float]:
    """计算帕累托前沿上每个方案的 TOPSIS 相对贴近度 ``C_i``。

    Args:
        front: 前沿解序列（需有 ``.objectives``）。
        weights: 准则权重（:class:`~learnpath.config.ObjectiveWeights`）。

    Returns:
        与 ``front`` 等长的贴近度列表，取值 ``[0,1]``，越大越优。

    归一化方式说明（**重要设计决策**）
    ----------------------------------
    经典 TOPSIS 用向量归一化 ``r_ij = x_ij / sqrt(Σ_i x_ij²)``，但本问题里四个准则
    在前沿上的**取值分布差异极大**：

    * ``gain ∈ [0.08, 0.82]``（跨一个数量级）；
    * ``time ∈ [0.52, 0.79]``（只有 ~1.5 倍差异，且绝对值都接近 1）。

    向量归一化按「列 L2 范数」缩放，而范数被较大的取值主导，于是 ``time`` 列被压缩到
    只占 ~0.03 的加权幅度，而 ``gain`` 列占 ~0.08——结果是 **``fast`` 预设
    （time 权重 0.45）也翻不动排序，三种 preset 会选出同一条路径**。

    因此这里采用 Hwang & Yoon 原文列出的另一种标准归一化——**线性极差变换
    （linear max-min scale transformation）**::

        r_ij = (x_ij - min_i x_ij) / (max_i x_ij - min_i x_ij)

    它把每个准则都拉伸到 ``[0,1]``，保证「时间短 60%」与「收益低 60%」在 TOPSIS
    里具有可比的权重语义；这也是让三种 preset 真正产生差异化路径的机制。
    若某准则在前沿上取值全同（极差 ≈ 0），该列统一置 0，此时正负理想解在该维相同，
    对排序没有影响。
    """
    n = len(front)
    if n == 0:
        return []
    values = objective_matrix(front)
    weight_vector = np.asarray([float(getattr(weights, key, 0.0)) for key in OBJECTIVE_KEYS])
    total_weight = float(weight_vector.sum())
    if total_weight <= _EPS:  # 权重全 0 时退化为等权
        weight_vector = np.ones(len(OBJECTIVE_KEYS))
        total_weight = float(weight_vector.sum())
    weight_vector = weight_vector / total_weight

    # 步骤 2：线性极差归一化（每个准则独立拉伸到 [0,1]）
    low = values.min(axis=0)
    high = values.max(axis=0)
    span = high - low
    normalised = np.divide(
        values - low,
        np.where(span > _EPS, span, 1.0),
        out=np.zeros_like(values),
        where=span > _EPS,
    )

    # 步骤 3：加权
    weighted = normalised * weight_vector

    # 步骤 4：正 / 负理想解（四个准则均为效益型）
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)

    # 步骤 5：到正负理想解的欧氏距离
    dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # 步骤 6：相对贴近度
    denominator = dist_best + dist_worst
    scores = np.divide(
        dist_worst,
        np.where(denominator > _EPS, denominator, 1.0),
        out=np.full(n, 0.5),
        where=denominator > _EPS,
    )
    return [float(value) for value in scores]


def select_compromise(front: Sequence[Any], weights: ObjectiveWeights) -> Any:
    """用 TOPSIS 从帕累托前沿中选出「妥协解」（最终推荐给学习者的路径）。

    Args:
        front: 帕累托前沿（非空）。
        weights: 该 preset 的目标权重。

    Returns:
        前沿中 ``C_i`` 最大的那个解（``front`` 的原对象，便于调用方保留
        ``steps`` / ``peload`` 等附加信息）。并列时取 ``front`` 中靠前者，
        保证结果可复现。

    Raises:
        ValueError: 前沿为空。
    """
    if not front:
        raise ValueError("帕累托前沿为空，无法选择妥协解")
    if len(front) == 1:
        return front[0]
    scores = topsis_scores(front, weights)
    best_index = 0
    best_score = scores[0]
    for index in range(1, len(scores)):
        if scores[index] > best_score + 1e-12:
            best_score = scores[index]
            best_index = index
    return front[best_index]


def weights_of(preset: str, presets: Mapping[str, ObjectiveWeights]) -> ObjectiveWeights:
    """按 preset 名取权重，未知 preset 回退到 ``balanced``（不存在则等权）。"""
    if preset in presets:
        return presets[preset]
    if "balanced" in presets:
        return presets["balanced"]
    return ObjectiveWeights()


__all__ = [
    "OBJECTIVE_KEYS",
    "SMOOTHNESS_JUMP_PENALTY",
    "ABILITY_LEVEL_TOLERANCE",
    "LearnerContext",
    "learner_context",
    "clamp01",
    "evaluate_objectives",
    "objective_matrix",
    "domination_matrix",
    "non_dominated_sort",
    "crowding_distance",
    "ParetoSolution",
    "simulated_binary_crossover",
    "polynomial_mutation",
    "nsga2",
    "topsis_scores",
    "select_compromise",
    "weights_of",
]
