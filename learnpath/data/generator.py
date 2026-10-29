"""完整数据集生成器（确定性、可复现）。

本模块按「先知识图谱、再课程、再目标、再学习者、最后行为日志」的顺序程序化
合成一份可用于推荐 / 规划 / 预测 / 分析全链路的在线学习平台数据集。所有随机性都
来自 ``random.Random(scale.seed + 偏移量)``，因此**同一个 seed 必然产出完全相同的数据**
（不依赖时间、不依赖网络、不依赖全局随机状态）。

生成机制（报告可直接引用）
--------------------------

**1. 知识点与先修图谱**

* 术语来源：7 个领域各备 64 个真实专业术语（见 :mod:`learnpath.data.vocab`），
  按难度 1~5 分档、每档 12~13 个；生成器按 ``1,2,3,4,5`` 难度**轮转抽取**，
  于是每个领域的知识点在 5 个难度层上均衡铺开（默认 420 个知识点 → 每领域 60 个）。
* 领域内结构：把领域内知识点按难度排序后，前 4 个作为**起点**（layer=0），
  其余知识点均匀铺到 layer=1..7。每个 layer=k 的知识点从 layer=k-1 中取 1~2 个
  先修父节点；同时**反向保证每个父节点至少有一个子节点**，因此叶子只可能出现在
  最后一层，叶子数量约等于「层宽 × 领域数」，不会碎成一地孤立点。
* 跨领域边：以领域级拓扑序
  ``数学基础 → 编程开发 → 数据与人工智能 → 语言学习 → 前端与设计 → 网络与安全 → 产品与职场``
  为约束添加，语义上就是「基础学科支撑应用学科」。其中约 1/3 指向对方领域的
  **入口知识点**（layer=0，会把整条链推深），其余指向中段知识点（更贴近真实课程
  的「横向前置」）。若最深链因此超过 11 层，则按确定顺序回退删除最深的跨领域边。
* 结果：领域内多棵 DAG（每领域 4 个起点）、无环、无孤立节点，
  ``max_depth`` 落在 7~11、``n_roots`` 落在 15~30、``n_leaves`` 落在 40~110。

**2. 课程**

* 课程数按领域知识点数比例分配；先保证**每个知识点至少被 1 门课程覆盖**
  （主课，权重 1.0），剩余课程做成「主知识点 + 1~3 个相邻知识点」的综合课
  （辅知识点权重 0.4~0.8），于是任何目标都能规划出课程路径。
* ``level`` = 覆盖知识点的最大难度（少量课程略高一级）；``duration_hours`` 随
  ``level`` 与覆盖数增长（2~40 小时）；``theory_ratio`` 随模态变化
  （项目实战偏低、图文课偏高）；``modality`` 四类轮转保证都存在；
  ``rating`` 用三角形分布右偏（峰在 4.55）；``enroll_count`` 取对数正态并截断到
  ``200~300000``；``price`` 取 ``{0,49,99,199,299}``。

**3. 职业目标**

* 8 个职业目标覆盖全部 7 个领域。每个目标用「**领域权重 + 先修深度区间 +
  关键词白/黑名单**」描述岗位语义：例如「数据分析师」= 数据与人工智能 72%
  （深度 3~6，排除大模型 / 强化学习 / 模型部署等研究类内容）+ 数学基础 20%
  （仅概率统计相关）+ 产品与职场 8%（仅看板 / 指标 / 实验相关）；
  「产品经理」只允许数据域的「指标 / 埋点 / 漏斗 / A/B / 看板」类浅中阶知识点。
  这样每个 goal 的目标知识点在**语义上都站得住**，报告里可以直接引用。
* 名额分配用「绝对缺口补齐」算法（:func:`_allocate_shares`），保证小权重领域
  不会被四舍五入成 0、主领域也不会被挤到超出设计占比（12 个名额 → 9/2/1）。
* 具体知识点挑法：在候选池内选若干条「根互不相同」的先修分支，从每条分支的
  **最深端**开始轮转取样，因此目标集合天然是连通的前修子结构，并横跨多个深度层。
* 最后做一次「闭包比率修复」：若 ``|目标闭包| / |目标集合| < 1.8``，就从主领域
  最浅的目标开始剥离（其先修仍留在闭包内），直到比率达标或达到目标数下限。

**4. 学习者**

* 目标在 8 个 goal 上均匀分布；``ability`` 由学历、专业对口度、工作年限与个体噪声
  合成后截断到 ``0.25~0.95``；``time_budget_hours_per_week`` 从
  ``{3,5,8,12,15,20}`` 加权抽样；``deadline_weeks`` = 目标默认周期 ± 4 周。
* ``initial_mastery`` 只覆盖**目标先修闭包内的低阶知识点**（difficulty ≤ 2），
  覆盖比例 15%~45%，比例与取值都由能力、专业对口度、工作年限和个体噪声决定，
  因此学习者之间存在明显差异（这正是推荐与规划能体现差别的来源）。

**5. 交互记录（learner-course）**

* 每位学习者按活跃度权重分配交互条数（最大余数法，总数精确等于
  ``scale.n_interactions``），候选课程 72% 来自「目标闭包覆盖的课程」，
  其余随机，保证行为分布有主题性又不至于极端。
* **完成概率由逻辑斯蒂模型生成**（因此可被机器学习学到）::

      z = 0.50·能力 + (-0.42)·(课程等级 − 当前能力等价等级)
          + (-0.30)·ln(1+课程时长) + 0.22·ln(每周预算/8)
          + 0.55·先修就绪度 + 0.30·模态匹配 + 0.42·课程质量 + 0.22·理论偏好匹配

      p_complete = sigmoid(z)，completed ~ Bernoulli(p_complete)

  其中「先修就绪度」按学习者**当时**的掌握度画像计算：随着学习推进，
  已学课程会按 BKT 风格抬高覆盖知识点的掌握度，后续课程的先修就绪度随之上升。
* 为了让数据既有信号又不「太干净」，采用**两遍法**：第一遍估计隐变量 ``z`` 的
  均值与标准差，第二遍按标定后的 ``z``（标准差固定为 ``_TARGET_LATENT_SIGMA``）
  抽样。这样整体完成率稳定在 50% 左右，而可学习性（AUC）落在 0.78~0.88 区间。
* 完成时 ``progress=1.0``，否则取 ``0.05~0.90`` 的低偏分布；``quiz_score``
  只在完成时按「能力 + 就绪度 + 课程等级」生成；``rating`` 有缺失（完成约 55%、
  未完成约 10% 给出评价）；时间戳在 2026-02-01 ~ 2026-09-28 内**按学习者内部
  时序严格递增**，最终整体按时间排序。
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np

from learnpath.config import (
    DATA_TIME_BASE,
    DEFAULT_SCALE,
    DOMAINS,
    MODALITIES,
    MOTIVATIONS,
    DatasetScale,
)
from learnpath.data import vocab
from learnpath.data.schema import (
    Background,
    Course,
    Dataset,
    Goal,
    Interaction,
    KPCoverage,
    KPEdge,
    KnowledgePoint,
    Learner,
    LearningStyle,
)
from learnpath.knowledge.graph import KnowledgeGraph

# ============================================================
# 0. 生成超参数（集中在此，便于报告中做敏感性说明）
# ============================================================

#: 逻辑斯蒂模型的隐变量目标标准差。AUC ≈ Φ(σ/√2)，1.35 对应约 0.83：
#: 既保证特征有足够信号（能被分类器学到），又保留个体随机性。
_TARGET_LATENT_SIGMA: float = 1.45

#: 交互时间窗（与 :data:`learnpath.config.DATA_TIME_BASE` 同一时间轴上）
_DATA_END: datetime = datetime(2026, 9, 28, 21, 0, 0)
_JOIN_WINDOW_DAYS: int = 19

#: 完成概率模型的标准化中心与尺度（先把各特征拉到 O(1) 量级，再线性组合）
_FEATURE_NAMES: tuple[str, ...] = (
    "learner_ability",
    "level_gap",
    "log_duration",
    "log_time_budget",
    "prereq_readiness",
    "modality_match",
    "course_quality",
    "theory_fit",
)
_FEATURE_CENTERS: tuple[float, ...] = (0.60, -0.40, 2.50, 0.00, 0.30, 0.32, 4.22, 0.72)
_FEATURE_SCALES: tuple[float, ...] = (0.20, 1.40, 0.60, 0.70, 0.30, 0.36, 0.34, 0.22)
_LOGIT_WEIGHTS: tuple[float, ...] = (0.50, -0.42, -0.30, 0.22, 0.55, 0.30, 0.42, 0.22)

#: 价格档位
_PRICE_TIERS: tuple[float, ...] = (0.0, 49.0, 99.0, 199.0, 299.0)
_PRICE_WEIGHTS: tuple[float, ...] = (0.50, 0.20, 0.17, 0.09, 0.04)

#: 每周可投入时长档位及其权重
_BUDGET_TIERS: tuple[float, ...] = (3.0, 5.0, 8.0, 12.0, 15.0, 20.0)
_BUDGET_WEIGHTS: tuple[float, ...] = (0.16, 0.22, 0.24, 0.18, 0.12, 0.08)

#: 跨领域先修关系（领域级拓扑序保证不会成环）
_CROSS_DOMAIN_PAIRS: tuple[tuple[str, str], ...] = (
    ("数学基础", "数据与人工智能"),
    ("编程开发", "数据与人工智能"),
    ("数学基础", "编程开发"),
    ("编程开发", "前端与设计"),
    ("编程开发", "网络与安全"),
    ("数据与人工智能", "产品与职场"),
    ("语言学习", "产品与职场"),
    ("数学基础", "网络与安全"),
    ("数据与人工智能", "网络与安全"),
    ("语言学习", "前端与设计"),
)

#: 图谱硬约束（默认规模下必须满足；小规模数据自动放宽）
_GRAPH_MAX_DEPTH: int = 11


# ============================================================
# 1. 小工具
# ============================================================


class GenerationError(RuntimeError):
    """生成过程中出现无法满足的约束时抛出。"""


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _fmt_time(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def _per_domain_counts(total: int) -> dict[str, int]:
    """把 ``total`` 个数量尽量均匀地分配到 7 个领域。"""
    base, remainder = divmod(total, len(DOMAINS))
    return {
        domain: base + (1 if index < remainder else 0)
        for index, domain in enumerate(DOMAINS)
    }


def _largest_remainder(total: int, weights: Sequence[float], floor: int) -> list[int]:
    """最大余数法分配整数：结果之和精确等于 ``total``，每项不小于 ``floor``。"""
    n = len(weights)
    if total < floor * n:
        raise GenerationError(f"总量 {total} 不足以给 {n} 个对象各分配 {floor} 个")
    weight_sum = sum(weights) or 1.0
    raw = [total * w / weight_sum for w in weights]
    counts = [max(floor, int(math.floor(r))) for r in raw]
    diff = total - sum(counts)
    order = sorted(range(n), key=lambda i: (-(raw[i] - math.floor(raw[i])), i))
    cursor = 0
    while diff > 0:
        counts[order[cursor % n]] += 1
        diff -= 1
        cursor += 1
    while diff < 0:
        for index in sorted(range(n), key=lambda i: (-counts[i], i)):
            if diff == 0:
                break
            if counts[index] > floor:
                counts[index] -= 1
                diff += 1
    return counts


# ============================================================
# 2. 知识点与先修图谱
# ============================================================


def _allocate_shares(total: int, weights: Sequence[float], floor: int = 1) -> list[int]:
    """按权重分配名额：先取整，再把余量补给「相对理想值最缺」的配额。

    与最大余数法的区别：这里按**绝对缺口**（理想值 − 已分配）轮流补，
    因此小权重领域不会被"四舍五入"成 0，主领域也不会被挤到超出设计占比
    （例如 12 个名额下 0.72/0.20/0.08 会稳定得到 9/2/1）。
    """
    n = len(weights)
    if total < floor * n:
        raise GenerationError(f"名额 {total} 不足以给 {n} 个配额各分配 {floor} 个")
    weight_sum = sum(weights) or 1.0
    ideal = [total * weight / weight_sum for weight in weights]
    counts = [max(floor, int(math.floor(value))) for value in ideal]
    guard = 0
    while sum(counts) < total and guard < total * 4:
        guard += 1
        index = max(
            range(n), key=lambda i: (ideal[i] - counts[i], weights[i], -i)
        )
        counts[index] += 1
    while sum(counts) > total:
        index = max(range(n), key=lambda i: (counts[i] - ideal[i], counts[i], -i))
        if counts[index] <= floor:
            break
        counts[index] -= 1
    return counts


def _choose_terms(domain: str, count: int) -> list[tuple[str, int]]:
    """按难度 1→5 轮转抽取 ``count`` 个 ``(术语, 难度)``。

    轮转抽样保证「每个难度层抽到的术语数量相当」，避免抽到 60 个术语时
    难度全挤在 1~2 层。领域内术语不足时提前结束（词汇表本身远大于需求量）。
    """
    tiers = vocab.KP_VOCAB[domain]
    level_order = (1, 2, 3, 4, 5)
    cursor = {level: 0 for level in level_order}
    chosen: list[tuple[str, int]] = []
    while len(chosen) < count:
        progressed = False
        for level in level_order:
            if len(chosen) >= count:
                break
            names = tiers[level]
            if cursor[level] < len(names):
                chosen.append((names[cursor[level]], level))
                cursor[level] += 1
                progressed = True
        if not progressed:
            break
    return chosen


def _layer_plan(count: int) -> tuple[int, int]:
    """返回 (领域内起点数, 层数)。默认规模下为 (4, 8)。"""
    if count >= 24:
        return 4, 8
    if count >= 10:
        return 2, min(5, count)
    return max(1, min(2, count - 1)), min(3, count)


def _build_knowledge_points(
    scale: DatasetScale,
) -> tuple[dict[str, KnowledgePoint], list[KPEdge], dict[str, int], KnowledgeGraph]:
    """构造知识点集合、先修边、层级映射与知识图谱。

    Returns:
        ``(kps, edges, layer_of, graph)``，``layer_of`` 是知识点所在的先修层
        （0 表示领域起点），课程与目标的生成都依赖它。
    """
    rng = random.Random(scale.seed + 1)
    counts = _per_domain_counts(scale.n_knowledge_points)

    kps: dict[str, KnowledgePoint] = {}
    edges: list[KPEdge] = []
    layer_of: dict[str, int] = {}
    domain_nodes: dict[str, list[str]] = {domain: [] for domain in DOMAINS}

    for domain_index, domain in enumerate(DOMAINS, start=1):
        terms = sorted(_choose_terms(domain, counts[domain]), key=lambda item: (item[1], item[0]))
        n_roots, n_layers = _layer_plan(len(terms))

        # ---- 2.1 分配先修层：起始若干知识点为 layer 0，其余均匀铺到后续层 ----
        nodes: list[str] = []
        for seq, (name, difficulty) in enumerate(terms, start=1):
            kp_id = f"KP-{domain_index:02d}-{seq:02d}"
            kps[kp_id] = KnowledgePoint(
                kp_id=kp_id,
                name=name,
                domain=domain,
                difficulty=difficulty,
                description=(
                    f"{domain}·{vocab.LEVEL_WORDING[difficulty]}知识点：{name}"
                ),
            )
            nodes.append(kp_id)
            domain_nodes[domain].append(kp_id)

        remaining = len(nodes) - n_roots
        upper_layers = max(1, n_layers - 1)
        for index, kp_id in enumerate(nodes):
            if index < n_roots:
                layer_of[kp_id] = 0
            else:
                offset = index - n_roots
                layer_of[kp_id] = 1 + (offset * upper_layers) // max(1, remaining)

        # ---- 2.2 领域内先修边：局域父节点 + 少量结构性汇合边 ----
        by_layer: dict[int, list[str]] = {}
        for kp_id in nodes:
            by_layer.setdefault(layer_of[kp_id], []).append(kp_id)

        used_pairs: set[tuple[str, str]] = set()
        has_child: set[str] = set()

        def _add_edge(source: str, target: str) -> None:
            if source == target or (source, target) in used_pairs:
                return
            used_pairs.add((source, target))
            has_child.add(source)
            edges.append(KPEdge(source=source, target=target))

        for layer in range(1, n_layers):
            parents = by_layer.get(layer - 1, [])
            children = by_layer.get(layer, [])
            if not parents or not children:
                continue
            width = len(parents)
            for index, child in enumerate(children):
                # 局域父节点：位置相近的上一层知识点，先修闭包因此是「窄锥形」
                base = (index * width) // len(children)
                _add_edge(parents[base], child)
                # 结构性汇合边：少量知识点同时依赖相邻分支，让图成为真正的 DAG
                # （既不是森林也不是全连接，闭包规模可控）
                if width > 2 and index % 4 == 3 and rng.random() < 0.75:
                    _add_edge(parents[(base + 2) % width], child)
            # 反向补齐：每个父节点至少有一个子节点，避免中间层出现「断头」
            for parent_index, parent in enumerate(parents):
                if parent in has_child:
                    continue
                fallback = children[(parent_index * len(children)) // len(parents)]
                _add_edge(parent, fallback)

    # ---- 2.3 跨领域先修边 ----
    n_cross = max(3, min(28, round(scale.n_knowledge_points * 0.05)))
    cross_edges: list[KPEdge] = []
    existing = {(edge.source, edge.target) for edge in edges}
    for index in range(n_cross):
        source_domain, target_domain = _CROSS_DOMAIN_PAIRS[index % len(_CROSS_DOMAIN_PAIRS)]
        source_pool = domain_nodes[source_domain]
        target_pool = domain_nodes[target_domain]
        if not source_pool or not target_pool:
            continue
        # 入口知识点取 layer=1（而不是 layer=0），这样每个领域内部的起点数量
        # 不会被跨领域边"吃掉"，同时又把整条先修链推深 1~2 层。
        entry_pool = [kp for kp in target_pool if layer_of[kp] == 1] or [
            kp for kp in target_pool if layer_of[kp] == 0
        ] or target_pool
        mid_pool = [kp for kp in target_pool if 2 <= layer_of[kp] <= 5] or target_pool

        # 每 3 条里放 1 条「指向入口知识点」的边：这类边会把整条先修链推深，
        # 是 max_depth 能达到 8~10 的主要原因。
        if index % 3 == 0:
            source_candidates = [kp for kp in source_pool if layer_of[kp] <= 1] or source_pool
            source = rng.choice(source_candidates)
            target = rng.choice(entry_pool)
        else:
            source_candidates = [kp for kp in source_pool if layer_of[kp] <= 2] or source_pool
            source = rng.choice(source_candidates)
            target = rng.choice(mid_pool)
        if source == target or (source, target) in existing:
            continue
        existing.add((source, target))
        cross_edges.append(KPEdge(source=source, target=target))

    # ---- 2.4 深度守卫：保证最深先修链不超过 _GRAPH_MAX_DEPTH ----
    def _build(extra: list[KPEdge]) -> KnowledgeGraph:
        return KnowledgeGraph(kps, list(edges) + list(extra))

    kept_cross = list(cross_edges)
    graph = _build(kept_cross)
    graph.validate()
    guard = 0
    while kept_cross and graph.stats().max_depth > _GRAPH_MAX_DEPTH and guard < 40:
        guard += 1
        depths = {kp_id: graph.depth(kp_id) for kp_id in kps}
        # 删掉「指向最深节点」的那条跨领域边（确定性：先比目标深度，再比 id）
        victim = max(
            kept_cross,
            key=lambda edge: (depths.get(edge.target, 0), edge.target, edge.source),
        )
        kept_cross = [edge for edge in kept_cross if edge is not victim]
        graph = _build(kept_cross)
        graph.validate()

    return kps, list(edges) + kept_cross, layer_of, graph


# ============================================================
# 3. 课程
# ============================================================


def _coverage_groups(kp_ids: list[str], n_courses: int) -> list[list[str]]:
    """把领域内知识点划分成 ``n_courses`` 个覆盖组，每组 1~4 个知识点且全覆盖。"""
    total = len(kp_ids)
    if total == 0:
        return []
    if n_courses * 4 < total:
        raise GenerationError(
            f"课程数 {n_courses} 太少，无法在「每门最多覆盖 4 个知识点」的前提下"
            f"覆盖 {total} 个知识点"
        )

    groups: list[list[str]] = []
    if n_courses >= total:
        # 每个知识点先有一门「主课」，剩余课程做成综合课
        groups = [[kp_id] for kp_id in kp_ids]
        for extra in range(n_courses - total):
            primary = (extra * 5) % total
            size = 2 + (extra % 3)
            members = [kp_ids[primary]]
            offset = 1
            while len(members) < size and offset < total:
                candidate = kp_ids[(primary + offset * 3) % total]
                if candidate not in members:
                    members.append(candidate)
                offset += 1
            groups.append(members)
    else:
        per_group = min(4, math.ceil(total / n_courses))
        cursor = 0
        while cursor < total:
            groups.append(kp_ids[cursor : cursor + per_group])
            cursor += per_group
        # 组数不足时把最长的组对半拆开，直到达到目标课程数
        while len(groups) < n_courses:
            target_index = max(range(len(groups)), key=lambda i: (len(groups[i]), -i))
            if len(groups[target_index]) < 2:
                break
            chunk = groups.pop(target_index)
            mid = len(chunk) // 2
            groups.insert(target_index, chunk[:mid])
            groups.insert(target_index + 1, chunk[mid:])
    return groups


def _build_courses(
    kps: dict[str, KnowledgePoint],
    layer_of: dict[str, int],
    scale: DatasetScale,
    rng: random.Random,
) -> dict[str, Course]:
    """生成课程库：主课保证知识点全覆盖，综合课负责制造课程间的关联。"""
    counts = _per_domain_counts(scale.n_courses)
    by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for kp in kps.values():
        by_domain[kp.domain].append(kp.kp_id)
    for domain in DOMAINS:
        by_domain[domain].sort(key=lambda kp_id: (layer_of[kp_id], kps[kp_id].difficulty, kp_id))

    courses: dict[str, Course] = {}
    used_titles: set[str] = set()
    level_tags = {1: "入门", 2: "基础", 3: "进阶", 4: "高级", 5: "高阶"}
    modality_theory = {
        "video": (0.35, 0.80),
        "text": (0.55, 0.95),
        "project": (0.10, 0.45),
        "mixed": (0.25, 0.70),
    }

    for domain in DOMAINS:
        groups = _coverage_groups(by_domain[domain], counts[domain])
        templates = vocab.COURSE_TEMPLATES[domain]
        titles = vocab.INSTRUCTOR_TITLES[domain]
        for index, members in enumerate(groups):
            covered = sorted(
                members, key=lambda kp_id: (-kps[kp_id].difficulty, layer_of[kp_id], kp_id)
            )
            main_kp = kps[covered[0]]
            aux_kp = kps[covered[1]] if len(covered) > 1 else None

            max_difficulty = max(kps[kp_id].difficulty for kp_id in covered)
            level = min(5, max_difficulty + (1 if rng.random() < 0.22 else 0))
            coverage = tuple(
                KPCoverage(
                    kp_id=kp_id,
                    weight=1.0 if position == 0 else round(rng.uniform(0.4, 0.8), 2),
                )
                for position, kp_id in enumerate(covered)
            )

            template = templates[index % len(templates)]
            title = template.format(
                kp=main_kp.name,
                kp2=aux_kp.name if aux_kp else main_kp.name,
                n=2 + (index % 9),
            )
            if title in used_titles:
                title = f"{title}（{level_tags[level]}·第 {1 + index % 12} 期）"
            suffix = 2
            while title in used_titles:
                title = f"{title}（{suffix}）"
                suffix += 1
            used_titles.add(title)

            duration = (2.0 + 2.9 * level + 1.9 * (len(covered) - 1)) * rng.uniform(0.75, 1.28)
            duration = round(_clip(duration, 2.0, 40.0), 1)
            modality = MODALITIES[index % len(MODALITIES)]
            low, high = modality_theory[modality]
            rating = round(_clip(rng.triangular(3.4, 4.95, 4.55), 3.4, 4.95), 2)
            enroll = int(_clip(rng.lognormvariate(8.1, 1.05), 200, 300000))

            tag_pool = vocab.TAG_POOL[domain]
            tag_count = rng.randint(2, 4)
            tags = {rng.choice(tag_pool), level_tags[level]}
            while len(tags) < tag_count:
                tags.add(rng.choice(tag_pool + vocab.GENERIC_TAGS))
            tags = tuple(sorted(tags))

            course_id = f"C-{len(courses) + 1:04d}"
            courses[course_id] = Course(
                course_id=course_id,
                title=title,
                domain=domain,
                level=level,
                duration_hours=duration,
                kp_coverage=coverage,
                theory_ratio=round(rng.uniform(low, high), 3),
                modality=modality,
                instructor=f"{rng.choice(vocab.INSTRUCTOR_NAMES)} · {rng.choice(titles)}",
                rating=rating,
                enroll_count=enroll,
                price=rng.choices(_PRICE_TIERS, weights=_PRICE_WEIGHTS, k=1)[0],
                tags=tags,
            )
    return courses


# ============================================================
# 4. 职业目标
# ============================================================


@dataclass(frozen=True, slots=True)
class _DomainQuota:
    """职业目标中某个领域的配额。

    Attributes:
        domain: 领域名。
        weight: 该领域在目标知识点中的占比。
        depth_min / depth_max: 允许的先修深度区间（0 为领域入口，越大越专业）。
        include_keywords: 仅接受名称含这些关键词的知识点（空表示不过滤）。
        exclude_keywords: 剔除名称含这些关键词的知识点。
    """

    domain: str
    weight: float
    depth_min: int
    depth_max: int
    include_keywords: tuple[str, ...] = ()
    exclude_keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _GoalSpec:
    """职业目标模板：按「领域权重 + 深度区间 + 关键词」描述语义边界。"""

    goal_id: str
    name: str
    target_role: str
    weeks: int
    n_target: int
    description: str
    quotas: tuple[_DomainQuota, ...]


#: 「技术负责人 / 平台架构」类知识点：属于资深架构岗，普通工程岗不应作为求职目标
_TECH_LEAD_KEYWORDS: tuple[str, ...] = (
    "架构评审",
    "技术规划",
    "技术协作",
    "跨团队治理",
    "团队建设",
    "组织建设",
)

#: 「重型平台 / 分布式」类知识点：偏架构与 SRE，非数据 / 产品 / 前端岗位的主线
_PLATFORM_KEYWORDS: tuple[str, ...] = (
    "分库分表",
    "分布式事务",
    "微服务",
    "消息队列",
    "服务治理",
    "缓存一致性",
    "领域驱动",
    "高并发",
    "容量评估",
    "持续集成",
    "云原生",
    "设计模式",
)

#: 「前沿模型研究」类知识点：算法研究岗内容，数据分析 / 产品岗不需要
_RESEARCH_KEYWORDS: tuple[str, ...] = (
    "神经网络",
    "卷积",
    "Transformer",
    "词向量",
    "多模态",
    "RAG",
    "向量数据库",
    "强化学习",
    "大模型",
    "分布式训练",
    "模型压缩",
    "MLOps",
    "数据漂移",
    "自然语言处理",
    "检索增强",
)

#: 「高等纯数学」类知识点：数学系研究内容，工程侧数学基础不需要
_PURE_MATH_KEYWORDS: tuple[str, ...] = (
    "测度论",
    "泛函",
    "拓扑",
    "算子",
    "随机微分",
    "随机矩阵",
    "变分法",
)

_GOAL_SPECS: tuple[_GoalSpec, ...] = (
    _GoalSpec(
        goal_id="goal_da",
        name="数据分析师",
        target_role="业务数据分析师",
        weeks=12,
        n_target=12,
        description="从数据清洗、统计推断到建模与可视化，能独立完成业务分析报告",
        quotas=(
            _DomainQuota(
                "数据与人工智能",
                0.72,
                3,
                6,
                include_keywords=(
                    "数据",
                    "SQL",
                    "统计",
                    "回归",
                    "模型评估",
                    "可视化",
                    "透视",
                    "特征",
                    "A/B",
                    "异常检测",
                    "聚类",
                    "降维",
                    "时间序列",
                    "Pandas",
                    "交叉验证",
                    "混淆矩阵",
                    "抽样",
                    "相关分析",
                ),
                exclude_keywords=_RESEARCH_KEYWORDS + ("模型部署", "特征仓库"),
            ),
            _DomainQuota(
                "数学基础",
                0.20,
                1,
                5,
                include_keywords=(
                    "概率",
                    "统计",
                    "假设检验",
                    "分布",
                    "期望",
                    "方差",
                    "回归",
                    "协方差",
                    "中心极限",
                    "极大似然",
                    "描述",
                ),
            ),
            _DomainQuota(
                "产品与职场",
                0.08,
                0,
                3,
                include_keywords=("看板", "指标", "实验", "漏斗", "埋点", "数据"),
            ),
        ),
    ),
    _GoalSpec(
        goal_id="goal_fe",
        name="前端工程师",
        target_role="前端开发工程师",
        weeks=16,
        n_target=12,
        description="掌握语义化结构、布局与组件化框架，能独立还原设计稿并做性能优化",
        quotas=(
            _DomainQuota(
                "前端与设计",
                0.75,
                3,
                6,
                exclude_keywords=_TECH_LEAD_KEYWORDS + ("品牌体系",),
            ),
            _DomainQuota(
                "编程开发",
                0.25,
                1,
                5,
                include_keywords=(
                    "模块",
                    "异步",
                    "测试",
                    "接口",
                    "类型",
                    "构建",
                    "工程",
                    "Git",
                    "代码",
                    "调试",
                    "正则",
                    "依赖",
                    "重构",
                    "前端",
                ),
                exclude_keywords=_PLATFORM_KEYWORDS + _TECH_LEAD_KEYWORDS,
            ),
        ),
    ),
    _GoalSpec(
        goal_id="goal_be",
        name="后端工程师",
        target_role="后端开发工程师",
        weeks=18,
        n_target=13,
        description="覆盖语言基础、数据结构、数据库与并发，具备服务端系统开发能力",
        quotas=(
            _DomainQuota(
                "编程开发",
                0.70,
                4,
                7,
                exclude_keywords=_TECH_LEAD_KEYWORDS,
            ),
            _DomainQuota(
                "网络与安全",
                0.30,
                2,
                6,
                include_keywords=(
                    "网络",
                    "IP",
                    "DNS",
                    "HTTP",
                    "TCP",
                    "UDP",
                    "TLS",
                    "HTTPS",
                    "代理",
                    "防火墙",
                    "SQL 注入",
                    "XSS",
                    "CSRF",
                    "会话",
                    "Cookie",
                    "密码",
                    "加密",
                    "OAuth",
                    "JWT",
                    "SSO",
                    "权限模型",
                    "安全编码",
                    "日志",
                    "抓包",
                    "编码基础",
                    "排错",
                ),
                exclude_keywords=(
                    "APT",
                    "威胁狩猎",
                    "攻防演练",
                    "红蓝对抗",
                    "渗透测试",
                    "安全体系",
                    "合规审计",
                    "安全运营中心",
                    "风险评估",
                    "资产测绘",
                    "取证",
                    "威胁情报",
                ),
            ),
        ),
    ),
    _GoalSpec(
        goal_id="goal_ml",
        name="算法工程师",
        target_role="机器学习算法工程师",
        weeks=24,
        n_target=14,
        description="以数学与统计为底座，掌握特征工程、模型训练与部署的完整链路",
        quotas=(
            _DomainQuota(
                "数据与人工智能",
                0.65,
                3,
                6,
                exclude_keywords=("模型压缩", "MLOps", "数据漂移"),
            ),
            _DomainQuota(
                "数学基础",
                0.35,
                3,
                7,
                include_keywords=(
                    "概率",
                    "统计",
                    "分布",
                    "期望",
                    "方差",
                    "矩阵",
                    "特征值",
                    "向量空间",
                    "SVD",
                    "二次型",
                    "凸",
                    "梯度",
                    "拉格朗日",
                    "最优化",
                    "KKT",
                    "蒙特卡洛",
                    "马尔可夫",
                    "信息论",
                    "数值",
                    "极大似然",
                    "假设检验",
                    "回归",
                    "概率图",
                ),
                exclude_keywords=_PURE_MATH_KEYWORDS,
            ),
        ),
    ),
    _GoalSpec(
        goal_id="goal_pm",
        name="产品经理",
        target_role="互联网产品经理",
        weeks=10,
        n_target=11,
        description="从需求调研到增长实验，能够用数据驱动产品决策与跨团队推进",
        quotas=(
            _DomainQuota(
                "产品与职场",
                0.75,
                3,
                6,
                exclude_keywords=(
                    "组织建设",
                    "组织管理",
                    "团队建设",
                    "数字化转型",
                    "第二曲线",
                    "产品组合",
                    "投资决策",
                    "高阶谈判",
                    "行业趋势",
                    "产品负责人",
                    "战略拆解",
                ),
            ),
            _DomainQuota(
                "数据与人工智能",
                0.25,
                1,
                5,
                include_keywords=(
                    "指标",
                    "埋点",
                    "漏斗",
                    "A/B",
                    "看板",
                    "实验",
                    "描述性",
                    "数据质量",
                    "可视化",
                    "分组聚合",
                ),
                exclude_keywords=_RESEARCH_KEYWORDS
                + ("聚类", "降维", "推荐系统", "异常检测"),
            ),
        ),
    ),
    _GoalSpec(
        goal_id="goal_sec",
        name="网络安全工程师",
        target_role="网络安全工程师",
        weeks=16,
        n_target=12,
        description="协议与攻防双线并进，覆盖加固、审计、应急响应与合规要求",
        quotas=(
            _DomainQuota("网络与安全", 0.75, 4, 7),
            _DomainQuota(
                "编程开发",
                0.25,
                1,
                5,
                exclude_keywords=_PLATFORM_KEYWORDS + _TECH_LEAD_KEYWORDS,
            ),
        ),
    ),
    _GoalSpec(
        goal_id="goal_biz",
        name="商务英语",
        target_role="外贸/跨国团队业务骨干",
        weeks=8,
        n_target=9,
        description="面向商务场景的听说读写强化，覆盖邮件、会议、谈判与演讲",
        quotas=(
            _DomainQuota(
                "语言学习",
                1.0,
                1,
                7,
                include_keywords=(
                    "商务",
                    "会议",
                    "谈判",
                    "演讲",
                    "跨文化",
                    "面试",
                    "术语",
                    "路演",
                    "电话",
                    "邮件",
                    "高管",
                    "国际",
                ),
                exclude_keywords=(
                    "学术",
                    "雅思",
                    "托福",
                    "翻译",
                    "笔译",
                    "口译",
                    "同传",
                    "语料库",
                    "教学法",
                    "语言学",
                    "测试与评估",
                ),
            ),
        ),
    ),
    _GoalSpec(
        goal_id="goal_math",
        name="数学基础强化",
        target_role="机器学习/量化方向预备",
        weeks=10,
        n_target=10,
        description="系统补强微积分、线性代数、概率统计与最优化，为算法学习打底",
        quotas=(
            _DomainQuota(
                "数学基础",
                1.0,
                2,
                7,
                include_keywords=(
                    "矩阵",
                    "行列式",
                    "特征值",
                    "向量",
                    "概率",
                    "分布",
                    "期望",
                    "方差",
                    "协方差",
                    "中心极限",
                    "大数",
                    "极大似然",
                    "假设检验",
                    "凸",
                    "梯度",
                    "拉格朗日",
                    "牛顿",
                    "KKT",
                    "最优化",
                    "SVD",
                    "信息论",
                    "蒙特卡洛",
                    "马尔可夫",
                    "数值",
                    "回归",
                    "二次型",
                    "导数",
                    "积分",
                    "函数",
                    "级数",
                    "集合",
                ),
                exclude_keywords=_PURE_MATH_KEYWORDS,
            ),
        ),
    ),
)


def _path_to_root(graph: KnowledgeGraph, kp_id: str) -> list[str]:
    """从知识点出发沿「id 最小的先修」回溯到领域起点（确定性，含自身）。"""
    path = [kp_id]
    node = kp_id
    seen = {node}
    for _ in range(64):
        preds = graph.predecessors(node)
        if not preds:
            break
        node = min(preds)
        if node in seen:
            break
        seen.add(node)
        path.append(node)
    return path


def _pick_branch_paths(
    graph: KnowledgeGraph,
    layer_of: dict[str, int],
    domain_kps: list[str],
    n_branch: int,
    allowed: set[str] | None = None,
) -> list[list[str]]:
    """挑选若干条「根互不相同」的先修分支，返回自叶子到根的路径。

    Args:
        domain_kps: 可作「分支端点」的候选知识点。
        allowed: 若给出，路径会被裁剪为只保留该集合内的知识点
            （用于把目标限制在某个领域 / 深度区间 / 关键词白名单内）。
    """
    ranked = sorted(
        domain_kps,
        key=lambda kp_id: (-layer_of[kp_id], -graph.difficulty_of(kp_id), kp_id),
    )
    paths: list[list[str]] = []
    used_roots: set[str] = set()
    for tip in ranked:
        if layer_of[tip] < 1:
            continue
        path = _path_to_root(graph, tip)
        if allowed is not None:
            path = [kp_id for kp_id in path if kp_id in allowed]
        if len(path) < 2:
            continue
        root = path[-1]
        if root in used_roots:
            continue
        used_roots.add(root)
        paths.append(path)
        if len(paths) >= n_branch:
            break
    if not paths and ranked:
        fallback = _path_to_root(graph, ranked[0])
        if allowed is not None:
            fallback = [kp_id for kp_id in fallback if kp_id in allowed] or [ranked[0]]
        paths.append(fallback)
    return paths


def _goal_candidates(
    kps: dict[str, KnowledgePoint],
    layer_of: dict[str, int],
    quota: _DomainQuota,
    quantity: int,
) -> list[str]:
    """按配额构造候选知识点集合；候选不足时按「关键词 → 深度 → 领域」逐级放宽。"""
    pool = sorted(kp_id for kp_id, kp in kps.items() if kp.domain == quota.domain)
    if not pool:
        return []

    def _excluded(kp_id: str) -> bool:
        name = kps[kp_id].name
        return any(word in name for word in quota.exclude_keywords)

    def _keyword_ok(kp_id: str) -> bool:
        if _excluded(kp_id):
            return False
        if not quota.include_keywords:
            return True
        name = kps[kp_id].name
        return any(word in name for word in quota.include_keywords)

    def _depth_ok(kp_id: str) -> bool:
        return quota.depth_min <= layer_of[kp_id] <= quota.depth_max

    relaxed_groups = [
        [kp_id for kp_id in pool if _keyword_ok(kp_id) and _depth_ok(kp_id)],
        [kp_id for kp_id in pool if _keyword_ok(kp_id)],
        [kp_id for kp_id in pool if _depth_ok(kp_id) and not _excluded(kp_id)],
        [kp_id for kp_id in pool if not _excluded(kp_id)],
    ]
    widest: list[str] = []
    for group in relaxed_groups:
        if len(group) >= max(quantity, 2):
            return group
        if len(group) > len(widest):
            widest = group
    return widest


def _select_domain_targets(
    graph: KnowledgeGraph,
    layer_of: dict[str, int],
    pool: list[str],
    quantity: int,
) -> list[str]:
    """在候选池内选出 ``quantity`` 个目标：从各分支的最深端开始轮转，保证有深度梯度。"""
    if quantity <= 0 or not pool:
        return []
    allowed = set(pool)
    n_branch = max(1, min(4, math.ceil(quantity / 2)))
    paths = _pick_branch_paths(graph, layer_of, pool, n_branch, allowed=allowed)

    targets: list[str] = []
    round_index = 0
    while len(targets) < quantity and round_index < 16:
        added = False
        for path in paths:
            if round_index >= len(path):
                continue
            node = path[round_index]
            if node in targets:
                continue
            targets.append(node)
            added = True
            if len(targets) >= quantity:
                break
        if not added:
            break
        round_index += 1

    if len(targets) < quantity:  # 兜底：按深度降序补齐
        for kp_id in sorted(pool, key=lambda kp: (-layer_of[kp], kp)):
            if kp_id not in targets:
                targets.append(kp_id)
            if len(targets) >= quantity:
                break
    return targets[:quantity]


def _build_goals(
    kps: dict[str, KnowledgePoint],
    layer_of: dict[str, int],
    graph: KnowledgeGraph,
    rng: random.Random,
) -> tuple[dict[str, Goal], dict[str, frozenset[str]]]:
    """构造职业目标，并返回每个目标的知识点闭包（供学习者建模复用）。

    每个目标按 :class:`_DomainQuota` 的领域权重分配名额，再用「深度区间 + 关键词
    白/黑名单」把候选限制在**语义合理**的知识点上（例如数据分析师不会要求
    「数据库分库分表」，产品经理不会要求「词向量与文本分类」）。
    """
    by_domain: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    for kp in kps.values():
        by_domain[kp.domain].append(kp.kp_id)

    max_available = max(len(by_domain[domain]) for domain in DOMAINS)
    goals: dict[str, Goal] = {}
    closures: dict[str, frozenset[str]] = {}

    for spec in _GOAL_SPECS:
        quotas = [quota for quota in spec.quotas if by_domain[quota.domain]]
        if not quotas:
            quotas = [_DomainQuota(DOMAINS[0], 1.0, 0, 7)]
        n_target = max(len(quotas), min(spec.n_target, max_available - 1))

        # ---- 名额分配：按领域权重分配，保证每个领域至少 1 个且主领域占比不被挤偏 ----
        shares = _allocate_shares(n_target, [quota.weight for quota in quotas], floor=1)

        targets: list[str] = []
        quota_of_kp: dict[str, int] = {}
        for quota_index, (quota, quantity) in enumerate(zip(quotas, shares)):
            pool = _goal_candidates(kps, layer_of, quota, quantity)
            picked = _select_domain_targets(graph, layer_of, pool, quantity)
            for kp_id in picked:
                if kp_id not in quota_of_kp:
                    targets.append(kp_id)
                    quota_of_kp[kp_id] = quota_index

        min_targets = max(len(quotas), min(8, n_target))
        if len(targets) < min_targets:
            fallback = sorted(
                (kp_id for quota in quotas for kp_id in by_domain[quota.domain]),
                key=lambda kp_id: (-layer_of[kp_id], kp_id),
            )
            for kp_id in fallback:
                if kp_id not in quota_of_kp:
                    targets.append(kp_id)
                    quota_of_kp[kp_id] = 0
                if len(targets) >= min_targets:
                    break

        # 目标按「深度降序」稳定排列，便于后续做闭包比率修复
        targets.sort(key=lambda kp_id: (-layer_of[kp_id], kp_id))
        closure = graph.prerequisite_closure(targets)

        # ---- 闭包比率修复：先修闭包应明显大于目标集合，路径规划才有意义 ----
        # 只从「名额最多」的领域剥离最浅目标，避免破坏主/次领域的构成比例。
        primary_index = max(range(len(shares)), key=lambda index: (shares[index], -index))
        while len(targets) > min_targets and len(closure) / len(targets) < 1.8:
            droppable = [
                kp_id
                for kp_id in targets
                if quota_of_kp[kp_id] == primary_index
                and sum(1 for value in quota_of_kp.values() if value == primary_index) > 1
            ]
            if not droppable:
                break
            shallowest = min(droppable, key=lambda kp_id: (layer_of[kp_id], kp_id))
            reduced = [kp_id for kp_id in targets if kp_id != shallowest]
            reduced_closure = graph.prerequisite_closure(reduced)
            if len(reduced_closure) / len(reduced) <= len(closure) / len(targets):
                break
            targets = reduced
            closure = reduced_closure
            del quota_of_kp[shallowest]

        goals[spec.goal_id] = Goal(
            goal_id=spec.goal_id,
            name=spec.name,
            target_role=spec.target_role,
            description=spec.description,
            target_kp_ids=tuple(targets),
            default_weeks=spec.weeks,
            required_mastery=round(rng.uniform(0.75, 0.85), 2),
        )
        closures[spec.goal_id] = closure

    return goals, closures


# ============================================================
# 5. 学习者
# ============================================================

_EDU_BONUS: dict[str, float] = {"专科": 0.0, "本科": 0.05, "硕士": 0.11, "博士": 0.16}


def _build_learners(
    kps: dict[str, KnowledgePoint],
    goals: dict[str, Goal],
    closures: dict[str, frozenset[str]],
    scale: DatasetScale,
    rng: random.Random,
) -> dict[str, Learner]:
    """生成学习者画像（背景、能力、时间预算、初始掌握度）。"""
    goal_ids = list(goals)
    # 目标的主领域：目标知识点中出现次数最多的领域
    goal_domain: dict[str, str] = {}
    for goal_id, goal in goals.items():
        tally: dict[str, int] = {}
        for kp_id in goal.target_kp_ids:
            domain = kps[kp_id].domain
            tally[domain] = tally.get(domain, 0) + 1
        goal_domain[goal_id] = max(sorted(tally), key=lambda d: tally[d])

    education_names = [name for name, _ in vocab.EDUCATION_LEVELS]
    education_weights = [weight for _, weight in vocab.EDUCATION_LEVELS]

    learners: dict[str, Learner] = {}
    used_nicknames: set[str] = set()

    for index in range(scale.n_learners):
        goal_id = goal_ids[index % len(goal_ids)]
        goal = goals[goal_id]
        primary_domain = goal_domain[goal_id]

        education = rng.choices(education_names, weights=education_weights, k=1)[0]
        major_match = rng.random() < 0.55
        if major_match:
            major = rng.choice(vocab.MAJORS_BY_DOMAIN[primary_domain])
        else:
            major = rng.choice(vocab.CROSS_MAJORS)

        work_years = rng.choices(
            (0, 1, 2, 3, 5, 8, 12), weights=(0.22, 0.16, 0.16, 0.16, 0.14, 0.10, 0.06), k=1
        )[0]
        if work_years == 0:
            current_role = "在校学生"
        else:
            current_role = rng.choice(vocab.ROLES_BY_DOMAIN[primary_domain])

        ability_raw = (
            0.42
            + _EDU_BONUS[education]
            + 0.012 * min(work_years, 10)
            + (0.07 if major_match else 0.0)
            + rng.gauss(0.0, 0.115)
        )
        ability = round(_clip(ability_raw, 0.25, 0.95), 4)

        # ---- 初始掌握度：只覆盖目标先修闭包内的低阶知识点 ----
        closure = closures[goal_id]
        low_level = [kp_id for kp_id in closure if kps[kp_id].difficulty <= 2]
        if not low_level:
            low_level = [kp_id for kp_id in closure if kps[kp_id].difficulty <= 3]
        if not low_level:
            low_level = sorted(closure)
        low_level.sort()

        coverage_ratio = _clip(
            0.15
            + 0.30 * (ability - 0.25) / 0.70
            + (0.06 if major_match else 0.0)
            + 0.004 * min(work_years, 8)
            + rng.uniform(-0.07, 0.07),
            0.15,
            0.45,
        )
        pick_count = int(round(coverage_ratio * len(low_level)))
        pick_count = max(1, min(len(low_level), pick_count))
        picked = sorted(rng.sample(low_level, pick_count))
        initial_mastery: dict[str, float] = {}
        for kp_id in picked:
            base = (
                0.45 * ability
                + 0.30 * (1.0 if major_match else 0.35)
                + 0.10 * min(work_years, 6) / 6.0
                + 0.25 * rng.random()
            )
            initial_mastery[kp_id] = round(_clip(0.20 + 0.62 * base, 0.20, 0.95), 4)

        # ---- 学习风格 / 动机 / 兴趣 ----
        time_budget = rng.choices(_BUDGET_TIERS, weights=_BUDGET_WEIGHTS, k=1)[0]
        weekly_days = int(_clip(round(time_budget / 3.0) + rng.choice((-1, 0, 0, 1)), 1, 7))
        if work_years == 0:
            motivation = rng.choices(("学业", "兴趣", "求职", "转行"), weights=(0.55, 0.25, 0.16, 0.04), k=1)[0]
        else:
            motivation = rng.choices(
                MOTIVATIONS, weights=(0.30, 0.24, 0.20, 0.06, 0.20), k=1
            )[0]

        interests = {primary_domain}
        if rng.random() < 0.6:
            other = DOMAINS[rng.randrange(len(DOMAINS))]
            if other != primary_domain:
                interests.add(other)

        # ---- 昵称：形容词 + 名词，冲突时加数字后缀 ----
        nickname = f"{rng.choice(vocab.NICK_ADJ)}{rng.choice(vocab.NICK_NOUN)}"
        if nickname in used_nicknames:
            nickname = f"{nickname}{index % 97 + 1:02d}"
        suffix = 2
        while nickname in used_nicknames:
            nickname = f"{rng.choice(vocab.NICK_ADJ)}{rng.choice(vocab.NICK_NOUN)}{suffix}"
            suffix += 1
        used_nicknames.add(nickname)

        joined = datetime.fromisoformat(DATA_TIME_BASE) + timedelta(
            days=rng.randint(0, _JOIN_WINDOW_DAYS),
            hours=rng.randint(8, 22),
            minutes=rng.choice((0, 15, 30, 45)),
        )

        learner_id = f"L-{index + 1:04d}"
        learners[learner_id] = Learner(
            learner_id=learner_id,
            nickname=nickname,
            background=Background(
                education=education,
                major=major,
                work_years=work_years,
                current_role=current_role,
            ),
            goal_id=goal_id,
            motivation=motivation,
            deadline_weeks=max(4, goal.default_weeks + rng.choice((-4, -3, -2, -1, 0, 0, 1, 2, 3, 4))),
            time_budget_hours_per_week=time_budget,
            learning_style=LearningStyle(
                theory_preference=round(rng.uniform(0.12, 0.88), 3),
                prefer_modality=rng.choices(
                    MODALITIES, weights=(0.42, 0.20, 0.28, 0.10), k=1
                )[0],
                session_len_min=rng.choice((25, 30, 45, 60, 90)),
                weekly_days=weekly_days,
            ),
            ability=ability,
            initial_mastery=initial_mastery,
            interests=tuple(sorted(interests)),
            joined_at=_fmt_time(joined),
        )
    return learners


# ============================================================
# 6. 交互记录（逻辑斯蒂完成率模型）
# ============================================================


def _completion_features(
    learner: Learner, course: Course, readiness: float
) -> tuple[float, ...]:
    """完成概率模型的 8 个自变量（未标准化）。"""
    ability = learner.ability
    level_gap = float(course.level) - (1.0 + 4.0 * ability)
    log_duration = math.log1p(float(course.duration_hours))
    log_budget = math.log(max(1e-6, learner.time_budget_hours_per_week / 8.0))
    preferred = learner.learning_style.prefer_modality
    if course.modality == preferred:
        modality_match = 1.0
    elif course.modality == "mixed" or preferred == "mixed":
        modality_match = 0.5
    else:
        modality_match = 0.0
    theory_fit = 1.0 - abs(float(course.theory_ratio) - learner.learning_style.theory_preference)
    return (
        ability,
        level_gap,
        log_duration,
        log_budget,
        readiness,
        modality_match,
        float(course.rating),
        theory_fit,
    )


def _latent_score(features: Sequence[float]) -> float:
    """标准化后线性组合，得到逻辑斯蒂模型的隐变量 ``z``。"""
    z = 0.0
    for value, center, scale, weight in zip(
        features, _FEATURE_CENTERS, _FEATURE_SCALES, _LOGIT_WEIGHTS
    ):
        z += weight * ((value - center) / scale)
    return z


def _readiness(mastery: dict[str, float], prereq_ids: Sequence[str]) -> float:
    """先修就绪度：先修知识点的掌握度达到 0.55 视为「已就绪」。"""
    if not prereq_ids:
        return 1.0
    total = 0.0
    for kp_id in prereq_ids:
        total += min(1.0, mastery.get(kp_id, 0.05) / 0.55)
    return total / len(prereq_ids)


def _update_mastery(
    mastery: dict[str, float], course: Course, completed: bool, ability: float
) -> None:
    """BKT 风格的掌握度更新：完成课程按能力抬升，未完成仅小幅增长。"""
    for kp_id, weight in course.kp_weights.items():
        current = mastery.get(kp_id, 0.10)
        if completed:
            gain = (0.95 - current) * (0.25 + 0.50 * ability) * float(weight)
        else:
            gain = (0.95 - current) * 0.05 * float(weight)
        mastery[kp_id] = min(1.0, current + gain)


def _simulate(
    scale: DatasetScale,
    learners: dict[str, Learner],
    courses: dict[str, Course],
    counts: dict[str, int],
    goal_pools: dict[str, list[str]],
    course_prereq: dict[str, tuple[str, ...]],
    calibrate: tuple[float, float] | None,
    seed: int,
) -> list[tuple[Any, ...]]:
    """执行一遍「学习者—课程」模拟。

    Args:
        calibrate: ``(中心, 缩放系数)``；为 ``None`` 时直接用原始 ``z`` 抽样
            （第一遍），否则用标定后的 ``z``（第二遍，正式输出）。
        seed: 随机种子。两遍使用同一 seed，保证课程选择序列一致。

    Returns:
        记录列表，每条记录为
        ``(learner_id, course_id, timestamp, completed, progress, rating,
        hours_spent, quiz_score, features, raw_z)``。
    """
    rng = random.Random(seed)
    course_ids = list(courses)
    records: list[tuple[Any, ...]] = []

    for learner in learners.values():
        mastery = dict(learner.initial_mastery)
        pool = goal_pools[learner.goal_id]
        used: set[str] = set()
        start = datetime.fromisoformat(learner.joined_at) + timedelta(days=1, hours=rng.randint(0, 6))
        cursor = start
        remaining_total = counts[learner.learner_id]

        for step in range(remaining_total):
            # ---- 候选课程：72% 落在目标闭包相关课程上 ----
            if pool and rng.random() < 0.72:
                course_id = pool[rng.randrange(len(pool))]
            else:
                course_id = course_ids[rng.randrange(len(course_ids))]
            tries = 0
            while course_id in used and tries < 48:
                course_id = course_ids[rng.randrange(len(course_ids))]
                tries += 1
            if course_id in used:  # 兜底：线性扫描第一个未学课程
                for candidate in pool + course_ids:
                    if candidate not in used:
                        course_id = candidate
                        break
            used.add(course_id)
            course = courses[course_id]

            # ---- 时间推进：后一条必然晚于前一条，且整体落在时间窗内 ----
            remaining = remaining_total - step
            days_left = max(0.0, (_DATA_END - cursor).total_seconds() / 86400.0)
            step_days = min(4.2, max(0.25, days_left / max(1, remaining) * 0.85))
            cursor = cursor + timedelta(days=step_days * rng.uniform(0.55, 1.45))
            if cursor > _DATA_END:
                cursor = _DATA_END - timedelta(hours=1.5 * max(0, remaining - 1))
            cursor = cursor.replace(microsecond=0)

            # ---- 逻辑斯蒂完成概率 ----
            readiness = _readiness(mastery, course_prereq[course_id])
            features = _completion_features(learner, course, readiness)
            raw_z = _latent_score(features)
            if calibrate is None:
                z = raw_z
            else:
                center, factor = calibrate
                z = (raw_z - center) * factor
            p_complete = _sigmoid(z)
            completed = rng.random() < p_complete

            # ---- 结果字段 ----
            if completed:
                progress = 1.0
            else:
                progress = round(_clip(0.05 + 0.85 * (rng.random() ** 1.7), 0.05, 0.90), 4)

            theory_fit = features[7]
            if completed and rng.random() < 0.86:
                quiz = _clip(
                    0.30
                    + 0.42 * learner.ability
                    + 0.18 * theory_fit
                    + 0.10 * readiness
                    - 0.045 * (course.level - 1)
                    + rng.gauss(0.0, 0.07),
                    0.0,
                    1.0,
                )
                quiz_score: float | None = round(quiz, 4)
            else:
                quiz_score = None

            give_rating = (completed and rng.random() < 0.55) or (
                (not completed) and rng.random() < 0.10
            )
            if give_rating:
                baseline = 3.0 + 1.8 * (quiz_score if quiz_score is not None else 0.62)
                rating_value = _clip(
                    0.58 * course.rating + 0.42 * baseline + rng.gauss(0.0, 0.24), 1.0, 5.0
                )
                course_rating: float | None = round(rating_value, 1)
            else:
                course_rating = None

            if completed:
                hours = float(course.duration_hours) * rng.uniform(0.95, 1.22)
            else:
                hours = float(course.duration_hours) * progress * rng.uniform(0.60, 0.98)

            records.append(
                (
                    learner.learner_id,
                    course_id,
                    _fmt_time(cursor),
                    bool(completed),
                    float(progress),
                    course_rating,
                    round(hours, 2),
                    quiz_score,
                    features,
                    raw_z,
                )
            )

            _update_mastery(mastery, course, bool(completed), learner.ability)

    return records


def _build_interactions(
    scale: DatasetScale,
    kps: dict[str, KnowledgePoint],
    courses: dict[str, Course],
    learners: dict[str, Learner],
    goals: dict[str, Goal],
    graph: KnowledgeGraph,
) -> tuple[list[Interaction], list[list[float]], list[bool]]:
    """生成交互日志，并返回用于自评可学习性的特征矩阵与标签。"""
    # 课程先修闭包（不含自身覆盖的知识点）
    course_prereq: dict[str, tuple[str, ...]] = {}
    for course_id, course in courses.items():
        closure = graph.prerequisite_closure(course.kp_ids)
        course_prereq[course_id] = tuple(sorted(closure - set(course.kp_ids)))

    # 每个目标的「相关课程池」
    goal_closure: dict[str, frozenset[str]] = {
        goal_id: graph.prerequisite_closure(goal.target_kp_ids)
        for goal_id, goal in goals.items()
    }
    goal_pools: dict[str, list[str]] = {}
    for goal_id, closure in goal_closure.items():
        goal_pools[goal_id] = [
            course_id
            for course_id, course in courses.items()
            if any(kp_id in closure for kp_id in course.kp_ids)
        ]
        if not goal_pools[goal_id]:
            goal_pools[goal_id] = list(courses)

    # 交互条数分配：活跃度由时间预算、动机与能力共同决定
    motivation_weight = {"求职": 1.25, "晋升": 1.1, "转行": 1.2, "学业": 1.0, "兴趣": 0.9}
    learner_ids = list(learners)
    weights = []
    for learner_id in learner_ids:
        learner = learners[learner_id]
        weights.append(
            0.55
            + 0.42 * (learner.time_budget_hours_per_week / 20.0)
            + 0.25 * (motivation_weight[learner.motivation] - 1.0)
            + 0.35 * learner.ability
        )
    floor = 1
    counts = dict(
        zip(
            learner_ids,
            _largest_remainder(scale.n_interactions, weights, floor),
        )
    )
    for learner_id in learner_ids:
        counts[learner_id] = min(counts[learner_id], len(courses))

    seed = scale.seed + 11
    # ---- 第一遍：估计隐变量 z 的均值与标准差（用于标定）----
    first_pass = _simulate(
        scale, learners, courses, counts, goal_pools, course_prereq, None, seed
    )
    raw_latents = [row[9] for row in first_pass]
    mean_z = sum(raw_latents) / max(1, len(raw_latents))
    variance = sum((z - mean_z) ** 2 for z in raw_latents) / max(1, len(raw_latents))
    std_z = math.sqrt(variance) or 1.0
    calibrate = (mean_z, _TARGET_LATENT_SIGMA / std_z)

    # ---- 第二遍：按标定后的模型正式抽样 ----
    records = _simulate(
        scale, learners, courses, counts, goal_pools, course_prereq, calibrate, seed
    )
    # 按时间排序（同一时刻按学习者 id 稳定排序），并让特征矩阵保持同序，
    # 否则后续用「特征 → 标签」训练分类器会出现错位。
    records.sort(key=lambda row: (row[2], row[0]))

    interactions = [
        Interaction(
            learner_id=row[0],
            course_id=row[1],
            progress=row[4],
            completed=row[3],
            rating=row[5],
            hours_spent=row[6],
            quiz_score=row[7],
            timestamp=row[2],
        )
        for row in records
    ]
    features = [list(row[8]) for row in records]
    labels = [row[3] for row in records]
    return interactions, features, labels


# ============================================================
# 7. 完成概率的可学习性自评（纯 numpy，结果写入 meta）
# ============================================================


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """基于秩（Mann-Whitney U）的 AUC，支持并列分数取平均秩。"""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks_sorted = np.empty(len(s), dtype=float)
    index = 0
    while index < len(s):
        end = index
        while end + 1 < len(s) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        ranks_sorted[index : end + 1] = (index + end) / 2.0 + 1.0
        index = end + 1
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ranks_sorted
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fit_logistic(design: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 60) -> np.ndarray:
    """带 L2 正则的牛顿法（IRLS）逻辑回归，截距不惩罚。"""
    n, d = design.shape
    weights = np.zeros(d, dtype=float)
    reg = np.full(d, l2, dtype=float)
    reg[0] = 0.0
    for _ in range(iters):
        z = design @ weights
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        gradient = design.T @ (p - y) / n + reg * weights / n
        curvature = p * (1.0 - p) + 1e-9
        hessian = (design * curvature[:, None]).T @ design / n + np.diag(reg) / n
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return weights


def build_completion_features(dataset: Dataset) -> tuple[list[list[float]], list[bool]]:
    """从已生成的数据集重放「完成概率模型」的特征矩阵与标签。

    生成交互时特征依赖「学习者当时的掌握度」这一隐藏状态，因此这里按
    学习者内部时间顺序重放 BKT 风格的掌握度更新，**逐条复现**生成时的特征。
    预测 / 评测模块可以直接用它训练分类器，避免重复实现一遍特征逻辑。

    Args:
        dataset: 完整数据集。

    Returns:
        ``(features, labels)``：每行 8 个特征（顺序见 ``_FEATURE_NAMES``），
        标签为是否完成。
    """
    graph = KnowledgeGraph.from_dataset(dataset)
    course_prereq: dict[str, tuple[str, ...]] = {}
    for course_id, course in dataset.courses.items():
        closure = graph.prerequisite_closure(course.kp_ids)
        course_prereq[course_id] = tuple(sorted(closure - set(course.kp_ids)))

    by_learner: dict[str, list[Interaction]] = {}
    for interaction in dataset.interactions:
        by_learner.setdefault(interaction.learner_id, []).append(interaction)

    features: list[list[float]] = []
    labels: list[bool] = []
    for learner_id, items in by_learner.items():
        learner = dataset.learners[learner_id]
        mastery = dict(learner.initial_mastery)
        for interaction in sorted(items, key=lambda item: item.timestamp):
            course = dataset.courses[interaction.course_id]
            readiness = _readiness(mastery, course_prereq[interaction.course_id])
            features.append(list(_completion_features(learner, course, readiness)))
            labels.append(bool(interaction.completed))
            _update_mastery(mastery, course, bool(interaction.completed), learner.ability)
    return features, labels


def evaluate_completion_signal(
    features: Sequence[Sequence[float]],
    labels: Sequence[bool],
    seed: int,
) -> dict[str, Any]:
    """用逻辑回归验证「完成概率可被机器学习学到」，返回 AUC 等指标。

    采用固定的 70/30 训练-测试划分，全部使用 numpy 实现，因此结果完全可复现。
    """
    matrix = np.asarray(features, dtype=float)
    y = np.asarray([1.0 if flag else 0.0 for flag in labels], dtype=float)
    n_samples = int(matrix.shape[0])
    if n_samples < 40 or y.sum() in (0.0, float(n_samples)):
        return {"task": "course_completion", "status": "skipped", "n_samples": n_samples}

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    split = int(n_samples * 0.7)
    train_idx, test_idx = order[:split], order[split:]

    mean = matrix[train_idx].mean(axis=0)
    std = matrix[train_idx].std(axis=0)
    std[std < 1e-9] = 1.0
    standardized = (matrix - mean) / std
    design = np.hstack([np.ones((n_samples, 1)), standardized])

    weights = _fit_logistic(design[train_idx], y[train_idx])
    p_train = 1.0 / (1.0 + np.exp(-np.clip(design[train_idx] @ weights, -30.0, 30.0)))
    p_test = 1.0 / (1.0 + np.exp(-np.clip(design[test_idx] @ weights, -30.0, 30.0)))

    importance_raw = np.abs(weights[1:])
    importance_sum = float(importance_raw.sum()) or 1.0
    importance = {
        name: round(float(value) / importance_sum, 4)
        for name, value in zip(_FEATURE_NAMES, importance_raw)
    }

    return {
        "task": "course_completion",
        "model": "逻辑回归（numpy IRLS + L2）",
        "n_samples": n_samples,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "positive_rate": round(float(y.mean()), 4),
        "auc_test": round(_auc(y[test_idx], p_test), 4),
        "auc_train": round(_auc(y[train_idx], p_train), 4),
        "feature_importance": importance,
        "features": list(_FEATURE_NAMES),
        "note": "完成标签由逻辑斯蒂模型抽样生成，AUC 落在 0.78~0.88 说明信号可学但不过分干净",
    }


# ============================================================
# 8. 数据集元信息
# ============================================================


def _build_meta(
    scale: DatasetScale,
    dataset: Dataset,
    graph: KnowledgeGraph,
    model_report: dict[str, Any],
    kps: dict[str, KnowledgePoint],
    courses: dict[str, Course],
) -> dict[str, Any]:
    """汇总数据集元信息（规模统计、图谱统计、目标闭包、可学习性指标）。"""
    goal_closures = {
        goal_id: {
            "n_target_kps": len(goal.target_kp_ids),
            "closure_size": len(graph.prerequisite_closure(goal.target_kp_ids)),
            "closure_ratio": round(
                len(graph.prerequisite_closure(goal.target_kp_ids))
                / max(1, len(goal.target_kp_ids)),
                3,
            ),
        }
        for goal_id, goal in dataset.goals.items()
    }

    domain_stats = []
    for domain in DOMAINS:
        domain_stats.append(
            {
                "domain": domain,
                "n_kps": sum(1 for kp in kps.values() if kp.domain == domain),
                "n_courses": sum(1 for c in courses.values() if c.domain == domain),
                "n_learners": sum(1 for l in dataset.learners.values() if domain in l.interests),
            }
        )

    completion_rate = dataset.stats()["completion_rate"]
    return {
        "name": "learnpath-synthetic",
        "version": "1.0.0",
        "generator": "learnpath.data.generator.generate_dataset",
        "seed": scale.seed,
        "scale": asdict(scale),
        "stats": dataset.stats(),
        "graph": graph.stats().to_dict(),
        "domains": domain_stats,
        "goal_closures": goal_closures,
        "completion_model": model_report,
        "mechanism": {
            "kp_layering": "领域内 4 个起点 + layer 1..7 均匀铺开，父节点至少 1 个子节点",
            "cross_domain_edges": "按领域级拓扑序（数学 → 编程 → 数据/语言 → 前端/网络 → 产品）添加",
            "course_coverage": "每个知识点至少 1 门主课（权重 1.0），其余为 1~3 辅知识点综合课",
            "goal_targets": (
                "按领域权重分配名额（主领域 70~80%）+ 先修深度区间 + 关键词白/黑名单，"
                "再沿互不相同的先修分支从最深端轮转取样；目标闭包 ≥ 目标集合 1.8 倍"
            ),
            "initial_mastery": "仅覆盖目标闭包内 difficulty≤2 的知识点，覆盖率 15%~45%",
            "completion_model": "逻辑斯蒂：能力 + 等级差 + 时长 + 时间预算 + 先修就绪度 + 模态匹配 + 质量 + 理论偏好",
            "latent_sigma": _TARGET_LATENT_SIGMA,
            "logit_weights": dict(zip(_FEATURE_NAMES, _LOGIT_WEIGHTS)),
        },
        "measurements": {
            "completion_rate": completion_rate,
            "avg_initial_mastery_points": round(
                sum(len(l.initial_mastery) for l in dataset.learners.values())
                / max(1, len(dataset.learners)),
                2,
            ),
            "quiz_score_mean": round(
                sum(
                    i.quiz_score for i in dataset.interactions if i.quiz_score is not None
                )
                / max(1, sum(1 for i in dataset.interactions if i.quiz_score is not None)),
                4,
            ),
            "rating_coverage": round(
                sum(1 for i in dataset.interactions if i.rating is not None)
                / max(1, len(dataset.interactions)),
                4,
            ),
        },
        "notes": [
            "所有数据均为程序化合成，不含任何真实个人信息",
            "同一 seed 可完全复现；生成过程不使用网络与系统时间",
            "completion_model.auc_test 由 numpy 逻辑回归在固定 70/30 划分上测得",
        ],
    }


# ============================================================
# 9. 对外主入口
# ============================================================


def generate_dataset(scale: DatasetScale = DEFAULT_SCALE) -> Dataset:
    """按给定规模生成完整数据集（确定性、可复现）。

    Args:
        scale: 规模与随机种子。默认 :data:`learnpath.config.DEFAULT_SCALE`
            （420 知识点 / 560 课程 / 720 学习者 / 14000 交互）。

    Returns:
        已通过 :meth:`learnpath.data.schema.Dataset.validate` 的数据集，
        ``meta`` 中含规模统计、图谱统计、目标闭包与完成率模型的可学习性指标。

    Raises:
        GenerationError: 规模参数互相矛盾（例如课程数不足以覆盖知识点）。
    """
    kps, edges, layer_of, graph = _build_knowledge_points(scale)
    graph.validate()

    courses = _build_courses(kps, layer_of, scale, random.Random(scale.seed + 2))
    goals, closures = _build_goals(kps, layer_of, graph, random.Random(scale.seed + 3))
    learners = _build_learners(kps, goals, closures, scale, random.Random(scale.seed + 4))

    interactions, features, labels = _build_interactions(
        scale, kps, courses, learners, goals, graph
    )
    model_report = evaluate_completion_signal(features, labels, scale.seed + 99)

    dataset = Dataset(
        knowledge_points=kps,
        kp_edges=edges,
        courses=courses,
        learners=learners,
        goals=goals,
        interactions=interactions,
    )
    dataset.validate()
    dataset.meta = _build_meta(scale, dataset, graph, model_report, kps, courses)
    return dataset


__all__ = [
    "generate_dataset",
    "evaluate_completion_signal",
    "build_completion_features",
    "GenerationError",
]
