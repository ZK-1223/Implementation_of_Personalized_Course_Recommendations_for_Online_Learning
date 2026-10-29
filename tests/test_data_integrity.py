"""数据集完整性测试。

覆盖范围（对应「数据工程」交付标准）：

1. **确定性**：同一 seed 两次生成的数据集逐字节一致（含 meta）；
2. **契约**：``Dataset.validate()`` 通过、图谱无环、无孤立节点；
3. **图谱形态**：每领域知识点数、起点数、叶子数、最大深度、跨领域边数量；
4. **课程**：每个知识点至少被 1 门课程覆盖；等级 / 时长 / 理论占比 / 评分 /
   报名量 / 价格 / 标签等字段落在设计区间内；
5. **目标语义**：目标知识点的主 / 次领域构成符合岗位画像，且不含明显不相关的
   知识点（例如数据分析师不应要求「数据库分库分表」，产品经理不应要求
   「词向量与文本分类」）；目标闭包 ≥ 目标集合的 1.5 倍且跨 ≥ 3 个先修深度；
6. **学习者**：能力、时间预算、初始掌握度的取值范围与「只覆盖目标闭包内低阶
   知识点」的规则；
7. **交互日志**：无重复「学习者—课程」对、时间戳格式合法且按学习者内部时序递增、
   完成 / 进度 / 测验 / 评分之间的一致性；
8. **可学习性**：用 sklearn 与 numpy 手写逻辑回归分别验证完成标签的 AUC 落在
   0.75~0.92，且 ``dataset_meta.json`` 记录的 AUC 落在 0.78~0.88；
9. **落盘一致性**：``loader`` 读写往返、``data/`` 中的文件与 ``meta`` 自洽。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from learnpath.config import (
    DEFAULT_SCALE,
    DOMAINS,
    MAX_LEVEL,
    MIN_LEVEL,
    MODALITIES,
    TINY_SCALE,
)
from learnpath.data.generator import build_completion_features, generate_dataset
from learnpath.data.loader import dataset_exists, load_dataset, save_dataset
from learnpath.data.schema import Dataset
from learnpath.knowledge.graph import KnowledgeGraph

# 交互日志时间窗（与生成器一致）
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S"
_WINDOW_START = datetime(2026, 2, 1, 0, 0, 0)
_WINDOW_END = datetime(2026, 10, 1, 0, 0, 0)


# ============================================================
# fixture
# ============================================================


@pytest.fixture(scope="module")
def full() -> Dataset:
    """默认规模数据集（420 / 560 / 720 / 14000）。"""
    return generate_dataset(DEFAULT_SCALE)


@pytest.fixture(scope="module")
def full_graph(full: Dataset) -> KnowledgeGraph:
    graph = KnowledgeGraph.from_dataset(full)
    graph.validate()
    return graph


@pytest.fixture(scope="module")
def generated_tiny() -> Dataset:
    """生成器产出的极小数据集（35 / 28 / 12 / 180）。"""
    return generate_dataset(TINY_SCALE)


# ============================================================
# 1. 确定性
# ============================================================


def test_generation_is_deterministic() -> None:
    """同一 seed 两次生成的数据集必须完全一致（逐字段、含 meta）。"""
    first = generate_dataset(DEFAULT_SCALE)
    second = generate_dataset(DEFAULT_SCALE)

    first_json = json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True)
    second_json = json.dumps(second.to_dict(), ensure_ascii=False, sort_keys=True)
    assert first_json == second_json, "两次生成结果不一致：生成过程存在非确定性来源"

    # 换个 seed 必须产生不同数据（否则说明 seed 未生效）
    other = generate_dataset(
        type(DEFAULT_SCALE)(
            n_knowledge_points=DEFAULT_SCALE.n_knowledge_points,
            n_courses=DEFAULT_SCALE.n_courses,
            n_learners=DEFAULT_SCALE.n_learners,
            n_interactions=DEFAULT_SCALE.n_interactions,
            seed=DEFAULT_SCALE.seed + 1,
        )
    )
    assert other.to_dict()["interactions"] != first.to_dict()["interactions"]


# ============================================================
# 2. 数据契约与图谱合法性
# ============================================================


def test_dataset_contract_and_acyclic_graph(full: Dataset, full_graph: KnowledgeGraph) -> None:
    """Dataset 校验通过、图谱无环、没有孤立知识点。"""
    full.validate()
    full_graph.validate()

    isolated = [
        kp_id
        for kp_id in full_graph.node_ids
        if not full_graph.predecessors(kp_id) and not full_graph.successors(kp_id)
    ]
    assert isolated == [], f"存在孤立知识点：{isolated[:5]}"

    # 规模必须与配置一致
    assert len(full.knowledge_points) == DEFAULT_SCALE.n_knowledge_points
    assert len(full.courses) == DEFAULT_SCALE.n_courses
    assert len(full.learners) == DEFAULT_SCALE.n_learners
    assert len(full.interactions) == DEFAULT_SCALE.n_interactions
    assert len(full.goals) >= 8


# ============================================================
# 3. 知识图谱形态
# ============================================================


def test_graph_shape(full: Dataset, full_graph: KnowledgeGraph) -> None:
    """领域规模、起点 / 叶子 / 深度、跨领域先修边的数量都在设计区间内。"""
    stats = full_graph.stats()

    # 领域内知识点数 50~70
    for domain in DOMAINS:
        count = sum(1 for kp in full.knowledge_points.values() if kp.domain == domain)
        assert 50 <= count <= 70, f"{domain} 知识点数量 {count} 超出 50~70"

        # 每个领域 4~8 个起点（领域内无先修的节点）
        domain_roots = [
            kp_id
            for kp_id in full_graph.roots()
            if full_graph.domain_of(kp_id) == domain
        ]
        assert 4 <= len(domain_roots) <= 8, f"{domain} 起点数量 {len(domain_roots)} 超出 4~8"

        # 难度 1~5 每个档位都应有知识点（难度阶梯分布）
        levels = {
            kp.difficulty for kp in full.knowledge_points.values() if kp.domain == domain
        }
        assert levels == set(range(MIN_LEVEL, MAX_LEVEL + 1)), f"{domain} 难度分布不全：{levels}"

    assert 15 <= stats.n_roots <= 30, f"全图起点 {stats.n_roots} 超出 15~30"
    assert 40 <= stats.n_leaves <= 110, f"全图叶子 {stats.n_leaves} 超出 40~110"
    assert 7 <= stats.max_depth <= 11, f"最大先修深度 {stats.max_depth} 超出 7~11"

    cross_edges = [
        edge
        for edge in full.kp_edges
        if full.knowledge_points[edge.source].domain
        != full.knowledge_points[edge.target].domain
    ]
    assert 15 <= len(cross_edges) <= 30, f"跨领域先修边 {len(cross_edges)} 条，超出 15~30"


# ============================================================
# 4. 课程库
# ============================================================


def test_every_knowledge_point_is_covered(full: Dataset) -> None:
    """每个知识点至少被 1 门课程覆盖（保证任何目标都能规划出路径）。"""
    covered = {kp_id for course in full.courses.values() for kp_id in course.kp_ids}
    missing = set(full.knowledge_points) - covered
    assert missing == set(), f"未被任何课程覆盖的知识点：{sorted(missing)[:5]}"


def test_course_attribute_ranges(full: Dataset) -> None:
    """课程各字段落在设计要求区间内。"""
    allowed_prices = {0.0, 49.0, 99.0, 199.0, 299.0}
    modalities_seen: set[str] = set()

    for course in full.courses.values():
        assert MIN_LEVEL <= course.level <= MAX_LEVEL
        assert 2.0 <= course.duration_hours <= 40.0, f"{course.course_id} 时长越界"
        assert 0.1 <= course.theory_ratio <= 0.95, f"{course.course_id} 理论占比越界"
        assert 3.4 <= course.rating <= 4.95, f"{course.course_id} 评分越界"
        assert 200 <= course.enroll_count <= 300000, f"{course.course_id} 报名量越界"
        assert course.price in allowed_prices, f"{course.course_id} 价格不在档位上"
        assert 2 <= len(course.tags) <= 4, f"{course.course_id} 标签数越界"
        assert 1 <= len(course.kp_coverage) <= 4, f"{course.course_id} 覆盖知识点数越界"
        for coverage in course.kp_coverage:
            assert 0.4 <= coverage.weight <= 1.0
        # 课程等级应等于或略高于所覆盖知识点的最大难度
        max_difficulty = max(
            full.knowledge_points[kp_id].difficulty for kp_id in course.kp_ids
        )
        assert course.level >= max_difficulty
        assert course.level - max_difficulty <= 1
        modalities_seen.add(course.modality)

    assert modalities_seen == set(MODALITIES), "四种课程模态都应出现"

    titles = [course.title for course in full.courses.values()]
    assert len(set(titles)) == len(titles), "课程标题必须唯一"
    assert all("知识点" not in title for title in titles), "标题不应出现占位式命名"


# ============================================================
# 5. 职业目标
# ============================================================


def test_goal_closure_and_depth_gradient(full: Dataset, full_graph: KnowledgeGraph) -> None:
    """目标闭包 ≥ 目标集合 1.5 倍，且目标跨越 ≥ 3 个先修深度层。"""
    assert len(full.goals) >= 8

    for goal in full.goals.values():
        targets = list(goal.target_kp_ids)
        if len(full.knowledge_points) >= 100:
            assert 8 <= len(targets) <= 15, f"{goal.goal_id} 目标知识点数量 {len(targets)} 越界"

        closure = full_graph.prerequisite_closure(targets)
        assert set(targets) <= closure
        assert len(closure) >= 1.5 * len(targets), (
            f"{goal.goal_id} 闭包 {len(closure)} 不足目标集合 {len(targets)} 的 1.5 倍，"
            "路径规划将失去意义"
        )

        depths = {full_graph.depth(kp_id) for kp_id in targets}
        assert len(depths) >= 3, f"{goal.goal_id} 目标只覆盖 {sorted(depths)} 个深度层"


#: 每个职业目标允许的领域构成（领域 -> 占比区间）与禁止出现的关键词。
#: 这张表就是「目标设置是否合理」的回归护栏，取值来自岗位能力画像。
_GOAL_DOMAIN_RULES: dict[str, dict[str, object]] = {
    "goal_da": {
        "allowed": {
            "数据与人工智能": (0.60, 0.85),
            "数学基础": (0.10, 0.30),
            "产品与职场": (0.00, 0.15),
        },
        "forbidden": (
            "架构评审",
            "分库分表",
            "持续集成",
            "设计模式",
            "领域驱动",
            "多模态",
            "强化学习",
            "词向量",
            "大模型",
        ),
    },
    "goal_fe": {
        "allowed": {"前端与设计": (0.65, 0.85), "编程开发": (0.15, 0.35)},
        "forbidden": ("分库分表", "架构评审", "分布式事务", "技术规划"),
    },
    "goal_be": {
        "allowed": {"编程开发": (0.60, 0.80), "网络与安全": (0.20, 0.40)},
        "forbidden": ("架构评审", "技术规划"),
    },
    "goal_ml": {
        "allowed": {"数据与人工智能": (0.55, 0.75), "数学基础": (0.25, 0.45)},
        "forbidden": ("测度论", "泛函分析", "拓扑与流形", "算子理论", "随机微分方程"),
    },
    "goal_pm": {
        "allowed": {"产品与职场": (0.65, 0.85), "数据与人工智能": (0.15, 0.35)},
        "forbidden": ("词向量", "多模态", "聚类", "神经网络", "Transformer", "RAG", "强化学习"),
    },
    "goal_sec": {
        "allowed": {"网络与安全": (0.65, 0.85), "编程开发": (0.15, 0.35)},
        "forbidden": ("分库分表", "架构评审", "技术规划"),
    },
    "goal_biz": {
        "allowed": {"语言学习": (1.0, 1.0)},
        "forbidden": ("学术论文", "文献综述", "同声传译基础", "语料库", "雅思写作", "托福听力"),
    },
    "goal_math": {
        "allowed": {"数学基础": (1.0, 1.0)},
        "forbidden": ("测度论", "泛函分析", "拓扑与流形", "算子理论", "随机微分方程", "随机矩阵"),
    },
}


def test_goal_domain_composition(full: Dataset) -> None:
    """每个目标的目标知识点领域构成符合岗位画像，且不含明显不相关的知识点。"""
    for goal_id, rule in _GOAL_DOMAIN_RULES.items():
        assert goal_id in full.goals, f"缺少职业目标 {goal_id}"
        goal = full.goals[goal_id]
        targets = list(goal.target_kp_ids)
        domains = [full.knowledge_points[kp_id].domain for kp_id in targets]

        allowed = rule["allowed"]
        assert isinstance(allowed, dict)

        unexpected = sorted(set(domains) - set(allowed))
        assert not unexpected, f"{goal_id} 出现未授权领域：{unexpected}"

        for domain, (low, high) in allowed.items():
            share = domains.count(domain) / len(domains)
            assert low <= share <= high, (
                f"{goal_id} 的「{domain}」占比 {share:.2f} 超出 [{low}, {high}]"
            )

        forbidden = rule["forbidden"]
        assert isinstance(forbidden, tuple)
        hits = [
            f"{full.knowledge_points[kp_id].name}({kp_id})"
            for kp_id in targets
            if any(word in full.knowledge_points[kp_id].name for word in forbidden)
        ]
        assert not hits, f"{goal_id} 目标含不相关知识点：{hits}"


# ============================================================
# 6. 学习者
# ============================================================


def test_learner_profiles(full: Dataset, full_graph: KnowledgeGraph) -> None:
    """学习者画像取值合理，初始掌握度只覆盖目标闭包内的低阶知识点。"""
    allowed_budgets = {3.0, 5.0, 8.0, 12.0, 15.0, 20.0}
    abilities = []
    mastery_values: list[float] = []
    coverage_ratios: list[float] = []

    for learner in full.learners.values():
        goal = full.goals[learner.goal_id]
        assert 0.25 <= learner.ability <= 0.95
        abilities.append(learner.ability)
        assert learner.time_budget_hours_per_week in allowed_budgets
        # 截止周数 = 目标默认周期 ± 4
        assert abs(learner.deadline_weeks - goal.default_weeks) <= 4
        assert learner.motivation in {"求职", "晋升", "兴趣", "学业", "转行"}

        closure = full_graph.prerequisite_closure(goal.target_kp_ids)
        low_level = {
            kp_id for kp_id in closure if full.knowledge_points[kp_id].difficulty <= 2
        }
        assert set(learner.initial_mastery) <= low_level, (
            f"{learner.learner_id} 的初始掌握度越出「目标闭包内 difficulty<=2」的范围"
        )
        for value in learner.initial_mastery.values():
            assert 0.20 <= value <= 0.95
            mastery_values.append(value)

        if low_level:
            ratio = len(learner.initial_mastery) / len(low_level)
            assert 0.10 <= ratio <= 0.50, (
                f"{learner.learner_id} 初始掌握覆盖率 {ratio:.2f} 偏离 15%~45% 设计值"
            )
            coverage_ratios.append(ratio)

    # 学习者之间必须存在差异（否则推荐 / 规划无法体现个性化）
    assert len(set(abilities)) > len(abilities) * 0.9
    assert max(abilities) - min(abilities) > 0.5
    assert len(set(mastery_values)) > 50
    assert max(coverage_ratios) - min(coverage_ratios) > 0.1


# ============================================================
# 7. 交互日志
# ============================================================


def test_interaction_pairs_and_timestamps(full: Dataset) -> None:
    """无重复「学习者—课程」对；时间戳合法、落在时间窗内且按学习者时序递增。"""
    seen: set[tuple[str, str]] = set()
    last_by_learner: dict[str, datetime] = {}

    for interaction in full.interactions:
        key = (interaction.learner_id, interaction.course_id)
        assert key not in seen, f"重复的交互对：{key}"
        seen.add(key)

        moment = datetime.strptime(interaction.timestamp, _TS_FORMAT)
        assert _WINDOW_START <= moment <= _WINDOW_END, f"时间戳越界：{interaction.timestamp}"
        # 时间戳必须是「2026-02 ~ 2026-09」
        assert moment.year == 2026 and 2 <= moment.month <= 9

        previous = last_by_learner.get(interaction.learner_id)
        if previous is not None:
            assert moment >= previous, (
                f"{interaction.learner_id} 的交互时间戳未单调递增："
                f"{previous} -> {moment}"
            )
        last_by_learner[interaction.learner_id] = moment


def test_interaction_field_consistency(full: Dataset) -> None:
    """完成 / 进度 / 测验 / 评分 / 投入时长之间的一致性约束。"""
    completed = 0
    rated = 0

    for interaction in full.interactions:
        assert 0.0 <= interaction.progress <= 1.0
        if interaction.completed:
            completed += 1
            assert interaction.progress == 1.0, "完成的交互进度必须为 1.0"
            if interaction.quiz_score is not None:
                assert 0.0 <= interaction.quiz_score <= 1.0
        else:
            assert 0.05 <= interaction.progress <= 0.90
            assert interaction.quiz_score is None, "未完成的交互不应有结课测验成绩"

        if interaction.rating is not None:
            rated += 1
            assert 1.0 <= interaction.rating <= 5.0

        course = full.courses[interaction.course_id]
        assert interaction.hours_spent >= 0
        assert interaction.hours_spent <= course.duration_hours * 1.5

    rate = completed / len(full.interactions)
    assert 0.35 <= rate <= 0.65, f"整体完成率 {rate:.3f} 偏离 50% 设计值"
    # 评分必须有缺失（不是每条都打分）
    assert 0.1 <= rated / len(full.interactions) <= 0.7


# ============================================================
# 8. 完成概率的可学习性
# ============================================================


def test_completion_probability_is_learnable(full: Dataset) -> None:
    """完成标签应能被逻辑回归学到：AUC 落在 0.75~0.92（有信号但不过分干净）。"""
    features, labels = build_completion_features(full)
    matrix = np.asarray(features, dtype=float)
    y = np.asarray([1.0 if flag else 0.0 for flag in labels], dtype=float)

    # 固定划分，保证断言可复现
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    split = int(len(y) * 0.7)
    train_idx, test_idx = order[:split], order[split:]

    mean = matrix[train_idx].mean(axis=0)
    std = matrix[train_idx].std(axis=0)
    std[std < 1e-9] = 1.0
    x_all = (matrix - mean) / std
    design = np.hstack([np.ones((len(y), 1)), x_all])

    # ---- (a) numpy 手写 IRLS 逻辑回归 ----
    weights = np.zeros(design.shape[1])
    reg = np.full(design.shape[1], 1.0)
    reg[0] = 0.0
    for _ in range(40):
        p = 1.0 / (1.0 + np.exp(-np.clip(design[train_idx] @ weights, -30, 30)))
        gradient = design[train_idx].T @ (p - y[train_idx]) / len(train_idx) + reg * weights / len(train_idx)
        curvature = p * (1.0 - p) + 1e-9
        hessian = (design[train_idx] * curvature[:, None]).T @ design[train_idx] / len(train_idx)
        hessian += np.diag(reg) / len(train_idx)
        weights -= np.linalg.solve(hessian, gradient)

    def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        positive = scores[y_true == 1]
        negative = scores[y_true == 0]
        # 用广播算 AUC 太慢，这里用秩统计（Mann-Whitney U）
        all_scores = np.concatenate([positive, negative])
        order_inner = np.argsort(all_scores, kind="mergesort")
        ranks = np.empty(len(all_scores), dtype=float)
        ranks[order_inner] = np.arange(1, len(all_scores) + 1, dtype=float)
        # 处理并列分数：把同一分数的秩取平均
        sorted_scores = all_scores[order_inner]
        index = 0
        while index < len(sorted_scores):
            end = index
            while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
                end += 1
            ranks[order_inner[index : end + 1]] = (index + end) / 2.0 + 1.0
            index = end + 1
        rank_sum = ranks[: len(positive)].sum()
        return float(
            (rank_sum - len(positive) * (len(positive) + 1) / 2.0)
            / (len(positive) * len(negative))
        )

    p_test = 1.0 / (1.0 + np.exp(-np.clip(design[test_idx] @ weights, -30, 30)))
    auc_numpy = _auc(y[test_idx], p_test)
    assert 0.75 <= auc_numpy <= 0.92, f"numpy 逻辑回归 AUC={auc_numpy:.4f} 不在 0.75~0.92"

    # ---- (b) sklearn 交叉验证 ----
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    model = LogisticRegression(max_iter=1000)
    model.fit(x_all[train_idx], y[train_idx])
    auc_sklearn = float(roc_auc_score(y[test_idx], model.predict_proba(x_all[test_idx])[:, 1]))
    assert 0.75 <= auc_sklearn <= 0.92, f"sklearn 逻辑回归 AUC={auc_sklearn:.4f} 不在 0.75~0.92"

    # 手工实现与 sklearn 应当高度一致（说明特征与标签确实对齐）
    assert abs(auc_numpy - auc_sklearn) < 0.03, "numpy 与 sklearn 的 AUC 差异过大"

    # ---- (c) meta 中记录的实测值必须在要求区间 ----
    reported = full.meta["completion_model"]["auc_test"]
    assert 0.78 <= reported <= 0.88, f"meta 中记录的 AUC={reported} 不在 0.78~0.88"


# ============================================================
# 9. 两种规模 / 落盘一致性
# ============================================================


def test_tiny_and_full_scales(generated_tiny: Dataset, full: Dataset) -> None:
    """tiny 与 full 两种规模都能生成并通过校验。"""
    generated_tiny.validate()
    full.validate()

    assert len(generated_tiny.knowledge_points) == TINY_SCALE.n_knowledge_points
    assert len(generated_tiny.courses) == TINY_SCALE.n_courses
    assert len(generated_tiny.learners) == TINY_SCALE.n_learners
    assert len(generated_tiny.interactions) == TINY_SCALE.n_interactions

    covered = {kp_id for course in generated_tiny.courses.values() for kp_id in course.kp_ids}
    assert covered == set(generated_tiny.knowledge_points)
    KnowledgeGraph.from_dataset(generated_tiny).validate()


def test_loader_roundtrip_and_meta_consistency(full: Dataset, tmp_path: Path) -> None:
    """save_dataset / load_dataset 往返一致，且 meta 统计与磁盘数据自洽。"""
    assert dataset_exists(tmp_path) is False
    written = save_dataset(full, tmp_path)
    assert len(written) == 7
    assert dataset_exists(tmp_path) is True

    reloaded = load_dataset(tmp_path)
    assert reloaded.stats() == full.stats()
    assert reloaded.meta["seed"] == DEFAULT_SCALE.seed
    assert len(reloaded.interactions) == len(full.interactions)

    # meta 中的图谱统计与目标闭包必须能由落盘数据独立重算出来
    graph = KnowledgeGraph.from_dataset(reloaded)
    recomputed = graph.stats().to_dict()
    for key, value in reloaded.meta["graph"].items():
        assert recomputed[key] == value, f"meta.graph.{key} 与落盘数据不一致"

    for goal_id, info in reloaded.meta["goal_closures"].items():
        targets = reloaded.goals[goal_id].target_kp_ids
        closure = graph.prerequisite_closure(targets)
        assert info["n_target_kps"] == len(targets)
        assert info["closure_size"] == len(closure)
        assert info["closure_ratio"] == round(len(closure) / len(targets), 3)

    # 读取缺失目录时必须给出清晰错误
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "not-there")


def test_committed_dataset_is_consistent() -> None:
    """仓库内 ``data/`` 的数据集必须能加载、校验，并与 meta 自洽。"""
    from learnpath.paths import DATA_DIR  # noqa: PLC0415

    if not dataset_exists(DATA_DIR):
        pytest.skip("data/ 尚未生成，跳过落盘数据检查")

    on_disk = load_dataset(DATA_DIR)
    on_disk.validate()
    graph = KnowledgeGraph.from_dataset(on_disk)
    graph.validate()

    assert on_disk.stats()["n_knowledge_points"] == on_disk.meta["stats"]["n_knowledge_points"]
    assert 0.78 <= on_disk.meta["completion_model"]["auc_test"] <= 0.88

    covered = {kp_id for course in on_disk.courses.values() for kp_id in course.kp_ids}
    assert covered == set(on_disk.knowledge_points)

    for goal_id, info in on_disk.meta["goal_closures"].items():
        closure = graph.prerequisite_closure(on_disk.goals[goal_id].target_kp_ids)
        assert info["closure_size"] == len(closure)
