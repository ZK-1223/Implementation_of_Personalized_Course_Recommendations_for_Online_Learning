"""微型数据集（fixture）。

作用：为单元测试与算法模块的早期开发提供**确定性、可运行、规模极小**的数据集，
使得算法开发不必等待完整数据生成器就绪。完整数据集由
:mod:`learnpath.data.generator` 生成。

数据集结构：
    7 个领域 × 5 个知识点 = 35 个知识点，领域内先修关系构成有向链，
    并额外加入 2 条跨领域先修边，保证图谱非平凡；
    28 门课程覆盖这些知识点；
    3 个职业目标；
    12 名学习者与约 180 条交互记录。
"""

from __future__ import annotations

import random
from typing import Iterable

from learnpath.config import DOMAINS, MIN_LEVEL, MOTIVATIONS, TINY_SCALE
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

# 每个领域的知识点名称（按难度递增排列，领域内先修链）
_DOMAIN_KP_NAMES: dict[str, tuple[str, ...]] = {
    "编程开发": ("变量与数据类型", "流程控制", "函数与模块", "面向对象", "并发编程"),
    "数据与人工智能": ("数据清洗", "描述性统计", "假设检验", "回归建模", "模型评估"),
    "前端与设计": ("HTML 语义化", "CSS 布局", "JavaScript 基础", "组件化开发", "性能优化"),
    "网络与安全": ("网络分层模型", "TCP 与 UDP", "Web 攻防基础", "加密与证书", "渗透测试"),
    "产品与职场": ("需求调研", "用户故事", "原型设计", "数据分析看板", "增长实验"),
    "语言学习": ("音标与连读", "核心词汇", "基础句型", "学术写作", "商务表达"),
    "数学基础": ("函数与极限", "线性代数", "概率论", "最优化基础", "数值计算"),
}

# 跨领域先修边：(先修知识点名, 后继知识点名)
_CROSS_EDGES: tuple[tuple[str, str], ...] = (
    ("函数与模块", "数据清洗"),
    ("概率论", "假设检验"),
    ("描述性统计", "数据分析看板"),
)

_GOALS: tuple[tuple[str, str, str, tuple[str, ...], int], ...] = (
    (
        "goal_da",
        "数据分析师",
        "面向零基础转行，覆盖数据处理、统计推断与可视化",
        ("数据清洗", "描述性统计", "假设检验", "回归建模", "数据分析看板"),
        12,
    ),
    (
        "goal_fe",
        "前端工程师",
        "从前端三件套到组件化与性能优化",
        ("HTML 语义化", "CSS 布局", "JavaScript 基础", "组件化开发", "性能优化"),
        16,
    ),
    (
        "goal_sec",
        "网络安全工程师",
        "网络协议基础 + Web 攻防 + 加密体系",
        ("网络分层模型", "TCP 与 UDP", "Web 攻防基础", "加密与证书"),
        14,
    ),
)

_ROLES = (
    ("学生", "本科", "信息管理", 0),
    ("运营专员", "本科", "市场营销", 2),
    ("后端工程师", "本科", "软件工程", 3),
    ("测试工程师", "硕士", "计算机", 1),
    ("行政助理", "专科", "文秘", 4),
    ("产品助理", "本科", "工商管理", 1),
)


def _build_knowledge_points() -> tuple[dict[str, KnowledgePoint], list[KPEdge]]:
    kps: dict[str, KnowledgePoint] = {}
    edges: list[KPEdge] = []
    name_to_id: dict[str, str] = {}

    for domain in DOMAINS:
        names = _DOMAIN_KP_NAMES[domain]
        previous_id: str | None = None
        for index, name in enumerate(names, start=1):
            kp_id = f"KP-{DOMAINS.index(domain) + 1:02d}-{index:02d}"
            kps[kp_id] = KnowledgePoint(
                kp_id=kp_id,
                name=name,
                domain=domain,
                difficulty=MIN_LEVEL + index - 1,
                description=f"{domain}·{name}",
            )
            name_to_id[name] = kp_id
            if previous_id is not None:
                edges.append(KPEdge(source=previous_id, target=kp_id))
            previous_id = kp_id

    for source_name, target_name in _CROSS_EDGES:
        edges.append(KPEdge(source=name_to_id[source_name], target=name_to_id[target_name]))

    return kps, edges


def _build_courses(
    kps: dict[str, KnowledgePoint],
    rng: random.Random,
    n_courses: int,
) -> dict[str, Course]:
    """按「单知识点课 / 双知识点综合课」两种模式生成课程。"""
    by_domain: dict[str, list[KnowledgePoint]] = {d: [] for d in DOMAINS}
    for kp in kps.values():
        by_domain[kp.domain].append(kp)
    for items in by_domain.values():
        items.sort(key=lambda kp: kp.difficulty)

    courses: dict[str, Course] = {}
    modalities = ("video", "text", "project", "mixed")
    instructors = ("王敏", "李泽宇", "陈思", "赵晓峰", "刘倩", "孙浩然")

    idx = 0
    while len(courses) < n_courses:
        domain = DOMAINS[idx % len(DOMAINS)]
        pool = by_domain[domain]
        idx += 1
        if idx % 3 == 0 and len(pool) >= 2:
            # 综合课：相邻两个知识点
            start = (idx // 3) % (len(pool) - 1)
            covered = pool[start : start + 2]
            coverage = tuple(
                KPCoverage(kp_id=kp.kp_id, weight=0.8 if i == 0 else 0.6)
                for i, kp in enumerate(covered)
            )
            level = max(kp.difficulty for kp in covered)
            title = f"{covered[0].name}与{covered[1].name}综合实战"
            duration = 10.0 + 2.0 * level
        else:
            kp = pool[(idx // len(DOMAINS)) % len(pool)]
            coverage = (KPCoverage(kp_id=kp.kp_id, weight=1.0),)
            level = kp.difficulty
            title = f"{kp.name}·系统精讲"
            duration = 3.0 + 1.5 * level

        course_id = f"C-{len(courses) + 1:04d}"
        courses[course_id] = Course(
            course_id=course_id,
            title=title,
            domain=domain,
            level=level,
            duration_hours=round(duration, 1),
            kp_coverage=coverage,
            theory_ratio=round(rng.uniform(0.3, 0.9), 2),
            modality=rng.choice(modalities),
            instructor=rng.choice(instructors),
            rating=round(rng.uniform(3.6, 4.9), 1),
            enroll_count=rng.randint(200, 40000),
            price=0.0 if rng.random() < 0.6 else rng.choice((49.0, 99.0, 199.0)),
            tags=(DOMAINS[DOMAINS.index(domain)],),
        )

    return courses


def _build_goals(kps: dict[str, KnowledgePoint]) -> dict[str, Goal]:
    name_to_id = {kp.name: kp.kp_id for kp in kps.values()}
    goals: dict[str, Goal] = {}
    for goal_id, name, desc, kp_names, weeks in _GOALS:
        goals[goal_id] = Goal(
            goal_id=goal_id,
            name=name,
            target_role=name,
            description=desc,
            target_kp_ids=tuple(name_to_id[n] for n in kp_names),
            default_weeks=weeks,
            required_mastery=0.80,
        )
    return goals


def _build_learners(
    kps: dict[str, KnowledgePoint],
    goals: dict[str, Goal],
    rng: random.Random,
    n_learners: int,
) -> dict[str, Learner]:
    by_domain: dict[str, list[KnowledgePoint]] = {d: [] for d in DOMAINS}
    for kp in kps.values():
        by_domain[kp.domain].append(kp)
    for items in by_domain.values():
        items.sort(key=lambda kp: kp.difficulty)

    goal_ids = list(goals)
    learners: dict[str, Learner] = {}
    for index in range(n_learners):
        goal_id = goal_ids[index % len(goal_ids)]
        goal = goals[goal_id]
        role, education, major, years = _ROLES[index % len(_ROLES)]
        ability = round(rng.uniform(0.28, 0.92), 3)

        # 起始掌握度：与目标知识点同领域、难度不超过能力上界的低阶知识点有一定基础
        initial: dict[str, float] = {}
        target_domains = {kps[k].domain for k in goal.target_kp_ids}
        for domain in target_domains:
            for kp in by_domain[domain]:
                if kp.difficulty <= MIN_LEVEL + 1 and rng.random() < 0.55:
                    initial[kp.kp_id] = round(rng.uniform(0.30, 0.92), 3)

        learners[f"L-{index + 1:04d}"] = Learner(
            learner_id=f"L-{index + 1:04d}",
            nickname=f"学员{index + 1:02d}",
            background=Background(
                education=education,
                major=major,
                work_years=years,
                current_role=role,
            ),
            goal_id=goal_id,
            motivation=MOTIVATIONS[index % len(MOTIVATIONS)],
            deadline_weeks=goal.default_weeks + rng.choice((-2, 0, 2)),
            time_budget_hours_per_week=round(rng.choice((3.0, 5.0, 8.0, 12.0, 15.0)), 1),
            learning_style=LearningStyle(
                theory_preference=round(rng.uniform(0.15, 0.85), 2),
                prefer_modality=rng.choice(("video", "text", "project", "mixed")),
                session_len_min=rng.choice((25, 30, 45, 60, 90)),
                weekly_days=rng.randint(2, 7),
            ),
            ability=ability,
            initial_mastery=initial,
            interests=tuple(target_domains),
            joined_at=f"2026-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}T09:00:00",
        )
    return learners


def _build_interactions(
    courses: dict[str, Course],
    learners: dict[str, Learner],
    rng: random.Random,
    n_interactions: int,
) -> list[Interaction]:
    course_ids = list(courses)
    completed_ids = set()
    interactions: list[Interaction] = []

    learner_ids = list(learners)
    attempt = 0
    while len(interactions) < n_interactions and attempt < n_interactions * 20:
        attempt += 1
        learner = learners[rng.choice(learner_ids)]
        course_id = rng.choice(course_ids)
        if (learner.learner_id, course_id) in completed_ids:
            continue
        completed_ids.add((learner.learner_id, course_id))

        course = courses[course_id]
        # 完成倾向：能力越高、课程越短越容易完成
        p_complete = min(0.95, 0.35 + 0.5 * learner.ability - 0.05 * course.level)
        done = rng.random() < p_complete
        progress = 1.0 if done else round(rng.uniform(0.05, 0.9), 3)
        quiz = round(min(1.0, max(0.0, rng.gauss(0.5 + 0.4 * learner.ability, 0.12))), 3) if done else None

        interactions.append(
            Interaction(
                learner_id=learner.learner_id,
                course_id=course_id,
                progress=progress,
                completed=done,
                rating=(round(rng.uniform(3.0, 5.0), 1) if done and rng.random() < 0.7 else None),
                hours_spent=round(course.duration_hours * progress, 2),
                quiz_score=quiz,
                timestamp=f"2026-{rng.randint(2, 9):02d}-{rng.randint(1, 28):02d}T1{rng.randint(0, 9)}:00:00",
            )
        )

    return interactions


def build_tiny_dataset(seed: int = TINY_SCALE.seed) -> Dataset:
    """构造微型数据集（确定性，可复现）。"""
    rng = random.Random(seed)
    kps, edges = _build_knowledge_points()
    courses = _build_courses(kps, rng, TINY_SCALE.n_courses)
    goals = _build_goals(kps)
    learners = _build_learners(kps, goals, rng, TINY_SCALE.n_learners)
    interactions = _build_interactions(courses, learners, rng, TINY_SCALE.n_interactions)

    dataset = Dataset(
        knowledge_points=kps,
        kp_edges=edges,
        courses=courses,
        learners=learners,
        goals=goals,
        interactions=interactions,
        meta={"name": "tiny-fixture", "seed": seed},
    )
    dataset.validate()
    return dataset


__all__ = ["build_tiny_dataset"]
