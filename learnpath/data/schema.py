"""数据模型定义 —— 全项目共享的数据契约。

所有模块（数据生成、知识图谱、学习者建模、推荐、规划、预测、API、前端）
都只依赖本文件定义的数据结构，因此**本文件是并行开发的唯一基准**。

设计原则：
    * 纯 Python 标准库，无第三方依赖，可被任意模块安全导入；
    * 每个实体都提供 ``to_dict()`` / ``from_dict()``，可直接 JSON 序列化；
    * 构造时做强校验（``__post_init__``），错误尽早暴露；
    * 金额 / 时长等数值统一单位，避免歧义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from learnpath.config import (
    DOMAINS,
    MAX_LEVEL,
    MIN_LEVEL,
    MODALITIES,
    MOTIVATIONS,
)


class SchemaError(ValueError):
    """数据不符合契约时抛出的异常。"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _check_unit(value: float, name: str) -> None:
    _check(isinstance(value, (int, float)), f"{name} 必须是数值，实际为 {type(value)!r}")
    _check(0.0 <= float(value) <= 1.0, f"{name} 必须落在 [0, 1]，实际为 {value}")


# ============================================================
# 知识点 / 知识图谱
# ============================================================


@dataclass(slots=True)
class KnowledgePoint:
    """一个可被掌握、可被课程覆盖的最小知识单元。"""

    kp_id: str
    name: str
    domain: str
    difficulty: int
    description: str = ""

    def __post_init__(self) -> None:
        _check(bool(self.kp_id), "KnowledgePoint.kp_id 不能为空")
        _check(bool(self.name), f"{self.kp_id}: name 不能为空")
        _check(self.domain in DOMAINS, f"{self.kp_id}: 未知领域 {self.domain!r}")
        _check(
            MIN_LEVEL <= int(self.difficulty) <= MAX_LEVEL,
            f"{self.kp_id}: difficulty 必须在 {MIN_LEVEL}~{MAX_LEVEL}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kp_id": self.kp_id,
            "name": self.name,
            "domain": self.domain,
            "difficulty": int(self.difficulty),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgePoint":
        return cls(
            kp_id=str(data["kp_id"]),
            name=str(data["name"]),
            domain=str(data["domain"]),
            difficulty=int(data["difficulty"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class KPEdge:
    """知识点先修关系：``source`` 是 ``target`` 的先修知识。"""

    source: str
    target: str

    def __post_init__(self) -> None:
        _check(self.source != self.target, f"先修关系不允许自环：{self.source}")
        _check(bool(self.source) and bool(self.target), "先修关系两端都不能为空")

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KPEdge":
        return cls(source=str(data["source"]), target=str(data["target"]))


@dataclass(frozen=True, slots=True)
class KPCoverage:
    """课程对某个知识点的覆盖强度（weight ∈ (0, 1]）。"""

    kp_id: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        _check(bool(self.kp_id), "KPCoverage.kp_id 不能为空")
        _check(0.0 < float(self.weight) <= 1.0, f"{self.kp_id}: 覆盖权重必须在 (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"kp_id": self.kp_id, "weight": round(float(self.weight), 4)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KPCoverage":
        return cls(kp_id=str(data["kp_id"]), weight=float(data.get("weight", 1.0)))


# ============================================================
# 课程
# ============================================================


@dataclass(slots=True)
class Course:
    """一门在线课程。"""

    course_id: str
    title: str
    domain: str
    level: int
    duration_hours: float
    kp_coverage: tuple[KPCoverage, ...]
    theory_ratio: float
    modality: str
    instructor: str
    rating: float
    enroll_count: int
    price: float = 0.0
    language: str = "zh"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check(bool(self.course_id), "Course.course_id 不能为空")
        _check(bool(self.title), f"{self.course_id}: title 不能为空")
        _check(self.domain in DOMAINS, f"{self.course_id}: 未知领域 {self.domain!r}")
        _check(
            MIN_LEVEL <= int(self.level) <= MAX_LEVEL,
            f"{self.course_id}: level 必须在 {MIN_LEVEL}~{MAX_LEVEL}",
        )
        _check(float(self.duration_hours) > 0, f"{self.course_id}: duration_hours 必须为正")
        _check(len(self.kp_coverage) > 0, f"{self.course_id}: 至少覆盖一个知识点")
        _check_unit(float(self.theory_ratio), f"{self.course_id}.theory_ratio")
        _check(self.modality in MODALITIES, f"{self.course_id}: 未知模态 {self.modality!r}")
        _check(0.0 <= float(self.rating) <= 5.0, f"{self.course_id}: rating 必须在 0~5")
        _check(int(self.enroll_count) >= 0, f"{self.course_id}: enroll_count 不能为负")
        _check(float(self.price) >= 0, f"{self.course_id}: price 不能为负")
        _check(
            len({c.kp_id for c in self.kp_coverage}) == len(self.kp_coverage),
            f"{self.course_id}: kp_coverage 存在重复知识点",
        )

    # ---- 便捷属性 ----
    @property
    def kp_ids(self) -> tuple[str, ...]:
        return tuple(c.kp_id for c in self.kp_coverage)

    @property
    def kp_weights(self) -> dict[str, float]:
        return {c.kp_id: float(c.weight) for c in self.kp_coverage}

    @property
    def difficulty_hours(self) -> float:
        """难度加权时长：同样时长下，高难度课占用更多认知资源。"""
        return float(self.duration_hours) * (0.8 + 0.1 * int(self.level))

    def to_dict(self) -> dict[str, Any]:
        return {
            "course_id": self.course_id,
            "title": self.title,
            "domain": self.domain,
            "level": int(self.level),
            "duration_hours": round(float(self.duration_hours), 2),
            "kp_coverage": [c.to_dict() for c in self.kp_coverage],
            "theory_ratio": round(float(self.theory_ratio), 4),
            "modality": self.modality,
            "instructor": self.instructor,
            "rating": round(float(self.rating), 2),
            "enroll_count": int(self.enroll_count),
            "price": round(float(self.price), 2),
            "language": self.language,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Course":
        return cls(
            course_id=str(data["course_id"]),
            title=str(data["title"]),
            domain=str(data["domain"]),
            level=int(data["level"]),
            duration_hours=float(data["duration_hours"]),
            kp_coverage=tuple(KPCoverage.from_dict(c) for c in data["kp_coverage"]),
            theory_ratio=float(data["theory_ratio"]),
            modality=str(data["modality"]),
            instructor=str(data.get("instructor", "未知讲师")),
            rating=float(data.get("rating", 0.0)),
            enroll_count=int(data.get("enroll_count", 0)),
            price=float(data.get("price", 0.0)),
            language=str(data.get("language", "zh")),
            tags=tuple(str(t) for t in data.get("tags", ())),
        )


# ============================================================
# 职业目标
# ============================================================


@dataclass(slots=True)
class Goal:
    """一个职业 / 学习目标，对应一组需要达成的目标知识点。"""

    goal_id: str
    name: str
    target_role: str
    description: str
    target_kp_ids: tuple[str, ...]
    default_weeks: int = 12
    required_mastery: float = 0.80

    def __post_init__(self) -> None:
        _check(bool(self.goal_id), "Goal.goal_id 不能为空")
        _check(bool(self.name), f"{self.goal_id}: name 不能为空")
        _check(len(self.target_kp_ids) > 0, f"{self.goal_id}: 必须至少有一个目标知识点")
        _check(int(self.default_weeks) > 0, f"{self.goal_id}: default_weeks 必须为正")
        _check_unit(float(self.required_mastery), f"{self.goal_id}.required_mastery")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "name": self.name,
            "target_role": self.target_role,
            "description": self.description,
            "target_kp_ids": list(self.target_kp_ids),
            "default_weeks": int(self.default_weeks),
            "required_mastery": round(float(self.required_mastery), 3),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Goal":
        return cls(
            goal_id=str(data["goal_id"]),
            name=str(data["name"]),
            target_role=str(data.get("target_role", data["name"])),
            description=str(data.get("description", "")),
            target_kp_ids=tuple(str(k) for k in data["target_kp_ids"]),
            default_weeks=int(data.get("default_weeks", 12)),
            required_mastery=float(data.get("required_mastery", 0.80)),
        )


# ============================================================
# 学习者
# ============================================================


@dataclass(slots=True)
class Background:
    """学习者背景。"""

    education: str = "本科"
    major: str = "计算机"
    work_years: int = 0
    current_role: str = "学生"

    def to_dict(self) -> dict[str, Any]:
        return {
            "education": self.education,
            "major": self.major,
            "work_years": int(self.work_years),
            "current_role": self.current_role,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Background":
        data = data or {}
        return cls(
            education=str(data.get("education", "本科")),
            major=str(data.get("major", "计算机")),
            work_years=int(data.get("work_years", 0)),
            current_role=str(data.get("current_role", "学生")),
        )


@dataclass(slots=True)
class LearningStyle:
    """学习风格偏好。"""

    theory_preference: float = 0.5   # 0=偏好实践，1=偏好理论
    prefer_modality: str = "video"   # video / text / project / mixed
    session_len_min: int = 45        # 单次可专注时长（分钟）
    weekly_days: int = 4             # 每周可学习天数

    def __post_init__(self) -> None:
        _check_unit(float(self.theory_preference), "LearningStyle.theory_preference")
        _check(
            self.prefer_modality in MODALITIES,
            f"LearningStyle.prefer_modality 未知：{self.prefer_modality!r}",
        )
        _check(int(self.session_len_min) > 0, "LearningStyle.session_len_min 必须为正")
        _check(1 <= int(self.weekly_days) <= 7, "LearningStyle.weekly_days 必须在 1~7")

    def to_dict(self) -> dict[str, Any]:
        return {
            "theory_preference": round(float(self.theory_preference), 3),
            "prefer_modality": self.prefer_modality,
            "session_len_min": int(self.session_len_min),
            "weekly_days": int(self.weekly_days),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "LearningStyle":
        data = data or {}
        return cls(
            theory_preference=float(data.get("theory_preference", 0.5)),
            prefer_modality=str(data.get("prefer_modality", "video")),
            session_len_min=int(data.get("session_len_min", 45)),
            weekly_days=int(data.get("weekly_days", 4)),
        )


@dataclass(slots=True)
class Learner:
    """学习者画像。"""

    learner_id: str
    nickname: str
    background: Background
    goal_id: str
    motivation: str
    deadline_weeks: int
    time_budget_hours_per_week: float
    learning_style: LearningStyle
    ability: float
    initial_mastery: dict[str, float] = field(default_factory=dict)
    interests: tuple[str, ...] = ()
    joined_at: str = ""

    def __post_init__(self) -> None:
        _check(bool(self.learner_id), "Learner.learner_id 不能为空")
        _check(bool(self.goal_id), f"{self.learner_id}: goal_id 不能为空")
        _check(
            self.motivation in MOTIVATIONS,
            f"{self.learner_id}: 未知动机 {self.motivation!r}",
        )
        _check(int(self.deadline_weeks) > 0, f"{self.learner_id}: deadline_weeks 必须为正")
        _check(
            float(self.time_budget_hours_per_week) > 0,
            f"{self.learner_id}: 每周学习时长必须为正",
        )
        _check_unit(float(self.ability), f"{self.learner_id}.ability")
        for kp_id, value in self.initial_mastery.items():
            _check_unit(float(value), f"{self.learner_id}.initial_mastery[{kp_id}]")
        for domain in self.interests:
            _check(domain in DOMAINS, f"{self.learner_id}: 未知兴趣领域 {domain!r}")

    # ---- 便捷属性 ----
    @property
    def total_available_hours(self) -> float:
        """到截止日期为止的总可投入时长。"""
        return float(self.time_budget_hours_per_week) * int(self.deadline_weeks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "nickname": self.nickname,
            "background": self.background.to_dict(),
            "goal_id": self.goal_id,
            "motivation": self.motivation,
            "deadline_weeks": int(self.deadline_weeks),
            "time_budget_hours_per_week": round(float(self.time_budget_hours_per_week), 2),
            "learning_style": self.learning_style.to_dict(),
            "ability": round(float(self.ability), 4),
            "initial_mastery": {k: round(float(v), 4) for k, v in self.initial_mastery.items()},
            "interests": list(self.interests),
            "joined_at": self.joined_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Learner":
        return cls(
            learner_id=str(data["learner_id"]),
            nickname=str(data.get("nickname", data["learner_id"])),
            background=Background.from_dict(data.get("background")),
            goal_id=str(data["goal_id"]),
            motivation=str(data.get("motivation", "兴趣")),
            deadline_weeks=int(data.get("deadline_weeks", 12)),
            time_budget_hours_per_week=float(data.get("time_budget_hours_per_week", 6.0)),
            learning_style=LearningStyle.from_dict(data.get("learning_style")),
            ability=float(data.get("ability", 0.5)),
            initial_mastery={
                str(k): float(v) for k, v in (data.get("initial_mastery") or {}).items()
            },
            interests=tuple(str(d) for d in data.get("interests", ())),
            joined_at=str(data.get("joined_at", "")),
        )


# ============================================================
# 交互行为
# ============================================================


@dataclass(frozen=True, slots=True)
class Interaction:
    """一条「学习者—课程」学习记录。

    字段语义：
        progress: 学习进度 0~1。
        completed: 是否完成（progress >= 1 或通过考核）。
        rating: 学习者评分 1~5，未评为 None。
        hours_spent: 实际投入小时数。
        quiz_score: 结课测验得分 0~1，未测为 None。
        timestamp: ISO 8601 时间字符串。
    """

    learner_id: str
    course_id: str
    progress: float
    completed: bool
    rating: float | None
    hours_spent: float
    quiz_score: float | None
    timestamp: str

    def __post_init__(self) -> None:
        _check(bool(self.learner_id), "Interaction.learner_id 不能为空")
        _check(bool(self.course_id), "Interaction.course_id 不能为空")
        _check_unit(float(self.progress), "Interaction.progress")
        _check(float(self.hours_spent) >= 0, "Interaction.hours_spent 不能为负")
        if self.rating is not None:
            _check(1.0 <= float(self.rating) <= 5.0, "Interaction.rating 必须在 1~5")
        if self.quiz_score is not None:
            _check_unit(float(self.quiz_score), "Interaction.quiz_score")

    @property
    def implicit_feedback(self) -> float:
        """隐式反馈强度 0~1，供协同过滤使用。

        权重设计：完成度 0.5 + 是否完成 0.2 + 评分 0.2 + 测验 0.1。
        """
        score = 0.5 * float(self.progress)
        score += 0.2 if self.completed else 0.0
        if self.rating is not None:
            score += 0.2 * (float(self.rating) - 1.0) / 4.0
        if self.quiz_score is not None:
            score += 0.1 * float(self.quiz_score)
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "course_id": self.course_id,
            "progress": round(float(self.progress), 4),
            "completed": bool(self.completed),
            "rating": None if self.rating is None else round(float(self.rating), 1),
            "hours_spent": round(float(self.hours_spent), 2),
            "quiz_score": None if self.quiz_score is None else round(float(self.quiz_score), 4),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Interaction":
        rating = data.get("rating")
        quiz = data.get("quiz_score")
        return cls(
            learner_id=str(data["learner_id"]),
            course_id=str(data["course_id"]),
            progress=float(data.get("progress", 0.0)),
            completed=bool(data.get("completed", False)),
            rating=None if rating is None else float(rating),
            hours_spent=float(data.get("hours_spent", 0.0)),
            quiz_score=None if quiz is None else float(quiz),
            timestamp=str(data.get("timestamp", "")),
        )


# ============================================================
# 数据集容器
# ============================================================


@dataclass(slots=True)
class Dataset:
    """完整数据集。

    使用 ``dict`` 而非 ``list`` 保存实体，以保证 O(1) 主键查找；
    ``interactions`` 保留列表形态以便顺序遍历与切分。
    """

    knowledge_points: dict[str, KnowledgePoint]
    kp_edges: list[KPEdge]
    courses: dict[str, Course]
    learners: dict[str, Learner]
    goals: dict[str, Goal]
    interactions: list[Interaction]
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------- 校验 ----------
    def validate(self) -> None:
        """引用完整性校验。任何一处不合法都直接抛 ``SchemaError``。"""
        kp_ids = set(self.knowledge_points)
        course_ids = set(self.courses)
        learner_ids = set(self.learners)
        goal_ids = set(self.goals)

        _check(len(self.knowledge_points) > 0, "知识点集合为空")
        _check(len(self.courses) > 0, "课程集合为空")

        for edge in self.kp_edges:
            _check(edge.source in kp_ids, f"先修边端点不存在：{edge.source}")
            _check(edge.target in kp_ids, f"先修边端点不存在：{edge.target}")

        for course in self.courses.values():
            for cov in course.kp_coverage:
                _check(cov.kp_id in kp_ids, f"{course.course_id}: 覆盖了不存在的知识点 {cov.kp_id}")

        for goal in self.goals.values():
            for kp_id in goal.target_kp_ids:
                _check(kp_id in kp_ids, f"{goal.goal_id}: 目标知识点不存在 {kp_id}")

        for learner in self.learners.values():
            _check(learner.goal_id in goal_ids, f"{learner.learner_id}: 目标不存在 {learner.goal_id}")
            for kp_id in learner.initial_mastery:
                _check(kp_id in kp_ids, f"{learner.learner_id}: 初始掌握度含未知知识点 {kp_id}")

        seen: set[tuple[str, str]] = set()
        for inter in self.interactions:
            _check(inter.learner_id in learner_ids, f"交互记录含未知学习者 {inter.learner_id}")
            _check(inter.course_id in course_ids, f"交互记录含未知课程 {inter.course_id}")
            key = (inter.learner_id, inter.course_id)
            _check(key not in seen, f"重复的交互记录：{key}")
            seen.add(key)

    # ---------- 规模统计 ----------
    def stats(self) -> dict[str, Any]:
        """数据集规模统计（供 README / 仪表板 / API 使用）。"""
        return {
            "n_knowledge_points": len(self.knowledge_points),
            "n_kp_edges": len(self.kp_edges),
            "n_courses": len(self.courses),
            "n_learners": len(self.learners),
            "n_goals": len(self.goals),
            "n_interactions": len(self.interactions),
            "n_completed": sum(1 for i in self.interactions if i.completed),
            "completion_rate": round(
                sum(1 for i in self.interactions if i.completed) / max(1, len(self.interactions)),
                4,
            ),
            "domains": sorted({kp.domain for kp in self.knowledge_points.values()}),
        }

    # ---------- 序列化 ----------
    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_points": [kp.to_dict() for kp in self.knowledge_points.values()],
            "kp_edges": [e.to_dict() for e in self.kp_edges],
            "courses": [c.to_dict() for c in self.courses.values()],
            "learners": [l.to_dict() for l in self.learners.values()],
            "goals": [g.to_dict() for g in self.goals.values()],
            "interactions": [i.to_dict() for i in self.interactions],
            "meta": dict(self.meta),
        }

    # ---------- 便捷访问 ----------
    def course_of(self, course_id: str) -> Course:
        return self.courses[course_id]

    def learner_of(self, learner_id: str) -> Learner:
        return self.learners[learner_id]

    def goal_of_learner(self, learner_id: str) -> Goal:
        return self.goals[self.learners[learner_id].goal_id]

    def target_kps_of(self, learner_id: str) -> tuple[str, ...]:
        return self.goal_of_learner(learner_id).target_kp_ids

    def interactions_of(self, learner_id: str) -> list[Interaction]:
        return [i for i in self.interactions if i.learner_id == learner_id]

    def courses_by_domain(self, domain: str) -> list[Course]:
        return [c for c in self.courses.values() if c.domain == domain]

    def kps_by_domain(self, domain: str) -> list[KnowledgePoint]:
        return [kp for kp in self.knowledge_points.values() if kp.domain == domain]

    def course_titles(self) -> dict[str, str]:
        return {cid: c.title for cid, c in self.courses.items()}


__all__ = [
    "SchemaError",
    "KnowledgePoint",
    "KPEdge",
    "KPCoverage",
    "Course",
    "Goal",
    "Background",
    "LearningStyle",
    "Learner",
    "Interaction",
    "Dataset",
]
