"""全局配置：领域体系、算法超参数、目标权重与数据规模。

集中管理所有「可调参数」，便于报告中的消融实验与参数敏感性分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# ============================================================
# 1. 领域体系
# ============================================================

DOMAINS: Final[tuple[str, ...]] = (
    "编程开发",
    "数据与人工智能",
    "前端与设计",
    "网络与安全",
    "产品与职场",
    "语言学习",
    "数学基础",
)

DOMAIN_ICONS: Final[dict[str, str]] = {
    "编程开发": "code",
    "数据与人工智能": "chart",
    "前端与设计": "layout",
    "网络与安全": "shield",
    "产品与职场": "briefcase",
    "语言学习": "language",
    "数学基础": "function",
}

# 课程承载形式
MODALITIES: Final[tuple[str, ...]] = ("video", "text", "project", "mixed")

MODALITY_NAMES: Final[dict[str, str]] = {
    "video": "视频课",
    "text": "图文课",
    "project": "项目实战",
    "mixed": "混合式",
}

# 学习动机
MOTIVATIONS: Final[tuple[str, ...]] = ("求职", "晋升", "兴趣", "学业", "转行")

# 难度等级取值范围（1~5）
MIN_LEVEL: Final[int] = 1
MAX_LEVEL: Final[int] = 5

# 掌握度阈值：达到该值视为「已掌握」，路径规划时可跳过
MASTERY_MASTERED: Final[float] = 0.85
# 掌握度阈值：低于该值视为「知识缺口」
MASTERY_GAP: Final[float] = 0.60


# ============================================================
# 2. 数据生成规模
# ============================================================


@dataclass(frozen=True)
class DatasetScale:
    """数据集规模配置。"""

    n_knowledge_points: int
    n_courses: int
    n_learners: int
    n_interactions: int
    seed: int


DEFAULT_SCALE: Final[DatasetScale] = DatasetScale(
    n_knowledge_points=420,
    n_courses=560,
    n_learners=720,
    n_interactions=14000,
    seed=20261029,
)

TINY_SCALE: Final[DatasetScale] = DatasetScale(
    n_knowledge_points=35,
    n_courses=28,
    n_learners=12,
    n_interactions=180,
    seed=7,
)


# ============================================================
# 3. 学习者建模 / 知识追踪（BKT-lite）
# ============================================================


@dataclass(frozen=True)
class BKTParams:
    """贝叶斯知识追踪四参数（用于掌握度更新）。

    Attributes:
        p_init: 未学习前已掌握的概率。
        p_learn: 单次学习后可掌握的概率（受学习者能力调制）。
        p_slip: 已掌握却答错/未达标的概率。
        p_guess: 未掌握却表现达标（蒙对）的概率。
    """

    p_init: float = 0.10
    p_learn: float = 0.40
    p_slip: float = 0.10
    p_guess: float = 0.20


BKT: Final[BKTParams] = BKTParams()

# 掌握度遗忘衰减：每空闲一周的保留系数
MASTERY_DECAY_PER_WEEK: Final[float] = 0.985


# ============================================================
# 4. 推荐融合权重
# ============================================================


@dataclass(frozen=True)
class RecommendWeights:
    """混合推荐各信号权重（和为 1.0）。"""

    content: float = 0.40
    collaborative: float = 0.25
    quality: float = 0.15
    readiness: float = 0.12
    social: float = 0.08


RECOMMEND_WEIGHTS: Final[RecommendWeights] = RecommendWeights()

# 多路召回参数
CF_SVD_COMPONENTS: Final[int] = 32
RECOMMEND_FETCH_MULTIPLIER: Final[int] = 5   # 每次召回候选 = k * multiplier
MMR_LAMBDA: Final[float] = 0.75              # 多样性重排序参数


# ============================================================
# 5. 路径规划 / 多目标优化
# ============================================================


@dataclass(frozen=True)
class ObjectiveWeights:
    """多目标优化的方案选择权重（TOPSIS 加权）。"""

    gain: float = 0.40        # 学习收益（最大化）
    time: float = 0.25        # 时间成本（最小化）
    smoothness: float = 0.20  # 难度梯度平滑度（最大化）
    fit: float = 0.15         # 学习风格 / 模态匹配度（最大化）


# 三种偏好预设（前端可切换）
PLAN_PRESETS: Final[dict[str, ObjectiveWeights]] = {
    "balanced": ObjectiveWeights(gain=0.40, time=0.25, smoothness=0.20, fit=0.15),
    "fast": ObjectiveWeights(gain=0.30, time=0.45, smoothness=0.10, fit=0.15),
    "thorough": ObjectiveWeights(gain=0.50, time=0.15, smoothness=0.20, fit=0.15),
}

PLAN_PRESET_NAMES: Final[dict[str, str]] = {
    "balanced": "均衡方案",
    "fast": "速通方案",
    "thorough": "夯实方案",
}

# NSGA-II 参数
NSGA_POPULATION: Final[int] = 60
NSGA_GENERATIONS: Final[int] = 40
NSGA_CROSSOVER_RATE: Final[float] = 0.85
NSGA_MUTATION_RATE: Final[float] = 0.20

# 路径规划约束
MAX_PATH_COURSES: Final[int] = 40        # 单条路径最多课程数
MIN_COURSE_GAIN: Final[float] = 0.01     # 低于该收益增量的课程不入路径


# ============================================================
# 6. 自适应调整
# ============================================================


@dataclass(frozen=True)
class AdaptiveThresholds:
    """反馈驱动的重规划触发阈值。"""

    low_quiz_score: float = 0.60      # 测验分低于此值 → 需要补强
    high_quiz_score: float = 0.90     # 高于此值 → 可跳过后续同类课
    behind_schedule: float = 0.70     # 实际进度 / 计划进度 低于此值 → 进度落后
    ahead_schedule: float = 1.30      # 高于此值 → 进度超前


ADAPTIVE: Final[AdaptiveThresholds] = AdaptiveThresholds()


# ============================================================
# 7. 社交学习
# ============================================================

PEER_TOP_K: Final[int] = 6
STUDY_GROUP_SIZE: Final[int] = 4


# ============================================================
# 8. 离线评测
# ============================================================

EVAL_K: Final[int] = 10
EVAL_TEST_RATIO: Final[float] = 0.2
EVAL_SEED: Final[int] = 42


# ============================================================
# 9. 日期与展示
# ============================================================

# 数据集时间基准（用于构造带时间戳的行为日志）
DATA_TIME_BASE: Final[str] = "2026-02-01T00:00:00"

WEEK_DAYS: Final[int] = 7
