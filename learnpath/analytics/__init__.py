"""学习分析仪表板（实现要求第 8 条）与离线评测。

    * :mod:`learnpath.analytics.dashboard` —— 仪表板数据聚合
    * :mod:`learnpath.analytics.evaluation` —— 推荐算法离线评测指标
"""

from __future__ import annotations

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
]
