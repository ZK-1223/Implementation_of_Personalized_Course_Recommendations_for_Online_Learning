"""学习路径规划路由（实现要求 ③④：多目标优化 + 帕累托前沿）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Path, Query

from app.schemas import PresetKey
from app.services import get_services

router = APIRouter(tags=["学习路径"])


@router.get("/api/path/{learner_id}", summary="学习路径规划（含三种预设对比）")
def learning_path(
    learner_id: str = Path(..., description="学习者 id"),
    preset: PresetKey = Query("balanced", description="方案预设：balanced/fast/thorough"),
) -> dict[str, Any]:
    """返回路径步骤、周计划、里程碑，以及另外两种预设方案的对比摘要。"""
    return get_services().path(learner_id, preset=preset)


@router.get("/api/path/{learner_id}/pareto", summary="帕累托前沿（时间—收益权衡）")
def pareto_front(
    learner_id: str = Path(..., description="学习者 id"),
    preset: PresetKey = Query("balanced", description="用于标记 is_selected 的预设"),
) -> dict[str, Any]:
    """返回散点图所需的全部候选方案点，并标记当前选中方案。"""
    return get_services().pareto(learner_id, preset=preset)
