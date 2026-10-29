"""学习分析仪表板路由（实现要求 ⑧）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Path, Query

from app.schemas import PresetKey
from app.services import get_services

router = APIRouter(tags=["学习分析"])


@router.get("/api/analytics/{learner_id}", summary="学习分析仪表板数据")
def analytics(
    learner_id: str = Path(..., description="学习者 id"),
    preset: PresetKey = Query("balanced", description="用于计算完成概率预测的路径预设"),
) -> dict[str, Any]:
    """返回 KPI、掌握度曲线、领域雷达、难度分布、模态构成、时间分配、完成概率与缺口。"""
    return get_services().analytics(learner_id, preset=preset)
