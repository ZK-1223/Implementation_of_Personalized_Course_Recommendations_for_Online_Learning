"""系统评测路由（离线指标）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.services import get_services

router = APIRouter(tags=["系统评测"])


@router.get("/api/evaluation", summary="离线评测报告（推荐算法对比 / 效果模型 / 消融）")
def evaluation() -> dict[str, Any]:
    """直接返回 ``learnpath.analytics.evaluation.build_evaluation`` 的结果。"""
    return get_services().evaluation()
