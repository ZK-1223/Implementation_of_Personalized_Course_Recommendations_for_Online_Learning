"""系统信息路由：健康检查与首屏元数据。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.schemas import HealthOut
from app.services import get_services

router = APIRouter(tags=["系统信息"])


@router.get("/api/health", response_model=HealthOut, summary="健康检查")
def health() -> HealthOut:
    """返回服务状态、版本号与数据层是否就绪。"""
    return HealthOut(**get_services().health())


@router.get("/api/meta", summary="首屏元数据（统计 / 图谱规模 / 枚举 / 学习者列表）")
def meta() -> dict[str, Any]:
    """一次返回前端首屏所需的全部枚举与统计，避免首屏多次往返。"""
    return get_services().meta()
