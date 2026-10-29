"""学习者画像路由（实现要求 ①：学习者建模）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Path, Query

from app.services import get_services

router = APIRouter(tags=["学习者画像"])


@router.get("/api/learners", summary="学习者分页检索")
def list_learners(
    q: str | None = Query(None, description="关键词（昵称 / 学习者 id）"),
    goal_id: str | None = Query(None, description="目标过滤"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    """分页返回学习者摘要（供顶部切换器与画像列表使用）。"""
    return get_services().list_learners(q=q, goal_id=goal_id, page=page, page_size=page_size)


@router.get("/api/learners/{learner_id}", summary="学习者画像详情")
def learner_detail(
    learner_id: str = Path(..., description="学习者 id，如 L-0001"),
) -> dict[str, Any]:
    """返回基础信息、目标、掌握度画像与学习历史。"""
    return get_services().learner_detail(learner_id)
