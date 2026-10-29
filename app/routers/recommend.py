"""智能推荐路由（实现要求 ①：混合推荐 + 可解释理由）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Path, Query

from app.services import get_services

router = APIRouter(tags=["智能推荐"])


@router.get("/api/recommend/{learner_id}", summary="个性化课程推荐 Top-k")
def recommend(
    learner_id: str = Path(..., description="学习者 id"),
    k: int = Query(10, ge=1, le=50, description="返回条数"),
) -> dict[str, Any]:
    """返回推荐列表：综合分、5 项分项得分、中文理由、预期增益与完成概率。"""
    return get_services().recommend(learner_id, k=k)
