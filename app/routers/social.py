"""社交学习路由（实现要求 ⑦：同伴匹配与学习小组）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Path, Query

from app.services import get_services

router = APIRouter(tags=["社交学习"])


@router.get("/api/social/{learner_id}", summary="相似同伴与学习小组推荐")
def social(
    learner_id: str = Path(..., description="学习者 id"),
    k: int = Query(6, ge=1, le=20, description="同伴数量"),
) -> dict[str, Any]:
    """返回相似度最高的同伴、互补性学习小组与同期群摘要。"""
    return get_services().social(learner_id, k=k)
