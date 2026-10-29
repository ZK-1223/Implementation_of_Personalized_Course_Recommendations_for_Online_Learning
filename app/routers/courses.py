"""课程库路由。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.services import get_services

router = APIRouter(tags=["课程库"])


@router.get("/api/courses", summary="课程库分页检索")
def list_courses(
    q: str | None = Query(None, description="关键词（课程名 / 讲师 / 知识点名）"),
    domain: str | None = Query(None, description="领域过滤"),
    level: int | None = Query(None, ge=1, le=5, description="难度过滤"),
    modality: str | None = Query(None, description="模态过滤：video/text/project/mixed"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    """返回分页结构，``items`` 元素为 ``Course.to_dict()`` 并附加 ``kp_names``。"""
    return get_services().list_courses(
        q=q, domain=domain, level=level, modality=modality, page=page, page_size=page_size
    )
