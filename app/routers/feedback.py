"""自适应反馈路由（实现要求 ⑥：学习反馈驱动重规划）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.schemas import FeedbackIn
from app.services import get_services

router = APIRouter(tags=["自适应反馈"])


@router.post("/api/feedback", summary="提交学习反馈并触发掌握度更新 / 路径重规划")
def submit_feedback(payload: FeedbackIn) -> dict[str, Any]:
    """真实调用 ``AdaptiveEngine``：更新掌握度、给出动作说明、必要时重规划路径。"""
    return get_services().apply_feedback(
        learner_id=payload.learner_id,
        course_id=payload.course_id,
        progress=payload.progress,
        quiz_score=payload.quiz_score,
        rating=payload.rating,
        hours_spent=payload.hours_spent,
        preset=payload.preset,
    )
