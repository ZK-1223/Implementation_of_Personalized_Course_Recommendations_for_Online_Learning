"""HTTP 层数据模型（Pydantic v2）。

只定义「进入服务层之前」需要校验的结构与少数用于 OpenAPI 文档的响应模型；
响应体的真实结构由 :mod:`app.services` 组装，严格以 ``docs/API.md`` 为准。

设计取舍：
    * 请求模型一律 ``extra="forbid"``，避免前端字段拼写错误被静默吞掉；
    * 数值范围（概率 / 评分 / 进度）在校验层就挡住，服务层因此不必重复判断；
    * 未知 ``learner_id`` / ``course_id`` 属于「资源不存在」而非「请求非法」，
      故不在本文件校验，交由服务层抛 :class:`app.services.NotFound` 映射为 404。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: 路径规划预设键（与 ``learnpath.config.PLAN_PRESETS`` 的键保持一致）
PresetKey = Literal["balanced", "fast", "thorough"]


# ============================================================
# 请求体
# ============================================================


class FeedbackIn(BaseModel):
    """``POST /api/feedback`` 的请求体（一个学习者的单条学习反馈）。"""

    model_config = ConfigDict(extra="forbid")

    learner_id: str = Field(..., min_length=1, description="学习者 id，如 L-0001")
    course_id: str = Field(..., min_length=1, description="课程 id，如 C-0007")
    progress: float = Field(1.0, ge=0.0, le=1.0, description="学习进度 0~1")
    quiz_score: float | None = Field(None, ge=0.0, le=1.0, description="结课测验得分 0~1")
    rating: float | None = Field(None, ge=1.0, le=5.0, description="学习者评分 1~5")
    hours_spent: float | None = Field(None, ge=0.0, description="实际投入小时数，缺省按进度折算")
    preset: PresetKey = Field("balanced", description="重规划使用的方案预设")

    @field_validator("learner_id", "course_id")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("不能为空字符串")
        return value


# ============================================================
# 响应模型（用于 OpenAPI 文档展示，服务层返回 dict）
# ============================================================


class HealthOut(BaseModel):
    """``GET /api/health`` 响应。"""

    status: str = "ok"
    version: str = "1.0.0"
    dataset_loaded: bool = True


class ErrorOut(BaseModel):
    """统一错误体：``{"detail": "错误说明"}``。"""

    detail: str


class PageOut(BaseModel):
    """分页外壳：``{"items": [...], "total": int, "page": int, "page_size": int}``。"""

    items: list[Any] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class FeedbackOut(BaseModel):
    """``POST /api/feedback`` 响应的顶层字段（内部结构见 API 文档第 5 节）。"""

    learner_id: str
    course_id: str
    mastery_updates: dict[str, float] = Field(default_factory=dict)
    delta: dict[str, float] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    replanned: bool = False
    reason: str = ""


__all__ = ["PresetKey", "FeedbackIn", "HealthOut", "ErrorOut", "PageOut", "FeedbackOut"]
