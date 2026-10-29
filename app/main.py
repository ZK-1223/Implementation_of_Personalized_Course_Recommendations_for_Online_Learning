"""LearnPath FastAPI 应用入口。

启动方式::

    .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

职责：
    * 装配路由（按领域拆分于 :mod:`app.routers`）；
    * 统一异常映射：``NotFound → 404``、``ModuleNotReady → 503``（算法模块未就绪时降级）；
    * 挂载单页应用：``/`` 返回 ``web/index.html``，``/static`` 指向 ``web/``；
    * 开启 CORS（演示环境放开为 ``*``）。

注意：本模块**不做任何重计算**，也不在 import 期加载数据，
因此算法子包缺失时应用仍可正常启动（相关接口返回 503）。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.routers import (
    analytics as analytics_router,
    courses as courses_router,
    evaluation as evaluation_router,
    feedback as feedback_router,
    graph as graph_router,
    learners as learners_router,
    meta as meta_router,
    planning as planning_router,
    recommend as recommend_router,
    social as social_router,
)
from app.services import VERSION, ModuleNotReady, NotFound
from learnpath.paths import WEB_DIR

logger = logging.getLogger("learnpath.api")

app = FastAPI(
    title="LearnPath 个性化学习路径推荐系统 API",
    version=VERSION,
    description=(
        "《智能应用系统设计》课程设计：在线学习的个性化课程推荐实现。\n\n"
        "* 学习者建模（BKT 掌握度追踪）\n"
        "* 混合推荐（内容 + 协同 + 质量 + 就绪度 + 社交）\n"
        "* 路径规划（多目标优化 + 帕累托前沿）\n"
        "* 学习效果预测（完成概率 / 掌握度增益）\n"
        "* 自适应反馈重规划、社交学习、学习分析、离线评测\n"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# 演示环境放开全部来源，便于本机用任意端口打开前端或做 curl 测试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 异常映射
# ============================================================


@app.exception_handler(NotFound)
async def _not_found_handler(request: Request, exc: NotFound) -> JSONResponse:
    """资源不存在 → 404 ``{"detail": "..."}``。"""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ModuleNotReady)
async def _module_not_ready_handler(request: Request, exc: ModuleNotReady) -> JSONResponse:
    """算法子模块尚未就绪 → 503，且说明清楚是哪个模块。"""
    logger.warning("模块未就绪：%s（path=%s）", exc.detail, request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": f"{exc.detail}，该接口暂不可用"},
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 FastAPI 默认的「错误列表」压平为契约要求的 ``{"detail": "字符串"}``。"""
    errors = exc.errors()
    if errors:
        first = errors[0]
        location = ".".join(
            str(part) for part in first.get("loc", ()) if part not in {"body", "query", "path"}
        )
        message = str(first.get("msg", "参数不合法"))
        detail = f"参数校验失败：{location or '请求'} {message}"
    else:  # pragma: no cover - 理论上不会发生
        detail = "参数校验失败"
    logger.info("参数校验失败：%s（path=%s）", detail, request.url.path)
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(Exception)
async def _unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：把未预期异常也转成契约格式，避免向演示前端抛出裸 500 HTML。"""
    logger.exception("未处理异常（path=%s）", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务内部错误：{type(exc).__name__}"},
    )


# ============================================================
# 路由装配
# ============================================================

app.include_router(meta_router.router)
app.include_router(courses_router.router)
app.include_router(learners_router.router)
app.include_router(recommend_router.router)
app.include_router(planning_router.router)
app.include_router(feedback_router.router)
app.include_router(social_router.router)
app.include_router(analytics_router.router)
app.include_router(graph_router.router)
app.include_router(evaluation_router.router)


# ============================================================
# 单页应用
# ============================================================

# /static/* 指向 web/ 目录（css / js / vendor）
app.mount("/static", StaticFiles(directory=str(WEB_DIR), html=False), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """返回单页应用入口（不做缓存，方便演示时修改前端立即生效）。"""
    return FileResponse(
        WEB_DIR / "index.html",
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """避免浏览器对 favicon 的 404 噪音（前端用内联 SVG 图标）。"""
    return Response(status_code=204)


@app.get("/api/__info", include_in_schema=False)
def api_info() -> dict[str, Any]:
    """开发期自检：列出已注册的 API 路径。"""
    routes = sorted(
        {route.path for route in app.routes if getattr(route, "path", "").startswith("/api/")}
    )
    return {"version": VERSION, "n_routes": len(routes), "paths": routes}
