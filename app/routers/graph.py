"""知识图谱可视化路由（实现要求 ②）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.services import get_services

router = APIRouter(tags=["知识图谱"])


@router.get("/api/knowledge-graph", summary="ECharts graph 数据（可领域过滤 / 掌握度染色）")
def knowledge_graph(
    domain: str | None = Query(None, description="领域过滤，缺省返回全图"),
    learner_id: str | None = Query(None, description="回填该学习者的掌握度用于节点染色"),
) -> dict[str, Any]:
    """返回节点、边、分类与高亮信息，供 ECharts graph 直接消费。"""
    return get_services().knowledge_graph(domain=domain, learner_id=learner_id)
