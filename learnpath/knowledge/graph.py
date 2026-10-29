"""课程知识图谱：知识点 DAG 上的先修关系推理。

本模块是全系统的「知识点关系真相来源」，所有路径规划、缺口计算、
难度分层都在此之上构建。实现要求中的第 2 条
「课程知识图谱：建立课程间的依赖关系和知识点关联，形成结构化的知识图谱」
由本模块承担。

内核：以知识点为节点、先修关系为有向边的 DAG。
    边 ``source -> target`` 表示「掌握 source 是学习 target 的前提」。
    课程通过 ``Course.kp_coverage`` 映射到若干知识点，因此课程层面的先修
    关系可以通过知识点闭包推导出来。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from learnpath.data.schema import Course, KPEdge, KnowledgePoint


class CycleError(ValueError):
    """知识点图中出现环，违反 DAG 约束。"""


@dataclass(frozen=True, slots=True)
class GraphStats:
    """图谱规模统计。"""

    n_nodes: int
    n_edges: int
    n_roots: int
    n_leaves: int
    max_depth: int
    avg_out_degree: float
    domains: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_roots": self.n_roots,
            "n_leaves": self.n_leaves,
            "max_depth": self.max_depth,
            "avg_out_degree": round(self.avg_out_degree, 3),
            "domains": list(self.domains),
        }


class KnowledgeGraph:
    """知识点有向无环图。

    Args:
        knowledge_points: ``kp_id -> KnowledgePoint`` 映射。
        edges: 先修边集合。

    Raises:
        CycleError: 边集合构成环时（在 :meth:`validate` 中检查）。
    """

    __slots__ = ("_kps", "_succ", "_pred", "_depth_cache", "_validated")

    def __init__(
        self,
        knowledge_points: Mapping[str, KnowledgePoint],
        edges: Iterable[KPEdge],
    ) -> None:
        self._kps: dict[str, KnowledgePoint] = dict(knowledge_points)
        self._succ: dict[str, list[str]] = {kp_id: [] for kp_id in self._kps}
        self._pred: dict[str, list[str]] = {kp_id: [] for kp_id in self._kps}
        self._depth_cache: dict[str, int] = {}
        self._validated = False

        seen: set[tuple[str, str]] = set()
        for edge in edges:
            if edge.source not in self._kps:
                raise KeyError(f"先修边起点不在知识点集合中：{edge.source}")
            if edge.target not in self._kps:
                raise KeyError(f"先修边终点不在知识点集合中：{edge.target}")
            key = (edge.source, edge.target)
            if key in seen:
                continue
            seen.add(key)
            self._succ[edge.source].append(edge.target)
            self._pred[edge.target].append(edge.source)

        for node in self._succ:
            self._succ[node].sort()
            self._pred[node].sort()

    # ========================================================
    # 构造与校验
    # ========================================================
    @classmethod
    def from_dataset(cls, dataset: Any) -> "KnowledgeGraph":
        """从 :class:`learnpath.data.schema.Dataset` 构造。"""
        return cls(dataset.knowledge_points, dataset.kp_edges)

    def validate(self) -> None:
        """校验无环（Kahn 拓扑排序），有环则抛 :class:`CycleError`。"""
        indegree = {node: len(self._pred[node]) for node in self._kps}
        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for nxt in self._succ[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if visited != len(self._kps):
            remaining = sorted(n for n, d in indegree.items() if d > 0)
            raise CycleError(f"知识点图存在环，涉及节点 {remaining[:10]}{'...' if len(remaining) > 10 else ''}")
        self._validated = True

    def _ensure_valid(self) -> None:
        if not self._validated:
            self.validate()

    # ========================================================
    # 基础查询
    # ========================================================
    def __len__(self) -> int:
        return len(self._kps)

    def __contains__(self, kp_id: object) -> bool:
        return kp_id in self._kps

    @property
    def kps(self) -> Mapping[str, KnowledgePoint]:
        return self._kps

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self._kps)

    def node(self, kp_id: str) -> KnowledgePoint:
        return self._kps[kp_id]

    def name_of(self, kp_id: str) -> str:
        return self._kps[kp_id].name

    def names_of(self, kp_ids: Iterable[str]) -> list[str]:
        return [self._kps[k].name for k in kp_ids if k in self._kps]

    def domain_of(self, kp_id: str) -> str:
        return self._kps[kp_id].domain

    def difficulty_of(self, kp_id: str) -> int:
        return self._kps[kp_id].difficulty

    def predecessors(self, kp_id: str) -> tuple[str, ...]:
        """直接先修（父节点）。"""
        return tuple(self._pred.get(kp_id, ()))

    def successors(self, kp_id: str) -> tuple[str, ...]:
        """直接后继（可解锁的知识点）。"""
        return tuple(self._succ.get(kp_id, ()))

    def roots(self) -> list[str]:
        return sorted(k for k in self._kps if not self._pred[k])

    def leaves(self) -> list[str]:
        return sorted(k for k in self._kps if not self._succ[k])

    # ========================================================
    # 闭包与可达性
    # ========================================================
    def prerequisite_closure(self, kp_ids: Iterable[str]) -> frozenset[str]:
        """返回所有先修知识点的闭包（包含自身）。

        例：要学「回归建模」，需先掌握「数据清洗 / 描述性统计 / 假设检验 …」。
        """
        seeds = [k for k in kp_ids]
        for kp_id in seeds:
            if kp_id not in self._kps:
                raise KeyError(f"未知知识点：{kp_id}")

        closure: set[str] = set()
        stack = list(seeds)
        while stack:
            node = stack.pop()
            if node in closure:
                continue
            closure.add(node)
            stack.extend(self._pred[node])
        return frozenset(closure)

    def descendants(self, kp_id: str) -> frozenset[str]:
        """返回该知识点的所有后继闭包（含自身）。"""
        if kp_id not in self._kps:
            raise KeyError(f"未知知识点：{kp_id}")
        result: set[str] = set()
        stack = [kp_id]
        while stack:
            node = stack.pop()
            if node in result:
                continue
            result.add(node)
            stack.extend(self._succ[node])
        return frozenset(result)

    def is_prerequisite_of(self, source: str, target: str) -> bool:
        """``source`` 是否为 ``target`` 的（间接）先修。"""
        return source in self.prerequisite_closure([target])

    # ========================================================
    # 拓扑排序与分层
    # ========================================================
    def topological_order(self, kp_ids: Iterable[str] | None = None) -> list[str]:
        """子图上的确定性拓扑序（同层按 id 排序，保证结果可复现）。"""
        subset = set(self._kps) if kp_ids is None else set(kp_ids)
        for kp_id in subset:
            if kp_id not in self._kps:
                raise KeyError(f"未知知识点：{kp_id}")

        indegree = {
            node: sum(1 for p in self._pred[node] if p in subset) for node in subset
        }
        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in self._succ[node]:
                if nxt not in subset:
                    continue
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(subset):
            raise CycleError("子图存在环，无法拓扑排序")
        return order

    def layers(self, kp_ids: Iterable[str] | None = None) -> list[list[str]]:
        """按拓扑层级分组，用于路径规划的分阶段排布。"""
        subset = set(self._kps) if kp_ids is None else set(kp_ids)
        remaining = set(subset)
        result: list[list[str]] = []
        while remaining:
            layer = sorted(
                n for n in remaining if not any(p in remaining for p in self._pred[n])
            )
            if not layer:  # 理论上不会发生（已校验无环）
                raise CycleError("分层失败：存在环")
            result.append(layer)
            remaining.difference_update(layer)
        return result

    def depth(self, kp_id: str) -> int:
        """最长先修链长度（根节点为 0），刻画知识点的「前置深度」。"""
        if kp_id in self._depth_cache:
            return self._depth_cache[kp_id]
        if kp_id not in self._kps:
            raise KeyError(f"未知知识点：{kp_id}")

        # 深度取决于祖先链，因此在与自身相连的先修闭包上做拓扑推进
        closure = self.prerequisite_closure([kp_id])
        order = self.topological_order(closure)
        local: dict[str, int] = {}
        for node in order:
            preds = [p for p in self._pred[node] if p in local]
            local[node] = 0 if not preds else 1 + max(local[p] for p in preds)
        self._depth_cache[kp_id] = local[kp_id]
        return local[kp_id]

    # ========================================================
    # 课程 ↔ 知识点映射
    # ========================================================
    def course_prerequisites(self, course: Course) -> frozenset[str]:
        """课程要求的先修知识点（不含课程自身覆盖的知识点）。"""
        own = {c.kp_id for c in course.kp_coverage}
        closure = self.prerequisite_closure(course.kp_ids)
        return frozenset(closure - own)

    def course_closure(self, course: Course) -> frozenset[str]:
        """课程及其全部先修知识点。"""
        return self.prerequisite_closure(course.kp_ids)

    def courses_by_domain(self, courses: Mapping[str, Course], domain: str) -> list[Course]:
        return [c for c in courses.values() if c.domain == domain]

    # ========================================================
    # 统计与可视化
    # ========================================================
    def stats(self) -> GraphStats:
        self._ensure_valid()
        n_nodes = len(self._kps)
        n_edges = sum(len(v) for v in self._succ.values())
        depths = [self.depth(k) for k in self._kps] or [0]
        return GraphStats(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_roots=len(self.roots()),
            n_leaves=len(self.leaves()),
            max_depth=max(depths),
            avg_out_degree=n_edges / n_nodes if n_nodes else 0.0,
            domains=tuple(sorted({kp.domain for kp in self._kps.values()})),
        )

    def to_echarts(
        self,
        domain: str | None = None,
        highlight: Iterable[str] = (),
        mastery: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """导出前端 ECharts graph 图表所需结构。

        Args:
            domain: 仅导出该领域（None 表示全图）。
            highlight: 需要高亮的知识点 id（例如目标知识点）。
            mastery: ``kp_id -> 掌握度``，用于节点染色。

        Returns:
            ``{"nodes": [...], "links": [...], "categories": [...]}``
        """
        nodes_all = [
            kp for kp in self._kps.values() if domain is None or kp.domain == domain
        ]
        keep = {kp.kp_id for kp in nodes_all}
        highlight_set = set(highlight)
        mastery = mastery or {}

        categories = sorted({kp.domain for kp in nodes_all})
        cat_index = {c: i for i, c in enumerate(categories)}

        nodes = []
        for kp in sorted(nodes_all, key=lambda x: x.kp_id):
            value = mastery.get(kp.kp_id)
            nodes.append(
                {
                    "id": kp.kp_id,
                    "name": kp.name,
                    "category": cat_index[kp.domain],
                    "domain": kp.domain,
                    "difficulty": kp.difficulty,
                    "depth": self.depth(kp.kp_id),
                    "mastery": None if value is None else round(float(value), 3),
                    "symbolSize": 12 + 4 * kp.difficulty,
                    "highlight": kp.kp_id in highlight_set,
                }
            )

        links = [
            {"source": src, "target": dst}
            for src in self._succ
            if src in keep
            for dst in self._succ[src]
            if dst in keep
        ]

        return {"nodes": nodes, "links": links, "categories": categories}

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"KnowledgeGraph(nodes={len(self._kps)}, "
            f"edges={sum(len(v) for v in self._succ.values())})"
        )


__all__ = ["KnowledgeGraph", "CycleError", "GraphStats"]
