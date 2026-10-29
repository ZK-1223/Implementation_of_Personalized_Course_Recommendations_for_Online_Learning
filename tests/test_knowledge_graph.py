"""知识图谱单元测试。

覆盖：
    * 构造期非法边的拒绝
    * 环检测
    * 先修闭包 / 后继闭包的正确性
    * 拓扑序与分层的不变量
    * 最长链深度
    * 课程级先修推导
    * ECharts 导出结构
"""

from __future__ import annotations

import pytest

from learnpath.data.schema import KPEdge
from learnpath.knowledge.graph import CycleError, KnowledgeGraph


# ============================================================
# 构造与校验
# ============================================================


class TestConstruction:
    def test_from_dataset(self, tiny_dataset, graph):
        assert len(graph) == len(tiny_dataset.knowledge_points)
        assert graph.kps is not None

    def test_unknown_edge_endpoint_rejected(self, tiny_dataset):
        bad = KPEdge(source="KP-NOT-EXIST", target=next(iter(tiny_dataset.knowledge_points)))
        with pytest.raises(KeyError):
            KnowledgeGraph(tiny_dataset.knowledge_points, [bad])

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError):
            KPEdge(source="KP-A", target="KP-A")

    def test_duplicate_edges_deduplicated(self, tiny_dataset):
        kp_ids = list(tiny_dataset.knowledge_points)
        edge = KPEdge(source=kp_ids[0], target=kp_ids[1])
        kg = KnowledgeGraph(tiny_dataset.knowledge_points, [edge, edge, edge])
        assert len(kg.successors(kp_ids[0])) == 1

    def test_cycle_detection(self, tiny_dataset):
        kp_ids = list(tiny_dataset.knowledge_points)
        # 从 tiny 数据集已有的链上构造一条回边必然成环
        chain = kg_chain(tiny_dataset)
        back_edge = KPEdge(source=chain[-1], target=chain[0])
        bad = KnowledgeGraph(
            tiny_dataset.knowledge_points, list(tiny_dataset.kp_edges) + [back_edge]
        )
        with pytest.raises(CycleError):
            bad.validate()

    def test_validate_passes_on_dag(self, graph):
        graph.validate()  # 不抛异常即通过


def kg_chain(dataset) -> list[str]:
    """找出 tiny 数据集中最长的一条领域内先修链（用于构造环）。"""
    succ: dict[str, list[str]] = {k: [] for k in dataset.knowledge_points}
    for e in dataset.kp_edges:
        succ[e.source].append(e.target)
    starts = [k for k in dataset.knowledge_points if not any(k == e.target for e in dataset.kp_edges)]
    best = [starts[0]]
    for start in starts:
        path, node = [start], start
        while succ[node]:
            node = succ[node][0]
            path.append(node)
        if len(path) > len(best):
            best = path
    return best


# ============================================================
# 闭包与可达性
# ============================================================


class TestClosure:
    def test_closure_contains_self(self, graph):
        kp = graph.node_ids[0]
        assert kp in graph.prerequisite_closure([kp])

    def test_closure_contains_direct_predecessors(self, graph):
        for kp in graph.node_ids:
            preds = graph.predecessors(kp)
            if preds:
                closure = graph.prerequisite_closure([kp])
                assert set(preds) <= closure
                break

    def test_closure_is_transitive(self, graph):
        # 对任意知识点，其所有先修的先修也在闭包内
        for kp in graph.node_ids:
            closure = graph.prerequisite_closure([kp])
            for ancestor in closure:
                assert graph.prerequisite_closure([ancestor]) <= closure

    def test_closure_multi_seed_is_union(self, graph):
        a, b = graph.node_ids[0], graph.node_ids[-1]
        union = graph.prerequisite_closure([a]) | graph.prerequisite_closure([b])
        assert graph.prerequisite_closure([a, b]) == union

    def test_descendants_reachability(self, graph):
        for src in graph.node_ids:
            for dst in graph.descendants(src):
                assert src in graph.prerequisite_closure([dst])

    def test_unknown_kp_raises(self, graph):
        with pytest.raises(KeyError):
            graph.prerequisite_closure(["KP-NOT-EXIST"])


# ============================================================
# 拓扑序与分层
# ============================================================


class TestTopology:
    def test_topological_order_respects_edges(self, graph):
        order = graph.topological_order()
        position = {kp: i for i, kp in enumerate(order)}
        assert len(order) == len(graph)
        for src in graph.node_ids:
            for dst in graph.successors(src):
                assert position[src] < position[dst], f"{src} 应排在 {dst} 之前"

    def test_topological_order_is_deterministic(self, graph):
        assert graph.topological_order() == graph.topological_order()

    def test_layers_cover_all_nodes_once(self, graph):
        layers = graph.layers()
        flat = [kp for layer in layers for kp in layer]
        assert len(flat) == len(set(flat)) == len(graph)

    def test_layers_satisfy_precedence(self, graph):
        """先修必须出现在不晚于后继的层里。"""
        layer_of = {}
        for index, layer in enumerate(graph.layers()):
            for kp in layer:
                layer_of[kp] = index
        for src in graph.node_ids:
            for dst in graph.successors(src):
                assert layer_of[src] < layer_of[dst]

    def test_subset_topological_order(self, graph):
        subset = graph.node_ids[:5]
        order = graph.topological_order(subset)
        assert set(order) == set(subset)

    def test_depth_monotonic_along_edges(self, graph):
        for src in graph.node_ids:
            for dst in graph.successors(src):
                assert graph.depth(dst) > graph.depth(src)

    def test_roots_have_zero_depth(self, graph):
        for root in graph.roots():
            assert graph.depth(root) == 0


# ============================================================
# 课程 ↔ 知识点映射
# ============================================================


class TestCourseMapping:
    def test_course_prerequisites_exclude_own_kps(self, graph, tiny_dataset):
        course = next(iter(tiny_dataset.courses.values()))
        prereq = graph.course_prerequisites(course)
        assert not (set(course.kp_ids) & prereq)

    def test_course_closure_includes_own_and_prereq(self, graph, tiny_dataset):
        course = next(iter(tiny_dataset.courses.values()))
        closure = graph.course_closure(course)
        assert set(course.kp_ids) <= closure
        assert graph.course_prerequisites(course) <= closure

    def test_hard_course_has_prerequisites(self, graph, tiny_dataset):
        """高难度课程应当先修非空——否则图谱约束没有实际意义。"""
        hard = [c for c in tiny_dataset.courses.values() if c.level >= 4]
        assert any(graph.course_prerequisites(c) for c in hard)


# ============================================================
# 统计与导出
# ============================================================


class TestExport:
    def test_stats_consistency(self, graph, tiny_dataset):
        stats = graph.stats()
        assert stats.n_nodes == len(tiny_dataset.knowledge_points)
        assert stats.n_edges == len(tiny_dataset.kp_edges)
        assert stats.n_roots == len(graph.roots())
        assert stats.n_leaves == len(graph.leaves())
        assert stats.max_depth >= 1

    def test_to_echarts_full_graph(self, graph):
        data = graph.to_echarts()
        assert len(data["nodes"]) == len(graph)
        assert len(data["categories"]) == len({kp.domain for kp in graph.kps.values()})
        node_ids = {n["id"] for n in data["nodes"]}
        for link in data["links"]:
            assert link["source"] in node_ids and link["target"] in node_ids

    def test_to_echarts_domain_filter(self, graph, tiny_dataset):
        domain = tiny_dataset.knowledge_points[graph.node_ids[0]].domain
        data = graph.to_echarts(domain=domain)
        assert all(n["domain"] == domain for n in data["nodes"])
        assert len(data["nodes"]) == sum(
            1 for kp in tiny_dataset.knowledge_points.values() if kp.domain == domain
        )

    def test_to_echarts_mastery_and_highlight(self, graph):
        kp = graph.node_ids[0]
        data = graph.to_echarts(highlight=[kp], mastery={kp: 0.75})
        node = next(n for n in data["nodes"] if n["id"] == kp)
        assert node["mastery"] == 0.75
        assert node["highlight"] is True
        assert node["symbolSize"] >= 12
