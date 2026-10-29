"""知识图谱模块（实现要求第 2 条）。

对外契约（由 :mod:`learnpath.knowledge.graph` 提供）：

    class KnowledgeGraph:
        def __init__(self, knowledge_points: Mapping[str, KnowledgePoint],
                     edges: Iterable[KPEdge]) -> None
        def validate(self) -> None                     # 有环 → raise CycleError
        def predecessors(self, kp_id: str) -> tuple[str, ...]
        def successors(self, kp_id: str) -> tuple[str, ...]
        def prerequisite_closure(self, kp_ids: Iterable[str]) -> frozenset[str]
        def descendants(self, kp_id: str) -> frozenset[str]
        def topological_order(self, kp_ids: Iterable[str] | None = None) -> list[str]
        def depth(self, kp_id: str) -> int
        def difficulty_of(self, kp_id: str) -> int
        def domain_of(self, kp_id: str) -> str
        def course_prerequisites(self, course: Course) -> frozenset[str]
        def to_echarts(self, domain: str | None = None,
                       highlight: Iterable[str] = ()) -> dict
"""
