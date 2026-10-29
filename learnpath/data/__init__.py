"""数据层：数据结构契约、数据集生成与读写。

对外主入口：
    * :class:`Dataset` —— 数据集容器（定义于 :mod:`learnpath.data.schema`）
    * :func:`learnpath.data.loader.load_dataset` —— 从 ``data/`` 读取
    * :func:`learnpath.data.generator.generate_dataset` —— 程序化生成
    * :func:`learnpath.data.fixtures.build_tiny_dataset` —— 微型数据集（测试用）
"""

from __future__ import annotations

from learnpath.data.schema import (
    Background,
    Course,
    Dataset,
    Goal,
    Interaction,
    KPCoverage,
    KPEdge,
    KnowledgePoint,
    Learner,
    LearningStyle,
    SchemaError,
)

__all__ = [
    "Background",
    "Course",
    "Dataset",
    "Goal",
    "Interaction",
    "KPCoverage",
    "KPEdge",
    "KnowledgePoint",
    "Learner",
    "LearningStyle",
    "SchemaError",
]
