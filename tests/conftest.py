"""pytest 公共 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保仓库根目录在 sys.path 上（支持从任意目录运行 pytest）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learnpath.data.fixtures import build_tiny_dataset  # noqa: E402
from learnpath.knowledge.graph import KnowledgeGraph  # noqa: E402


@pytest.fixture(scope="session")
def tiny_dataset():
    """确定性微型数据集（35 知识点 / 28 课程 / 12 学习者）。"""
    return build_tiny_dataset()


@pytest.fixture(scope="session")
def graph(tiny_dataset):
    """基于微型数据集构建并通过校验的知识图谱。"""
    kg = KnowledgeGraph.from_dataset(tiny_dataset)
    kg.validate()
    return kg
