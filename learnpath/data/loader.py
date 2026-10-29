"""数据集读写：按 :data:`learnpath.paths.DATASET_FILES` 拆分成多个 JSON 文件。

存储约定
--------
* 每个实体一个大文件（``knowledge_points.json`` / ``courses.json`` / ...），
  ``dataset_meta.json`` 保存 :attr:`learnpath.data.schema.Dataset.meta`；
* 统一 ``ensure_ascii=False, indent=2`` 写出，便于人阅读与 code review，
  中文不转义成 ``\\uXXXX``；
* 读入后一定调用 :meth:`learnpath.data.schema.Dataset.validate`，
  保证「磁盘上的数据」和「内存里的契约」一致，避免坏数据悄悄流入下游。

典型用法::

    from learnpath.data.loader import load_dataset, load_or_build, save_dataset

    dataset = load_or_build()          # 有数据就加载，没有就生成并落盘
    save_dataset(dataset, "data")      # 显式指定目录
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from learnpath.config import DEFAULT_SCALE, DatasetScale
from learnpath.data.schema import (
    Course,
    Dataset,
    Goal,
    Interaction,
    KPEdge,
    KnowledgePoint,
    Learner,
)
from learnpath.paths import (
    COURSES_FILE,
    DATASET_FILES,
    DATASET_META_FILE,
    DATA_DIR,
    GOALS_FILE,
    INTERACTIONS_FILE,
    KNOWLEDGE_POINTS_FILE,
    KP_EDGES_FILE,
    LEARNERS_FILE,
    ensure_dir,
)

#: 读取时允许缺失的「可选」文件（数据目录只放了部分实体时）
_REQUIRED_FILES = (
    KNOWLEDGE_POINTS_FILE,
    KP_EDGES_FILE,
    COURSES_FILE,
    LEARNERS_FILE,
    GOALS_FILE,
    INTERACTIONS_FILE,
)


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    """把 ``None`` / 字符串 / ``Path`` 统一解析成数据目录。"""
    if data_dir is None:
        return DATA_DIR
    return Path(data_dir).expanduser()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


# ============================================================
# 读
# ============================================================


def load_dataset(data_dir: Path | str | None = None) -> Dataset:
    """从数据目录读取完整数据集并做引用完整性校验。

    Args:
        data_dir: 数据目录，缺省为 :data:`learnpath.paths.DATA_DIR`。

    Returns:
        校验通过的 :class:`~learnpath.data.schema.Dataset`。

    Raises:
        FileNotFoundError: 必需文件缺失（提示先运行 ``scripts/gen_data.py``）。
        learnpath.data.schema.SchemaError: 数据不满足契约。
    """
    directory = resolve_data_dir(data_dir)
    missing = [name for name in _REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"数据目录 {directory} 缺少文件 {missing}；"
            "请先运行 python scripts/gen_data.py --scale full --out data"
        )

    knowledge_points = {
        item["kp_id"]: KnowledgePoint.from_dict(item)
        for item in _read_json(directory / KNOWLEDGE_POINTS_FILE)
    }
    kp_edges = [KPEdge.from_dict(item) for item in _read_json(directory / KP_EDGES_FILE)]
    courses = {
        item["course_id"]: Course.from_dict(item)
        for item in _read_json(directory / COURSES_FILE)
    }
    learners = {
        item["learner_id"]: Learner.from_dict(item)
        for item in _read_json(directory / LEARNERS_FILE)
    }
    goals = {
        item["goal_id"]: Goal.from_dict(item)
        for item in _read_json(directory / GOALS_FILE)
    }
    interactions = [
        Interaction.from_dict(item) for item in _read_json(directory / INTERACTIONS_FILE)
    ]

    meta_path = directory / DATASET_META_FILE
    meta: dict[str, Any] = _read_json(meta_path) if meta_path.is_file() else {}

    dataset = Dataset(
        knowledge_points=knowledge_points,
        kp_edges=kp_edges,
        courses=courses,
        learners=learners,
        goals=goals,
        interactions=interactions,
        meta=meta,
    )
    dataset.validate()
    return dataset


# ============================================================
# 写
# ============================================================


def save_dataset(dataset: Dataset, data_dir: Path | str | None = None) -> list[Path]:
    """把数据集按实体拆成多个 JSON 文件写出。

    Args:
        dataset: 待保存的数据集。
        data_dir: 目标目录，缺省为 :data:`learnpath.paths.DATA_DIR`。

    Returns:
        实际写出的文件路径列表（顺序与 ``paths.DATASET_FILES`` 一致）。
    """
    directory = ensure_dir(resolve_data_dir(data_dir))
    dataset.validate()

    written: list[Path] = []
    written.append(
        _write_json(
            directory / KNOWLEDGE_POINTS_FILE,
            [kp.to_dict() for kp in dataset.knowledge_points.values()],
        )
    )
    written.append(
        _write_json(directory / KP_EDGES_FILE, [edge.to_dict() for edge in dataset.kp_edges])
    )
    written.append(
        _write_json(
            directory / COURSES_FILE,
            [course.to_dict() for course in dataset.courses.values()],
        )
    )
    written.append(
        _write_json(
            directory / GOALS_FILE,
            [goal.to_dict() for goal in dataset.goals.values()],
        )
    )
    written.append(
        _write_json(
            directory / LEARNERS_FILE,
            [learner.to_dict() for learner in dataset.learners.values()],
        )
    )
    written.append(
        _write_json(
            directory / INTERACTIONS_FILE,
            [interaction.to_dict() for interaction in dataset.interactions],
        )
    )
    written.append(_write_json(directory / DATASET_META_FILE, dict(dataset.meta)))
    return written


# ============================================================
# 存在性与「加载或生成」
# ============================================================


def dataset_exists(data_dir: Path | str | None = None) -> bool:
    """数据目录中是否已存在一整套可加载的数据集文件。"""
    directory = resolve_data_dir(data_dir)
    return all((directory / name).is_file() for name in DATASET_FILES)


def load_or_build(
    data_dir: Path | str | None = None,
    scale: DatasetScale = DEFAULT_SCALE,
) -> Dataset:
    """存在则加载，否则生成并落盘（下游模块的统一入口）。"""
    directory = resolve_data_dir(data_dir)
    if dataset_exists(directory):
        return load_dataset(directory)

    from learnpath.data.generator import generate_dataset  # 延迟导入，避免重依赖

    dataset = generate_dataset(scale)
    save_dataset(dataset, directory)
    return dataset


__all__ = [
    "load_dataset",
    "save_dataset",
    "dataset_exists",
    "load_or_build",
    "resolve_data_dir",
    "DATASET_FILES",
]
