"""项目级路径约定。

所有模块统一通过本文件定位仓库内资源，禁止在业务代码里硬编码相对路径，
从而保证「从任意工作目录运行」「作为包安装后运行」「pytest 从根目录运行」都对。
"""

from __future__ import annotations

from pathlib import Path

# learnpath/paths.py -> learnpath/ -> 仓库根目录
PACKAGE_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = PACKAGE_DIR.parent

# ---- 资源目录 ----
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
WEB_DIR: Path = ROOT_DIR / "web"
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"
TESTS_DIR: Path = ROOT_DIR / "tests"

# ---- 交付物目录（报告 / PPT，均纳入版本控制）----
DOCS_DIR: Path = ROOT_DIR / "docs"
REPORT_DIR: Path = DOCS_DIR / "report"
SLIDES_DIR: Path = DOCS_DIR / "slides"

# ---- 数据集内各文件名（data/ 目录下）----
KNOWLEDGE_POINTS_FILE = "knowledge_points.json"
KP_EDGES_FILE = "kp_edges.json"
COURSES_FILE = "courses.json"
LEARNERS_FILE = "learners.json"
GOALS_FILE = "goals.json"
INTERACTIONS_FILE = "interactions.json"
DATASET_META_FILE = "dataset_meta.json"

DATASET_FILES: tuple[str, ...] = (
    KNOWLEDGE_POINTS_FILE,
    KP_EDGES_FILE,
    COURSES_FILE,
    LEARNERS_FILE,
    GOALS_FILE,
    INTERACTIONS_FILE,
    DATASET_META_FILE,
)

# 训练产物（二进制模型不入库，指标 / 模型卡入库）
COMPLETION_MODEL_FILE = "completion_model.joblib"
MASTERY_MODEL_FILE = "mastery_model.joblib"
MODEL_METRICS_FILE = "model_metrics.json"


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回该目录。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
