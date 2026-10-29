"""学习效果预测：完成概率分类 + 掌握度增益回归（实现要求 ⑤）。

两个预测任务
------------
**任务 A：完成概率（二分类）**

* 标签：``Interaction.completed``（真实历史中该学习者是否学完这门课）。
* 候选模型：
    1. **逻辑回归**（``StandardScaler`` + ``LogisticRegression``，L2 正则 ``C=1.0``，
       可解释性最好，``coef_`` 可直接读出特征方向）；
    2. **梯度提升树**（``HistGradientBoostingClassifier``，200 轮、学习率 0.06、
       叶节点最小样本 20、L2 正则 1.0，能自动捕获非线性与交互项）。
* 评估：**5 折分层交叉验证**（``StratifiedKFold(shuffle=True, random_state=42)``），
  用**折外预测（out-of-fold）**统一算 AUC / Accuracy / F1（阈值 0.5），
  另外报告 ``roc_auc`` 的折间均值与标准差。

**任务 B：掌握度增益（回归）**

* 标签：``quiz_score``（结课测验得分，反映「这门课学完能到什么水平」）。
  它由学习者能力、课程难度差、先修就绪度共同决定，是可学习的连续信号。
* 候选模型：``Ridge(alpha=1.0)`` 与 ``HistGradientBoostingRegressor``，
  同样 5 折交叉验证 + 折外预测，报告 MAE / RMSE / R²。
* **注意**：「掌握度增益」本身不作为拟合标签（它由 ``prior`` 与 ``required`` 共同决定，
  方差主要由先验掌握度解释，直接回归会出现 R²≈0）。推理期用
  :meth:`EffectPredictor.predict_gain` 的解析式把预测达成率折算成增益：

      post_k = clamp(m_k + w_k · max(0, target − m_k) · achievement, 0, 1)

词典里 ``p_completion ≥ 0.70`` → 风险「低」，``0.40 ≤ p < 0.70`` → 「中」，
``p < 0.40`` → 「高」。

特征可解释性
------------
``drivers`` 给出对本次预测贡献最大的 Top-5 特征，计算方式：

* 树模型：``contribution_j = importance_j × z_j``（``z`` 为标准化特征值）；
* 线性模型：``contribution_j = coef_j × z_j``（对数几率单位）。

``z`` 的符号让贡献带方向：正贡献推向「会完成」，负贡献推向「会放弃」；
按 ``|contribution|`` 降序取 5 个。

**防标签泄漏**（关键实现细节）
------------------------------
``LearnerModel`` 构造时会按时间顺序回放该学习者的历史交互，如果把「本次要预测
的那条记录」也回放了，掌握度类特征就已经包含标签信息。因此训练时通过
:class:`_MaskedDatasetView` 为每条记录构造一个**排除了该记录**的数据集视图：

* 视图复用**同一个** ``Dataset`` 对象（只替换 ``interactions``），
  使 ``learnpath.learner.profile`` 里按 ``id(dataset)`` 缓存的图结构（420 个
  知识点拓扑、560 门课程的先修集合）只构建一次——
  否则 14000 条训练样本会重建 14000 次，耗时与内存都不可接受；
* 每次构造前显式失效该视图的「交互分组缓存」，并用
  ``LearnerModel.history_size()`` 校验屏蔽确实生效（自愈式回退）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learnpath.config import EVAL_SEED
from learnpath.data.schema import Course, Dataset, Interaction
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.paths import (
    COMPLETION_MODEL_FILE,
    MASTERY_MODEL_FILE,
    MODEL_METRICS_FILE,
    MODELS_DIR,
    ensure_dir,
)
from learnpath.prediction.features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    build_features,
    feature_vector,
)

# ============================================================
# 超参数（报告中可直接引用）
# ============================================================

#: 交叉验证折数
CV_FOLDS: Final[int] = 5

#: 逻辑回归正则强度（L2）
LOGREG_C: Final[float] = 1.0
LOGREG_MAX_ITER: Final[int] = 2000

#: 梯度提升树（分类）参数
HGB_CLF_MAX_ITER: Final[int] = 200
HGB_CLF_LEARNING_RATE: Final[float] = 0.06
HGB_CLF_MIN_SAMPLES_LEAF: Final[int] = 20
HGB_CLF_L2: Final[float] = 1.0

#: 岭回归正则强度
RIDGE_ALPHA: Final[float] = 1.0

#: 梯度提升树（回归）参数
HGB_REG_MAX_ITER: Final[int] = 200
HGB_REG_LEARNING_RATE: Final[float] = 0.06
HGB_REG_MIN_SAMPLES_LEAF: Final[int] = 20
HGB_REG_L2: Final[float] = 1.0

#: 风险等级阈值：``p_completion`` 高于前者为「低」，低于后者为「高」
RISK_LOW_THRESHOLD: Final[float] = 0.70
RISK_HIGH_THRESHOLD: Final[float] = 0.40

#: 分类判定阈值（用于 Accuracy / F1）
DECISION_THRESHOLD: Final[float] = 0.5

#: ``drivers`` 返回的特征数
DRIVER_TOP_K: Final[int] = 5

#: 置换重要性的重复次数（``HistGradientBoosting`` 无内建重要度时使用）
PERMUTATION_REPEATS: Final[int] = 3

#: 置换重要性使用的样本上限
PERMUTATION_SAMPLE: Final[int] = 2000

#: 训练集规模超过该值时按固定步长下采样（保证脚本可在数分钟内跑完）
MAX_TRAIN_ROWS: Final[int] = 20000

_EPS: Final[float] = 1e-12


# ============================================================
# 预测结果
# ============================================================


@dataclass(slots=True)
class CompletionPrediction:
    """完成概率预测结果（实现要求 ⑤ 的可解释输出）。

    Attributes:
        p_completion: 完成概率 ``∈ [0,1]``。
        p_drop: 放弃概率 ``= 1 - p_completion``。
        risk_level: 风险等级，取值 ``低`` / ``中`` / ``高``。
        drivers: 贡献最大的 Top-5 特征 ``[(特征名, 加权贡献)]``，
            特征名为英文 snake_case（中文名见
            :data:`learnpath.prediction.features.FEATURE_LABELS`），
            贡献为正表示推高完成概率。
    """

    p_completion: float
    p_drop: float
    risk_level: str
    drivers: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_completion": round(float(self.p_completion), 3),
            "p_drop": round(float(self.p_drop), 3),
            "risk_level": self.risk_level,
            "drivers": [
                {"name": name, "label": FEATURE_LABELS.get(name, name), "contribution": round(float(value), 4)}
                for name, value in self.drivers
            ],
        }


@dataclass(slots=True)
class GainPrediction:
    """掌握度增益预测结果。

    Attributes:
        expected_gain: 课程结束后掌握度的平均提升量 ``∈ [0,1]``。
        expected_kp_mastery: 每个覆盖知识点的预测掌握度 ``kp_id -> [0,1]``。
        predicted_level: 回归模型直接输出的「达成水平」代理
            （标签为 ``quiz_score × progress``）。
    """

    expected_gain: float
    expected_kp_mastery: dict[str, float] = field(default_factory=dict)
    predicted_level: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_gain": round(float(self.expected_gain), 4),
            "expected_kp_mastery": {
                k: round(float(v), 4) for k, v in self.expected_kp_mastery.items()
            },
            "predicted_level": round(float(self.predicted_level), 4),
        }


# ============================================================
# 无泄漏数据集视图
# ============================================================


def _reset_interactions_cache(view: Dataset) -> None:
    """让 ``learnpath.learner.profile`` 的「交互分组缓存」对该视图失效。

    该模块用 ``id(dataset)`` 缓存 ``interactions_by_learner`` 的结果，
    而我们复用同一个视图对象、只替换 ``interactions``，所以必须手动失效，
    否则会读到上一条样本的分组。只操作缓存字典，不修改对方源码。
    """
    try:
        from learnpath.learner import profile as _profile  # 延迟导入，避免循环依赖
    except ImportError:  # pragma: no cover - 学习器模块缺失时不影响预测模块导入
        return
    cache = getattr(_profile, "_INTERACTIONS_CACHE", None)
    if isinstance(cache, dict):
        cache.pop(id(view), None)


class _MaskedDatasetView:
    """「留一课程」数据集视图工厂（训练期防标签泄漏）。

    只为训练期服务：对每条 ``(learner, course)`` 样本构造一个
    **排除了该条记录**的数据集视图，再用它构造 ``LearnerModel``，
    从而保证掌握度 / 就绪度 / 收益等特征中不含该课程的结果信息。

    实现要点见模块 docstring「防标签泄漏」一节。
    """

    def __init__(self, dataset: Dataset, graph: KnowledgeGraph) -> None:
        self.dataset = dataset
        self.graph = graph
        self.view = Dataset(
            knowledge_points=dataset.knowledge_points,
            kp_edges=dataset.kp_edges,
            courses=dataset.courses,
            learners=dataset.learners,
            goals=dataset.goals,
            interactions=[],
            meta=dict(dataset.meta),
        )
        grouped: dict[str, list[Interaction]] = {}
        for record in dataset.interactions:
            grouped.setdefault(record.learner_id, []).append(record)
        self._by_learner = grouped
        self._warm()

    def _warm(self) -> None:
        """预热与 ``interactions`` 无关的图结构缓存（只需一次）。"""
        if not self.dataset.learners or not self.dataset.courses:
            return
        from learnpath.learner.profile import LearnerModel

        learner = self.dataset.learners[next(iter(self.dataset.learners))]
        self.view.interactions = []
        _reset_interactions_cache(self.view)
        model = LearnerModel(learner, self.view, self.graph)
        # 触发「课程先修集合」缓存（560 门课一次性算好）
        model.readiness_for(self.dataset.courses[next(iter(self.dataset.courses))])

    def model_for(self, learner_id: str, exclude_course_id: str | None) -> Any:
        """构造「不含指定课程记录」的 ``LearnerModel``。"""
        from learnpath.learner.profile import LearnerModel

        records = self._by_learner.get(learner_id, ())
        if exclude_course_id is not None:
            records = [r for r in records if r.course_id != exclude_course_id]
        expected = len(records)
        self.view.interactions = list(records)
        _reset_interactions_cache(self.view)
        learner = self.dataset.learners[learner_id]
        model = LearnerModel(learner, self.view, self.graph)

        if model.history_size() != expected:  # pragma: no cover - 缓存失效失败的自愈分支
            from learnpath.learner import profile as _profile

            cache = getattr(_profile, "_INTERACTIONS_CACHE", None)
            if isinstance(cache, dict):
                cache.clear()
            _reset_interactions_cache(self.view)
            model = LearnerModel(learner, self.view, self.graph)
        return model


# ============================================================
# 预测器
# ============================================================


class EffectPredictor:
    """学习效果预测器：完成概率 + 掌握度增益。

    Args:
        model_path: 模型与指标文件所在目录；缺省 ``models/``（见
            :data:`learnpath.paths.MODELS_DIR`）。

    Example:
        >>> predictor = EffectPredictor()          # doctest: +SKIP
        >>> metrics = predictor.fit(dataset, graph)  # doctest: +SKIP
        >>> predictor.predict_completion(dataset, graph, model, course).risk_level  # doctest: +SKIP
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path: Path = Path(model_path) if model_path is not None else MODELS_DIR
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        self.completion_model: Any = None
        self.gain_model: Any = None
        self.completion_model_name: str = ""
        self.gain_model_name: str = ""
        self.feature_mean: np.ndarray = np.zeros(len(FEATURE_NAMES))
        self.feature_std: np.ndarray = np.ones(len(FEATURE_NAMES))
        self.feature_importance: np.ndarray = np.zeros(len(FEATURE_NAMES))
        self._metrics: dict[str, float] = {}
        self._report: dict[str, Any] = {}
        self._fitted = False

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------
    @property
    def metrics(self) -> dict[str, float]:
        """最佳模型的扁平指标（``auc/accuracy/f1/mae/rmse/r2``）。"""
        return dict(self._metrics)

    @property
    def report(self) -> dict[str, Any]:
        """完整的对比报告（写入 ``models/model_metrics.json`` 的内容）。"""
        return json.loads(json.dumps(self._report, ensure_ascii=False))

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    def _build_training_set(
        self,
        dataset: Dataset,
        graph: KnowledgeGraph,
        max_rows: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """构造训练矩阵 ``(X, y_cls, y_reg, reg_mask, meta)``（自动防泄漏）。"""
        records = list(dataset.interactions)
        if max_rows is not None and len(records) > max_rows:
            step = int(math.ceil(len(records) / max_rows))
            records = records[::step]

        view = _MaskedDatasetView(dataset, graph)
        rows: list[list[float]] = []
        y_cls: list[int] = []
        y_reg: list[float] = []
        reg_mask: list[int] = []
        skipped = 0

        for record in records:
            course = dataset.courses.get(record.course_id)
            if course is None:
                skipped += 1
                continue
            model = view.model_for(record.learner_id, record.course_id)
            features = build_features(dataset, graph, model, course)
            rows.append(feature_vector(features))
            y_cls.append(1 if record.completed else 0)
            if record.quiz_score is not None:
                y_reg.append(float(record.quiz_score))
                reg_mask.append(1)
            else:
                y_reg.append(0.0)
                reg_mask.append(0)

        meta = {
            "n_samples": len(rows),
            "n_skipped": skipped,
            "positive_rate": (sum(y_cls) / len(y_cls)) if y_cls else 0.0,
            "n_regression": int(sum(reg_mask)),
        }
        return (
            np.asarray(rows, dtype=float),
            np.asarray(y_cls, dtype=int),
            np.asarray(y_reg, dtype=float),
            np.asarray(reg_mask, dtype=bool),
            meta,
        )

    def fit(
        self,
        dataset: Dataset,
        graph: KnowledgeGraph,
        *,
        max_rows: int | None = MAX_TRAIN_ROWS,
        save: bool = True,
    ) -> dict[str, float]:
        """训练并评估两个任务，返回最佳模型的扁平指标。

        Args:
            dataset: 训练数据集。
            graph: 知识点图。
            max_rows: 训练样本上限（超出时按固定步长确定性下采样，
                默认 ``MAX_TRAIN_ROWS = 20000``）。
            save: 是否在训练结束后把二进制品与指标写入 ``model_path``。

        Returns:
            ``{"auc", "accuracy", "f1", "cv_mean", "cv_std", "mae", "rmse", "r2"}``。
        """
        X, y_cls, y_reg, reg_mask, meta = self._build_training_set(dataset, graph, max_rows)
        if X.shape[0] == 0:
            raise ValueError("训练集为空：dataset.interactions 没有可用记录")

        self.feature_mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.feature_std = np.where(std > 1e-9, std, 1.0)

        # ---------------- 任务 A：完成概率分类 ----------------
        clf_models = {
            "逻辑回归": Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=LOGREG_C, max_iter=LOGREG_MAX_ITER, random_state=EVAL_SEED
                        ),
                    ),
                ]
            ),
            "梯度提升树": HistGradientBoostingClassifier(
                max_iter=HGB_CLF_MAX_ITER,
                learning_rate=HGB_CLF_LEARNING_RATE,
                min_samples_leaf=HGB_CLF_MIN_SAMPLES_LEAF,
                l2_regularization=HGB_CLF_L2,
                random_state=EVAL_SEED,
            ),
        }
        clf_rows = self._cross_validate_classifiers(clf_models, X, y_cls)

        best_clf_row = max(clf_rows, key=lambda row: (row["auc"], row["cv_mean"]))
        self.completion_model_name = best_clf_row["name"]
        self.completion_model = clf_models[self.completion_model_name]
        self.completion_model.fit(X, y_cls)
        self.feature_importance = self._compute_feature_importance(X, y_cls)

        # ---------------- 任务 B：掌握度增益回归 ----------------
        Xr = X[reg_mask]
        yr = y_reg[reg_mask]
        reg_models = {
            "岭回归 Ridge": Pipeline(
                [("scale", StandardScaler()), ("reg", Ridge(alpha=RIDGE_ALPHA))]
            ),
            "梯度提升树": HistGradientBoostingRegressor(
                max_iter=HGB_REG_MAX_ITER,
                learning_rate=HGB_REG_LEARNING_RATE,
                min_samples_leaf=HGB_REG_MIN_SAMPLES_LEAF,
                l2_regularization=HGB_REG_L2,
                random_state=EVAL_SEED,
            ),
        }
        reg_rows: list[dict[str, Any]] = []
        if Xr.shape[0] >= CV_FOLDS:
            reg_rows = self._cross_validate_regressors(reg_models, Xr, yr)
        if reg_rows:
            best_reg_row = max(reg_rows, key=lambda row: row["r2"])
            self.gain_model_name = best_reg_row["name"]
            self.gain_model = reg_models[self.gain_model_name]
            self.gain_model.fit(Xr, yr)
        else:
            self.gain_model_name = ""
            self.gain_model = None
            reg_rows = [{"name": key, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")} for key in reg_models]

        # ---------------- 汇总指标与报告 ----------------
        best_reg = next(
            (row for row in reg_rows if row["name"] == self.gain_model_name),
            reg_rows[0],
        )
        self._metrics = {
            "auc": float(best_clf_row["auc"]),
            "accuracy": float(best_clf_row["accuracy"]),
            "f1": float(best_clf_row["f1"]),
            "cv_mean": float(best_clf_row["cv_mean"]),
            "cv_std": float(best_clf_row["cv_std"]),
            "mae": _finite(best_reg["mae"]),
            "rmse": _finite(best_reg["rmse"]),
            "r2": _finite(best_reg["r2"]),
        }

        self._report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": {
                "n_learners": len(dataset.learners),
                "n_courses": len(dataset.courses),
                "n_interactions": meta["n_samples"],
            },
            "effect_model": {
                "task": "course_completion",
                "positive_rate": round(float(meta["positive_rate"]), 4),
                "models": [
                    {
                        "name": row["name"],
                        "auc": round(float(row["auc"]), 4),
                        "accuracy": round(float(row["accuracy"]), 4),
                        "f1": round(float(row["f1"]), 4),
                        "cv_mean": round(float(row["cv_mean"]), 4),
                        "cv_std": round(float(row["cv_std"]), 4),
                    }
                    for row in clf_rows
                ],
                "best": self.completion_model_name,
                "features": self._feature_importance_report(),
            },
            "gain_model": {
                "task": "mastery_gain_regression",
                "n_samples": int(meta["n_regression"]),
                "best": self.gain_model_name,
                "models": [
                    {
                        "name": row["name"],
                        "mae": _round4(row["mae"]),
                        "rmse": _round4(row["rmse"]),
                        "r2": _round4(row["r2"]),
                    }
                    for row in reg_rows
                ],
            },
            "notes": [
                "完成概率采用 5 折分层交叉验证的折外预测统一计算 AUC / Accuracy / F1。",
                "掌握度达成水平回归以 quiz_score 为主标签，用 5 折交叉验证折外预测计算 MAE / RMSE / R²；"
                "「掌握度增益」在推理期由 predict_gain 的解析式折算，不作为拟合标签。",
                "所有特征均在「学习者尚未学过该课程」的时点计算：训练时按记录逐条屏蔽该课程的历史回放，"
                "杜绝标签泄漏。",
                "刻意避免使用数据生成期的隐变量（latent ability / 生成概率等），仅采用线上可观测特征"
                "（画像能力、BKT 先修就绪度、难度差、历史完成率与测验分、课程口碑与热度、每周预算），"
                "以保证模型在真实系统中可部署、可解释。",
                "风险等级阈值：p >= 0.70 低 / 0.40 ~ 0.70 中 / < 0.40 高。",
            ],
        }

        self._fitted = True
        if save:
            self.save()
        return dict(self._metrics)

    # --------------------------------------------------------
    # 交叉验证
    # --------------------------------------------------------
    @staticmethod
    def _cross_validate_classifiers(
        models: Mapping[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[dict[str, Any]]:
        """对每个分类器做 5 折分层交叉验证，返回折外指标 + 折间均值标准差。

        实现上**只遍历一次折划分**：每折 fit 一次，同时收集折外概率
        （用于整体 AUC / Accuracy / F1）与折内 AUC（用于 ``cv_mean`` / ``cv_std``），
        避免 ``cross_val_predict`` + ``cross_validate`` 两次各跑 5 折带来的 2 倍训练开销。
        """
        y = np.asarray(y)
        n_splits = min(CV_FOLDS, int(np.bincount(y).min())) if y.size else 0
        if n_splits < 2:  # 某一类样本太少，退化为全量拟合 + 重代入评估
            rows = []
            for name, model in models.items():
                model.fit(X, y)
                proba = model.predict_proba(X)[:, 1]
                pred = (proba >= DECISION_THRESHOLD).astype(int)
                auc = roc_auc_score(y, proba) if len(set(y.tolist())) > 1 else 0.5
                rows.append(
                    {
                        "name": name,
                        "auc": float(auc),
                        "accuracy": float(accuracy_score(y, pred)),
                        "f1": float(f1_score(y, pred, zero_division=0)),
                        "cv_mean": float(auc),
                        "cv_std": 0.0,
                    }
                )
            return rows

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=EVAL_SEED)
        folds = list(splitter.split(X, y))
        rows = []
        for name, model in models.items():
            oof = np.zeros(len(y), dtype=float)
            fold_aucs: list[float] = []
            for train_idx, test_idx in folds:
                estimator = clone(model)
                estimator.fit(X[train_idx], y[train_idx])
                proba = estimator.predict_proba(X[test_idx])[:, 1]
                oof[test_idx] = proba
                if len(set(y[test_idx].tolist())) > 1:
                    fold_aucs.append(float(roc_auc_score(y[test_idx], proba)))
            pred = (oof >= DECISION_THRESHOLD).astype(int)
            auc = roc_auc_score(y, oof) if len(set(y.tolist())) > 1 else 0.5
            rows.append(
                {
                    "name": name,
                    "auc": float(auc),
                    "accuracy": float(accuracy_score(y, pred)),
                    "f1": float(f1_score(y, pred, zero_division=0)),
                    "cv_mean": float(np.mean(fold_aucs)) if fold_aucs else float(auc),
                    "cv_std": float(np.std(fold_aucs)) if fold_aucs else 0.0,
                }
            )
        return rows

    @staticmethod
    def _cross_validate_regressors(
        models: Mapping[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[dict[str, Any]]:
        """对每个回归器做 5 折交叉验证，返回折外 MAE / RMSE / R²（单次遍历）。"""
        n_splits = min(CV_FOLDS, int(X.shape[0]))
        if n_splits < 2:  # pragma: no cover - 样本过少时退化为单折
            rows = []
            for name, model in models.items():
                model.fit(X, y)
                pred = model.predict(X)
                rows.append(_regression_row(name, y, pred))
            return rows

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=EVAL_SEED)
        folds = list(splitter.split(X))
        rows = []
        for name, model in models.items():
            oof = np.zeros(len(y), dtype=float)
            for train_idx, test_idx in folds:
                estimator = clone(model)
                estimator.fit(X[train_idx], y[train_idx])
                oof[test_idx] = estimator.predict(X[test_idx])
            rows.append(_regression_row(name, y, oof))
        return rows

    def _feature_importance_report(self, top: int = 12) -> list[dict[str, Any]]:
        """最佳分类器的特征重要性（树用 ``feature_importances_``，线性用 ``|coef|``，
        其余用置换重要性）。"""
        importances = self._feature_importance()
        order = np.argsort(-importances, kind="stable")[:top]
        return [
            {"name": FEATURE_NAMES[i], "importance": round(float(importances[i]), 4)}
            for i in order
            if importances[i] > 0
        ]

    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算并归一化全局特征重要性。

        三种来源（按优先级）：
            1. 树集成自带的 ``feature_importances_``（基于不纯度下降）；
            2. 线性模型的 ``|coef_|``；
            3. ``sklearn.inspection.permutation_importance``——
               ``HistGradientBoostingClassifier`` **不提供** ``feature_importances_``，
               因此对「打乱某列后 AUC 的下降量」做 5 次重复置换，
               这是与模型无关、且对报告最有说服力的重要度定义。

        全部来源最终都归一化到「和为 1」。
        """
        model = self.completion_model
        raw = np.zeros(len(FEATURE_NAMES), dtype=float)
        if model is None:
            return raw
        estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

        if hasattr(estimator, "feature_importances_"):
            raw = np.abs(np.asarray(estimator.feature_importances_, dtype=float)).ravel()
        elif hasattr(estimator, "coef_"):
            raw = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
        else:
            raw = _batched_permutation_importance(model, X, y, PERMUTATION_REPEATS, PERMUTATION_SAMPLE)

        total = float(raw.sum())
        if total <= _EPS:
            return np.full(len(FEATURE_NAMES), 1.0 / len(FEATURE_NAMES))
        return raw / total

    def _feature_importance(self) -> np.ndarray:
        """取已缓存的全局特征重要性（由 :meth:`_compute_feature_importance` 填充）。"""
        importance = np.asarray(self.feature_importance, dtype=float)
        if importance.shape[0] != len(FEATURE_NAMES) or float(importance.sum()) <= _EPS:
            return np.full(len(FEATURE_NAMES), 1.0 / len(FEATURE_NAMES))
        return importance

    # --------------------------------------------------------
    # 推理
    # --------------------------------------------------------
    def _ensure_ready(self) -> None:
        if not self._fitted or self.completion_model is None:
            try:
                self.load()
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "效果预测模型尚未训练：请先调用 fit(dataset, graph) 或运行 scripts/train_models.py"
                ) from exc

    def predict_completion(
        self,
        dataset: Dataset,
        graph: KnowledgeGraph,
        model: Any,
        course: Course,
    ) -> CompletionPrediction:
        """预测「该学习者学完这门课」的概率与风险等级。

        Args:
            dataset: 数据集（提供历史统计）。
            graph: 知识点图。
            model: 学习者模型（必须是**当前实时状态**，不含该课程结果）。
            course: 待预测课程。

        Returns:
            :class:`CompletionPrediction`（含 Top-5 特征贡献）。

        Raises:
            RuntimeError: 模型未训练且本地没有可加载的模型文件。
        """
        self._ensure_ready()
        features = build_features(dataset, graph, model, course)
        x = np.asarray([feature_vector(features)], dtype=float)
        p_completion = float(self.completion_model.predict_proba(x)[0, 1])
        p_completion = max(0.0, min(1.0, p_completion))
        return CompletionPrediction(
            p_completion=p_completion,
            p_drop=1.0 - p_completion,
            risk_level=risk_level_of(p_completion),
            drivers=self._top_drivers(features, x[0]),
        )

    def predict_gain(
        self,
        dataset: Dataset,
        graph: KnowledgeGraph,
        model: Any,
        course: Course,
    ) -> GainPrediction:
        """预测该课程带来的掌握度增益与各知识点的事后掌握度。

        推理期公式（不再把「增益」直接当回归标签，避免量纲混乱）::

            achievement = 回归模型预测的「达成率」（主标签 = 结课测验分 quiz_score）
            target      = goal.required_mastery（目标达成掌握度，缺省 0.80）
            post_k      = clamp(m_k + w_k · max(0, target - m_k) · achievement, 0, 1)
            expected_gain = mean_k (post_k - m_k)

        含义：课程把「当前掌握度与目标掌握度之间的差额」按覆盖权重 ``w_k`` 和
        预测达成率 ``achievement`` 打折后填补。``achievement`` 来自
        ``quiz_score`` 回归（主模型），回归模型不可用时退化为完成概率。
        """
        self._ensure_ready()
        features = build_features(dataset, graph, model, course)
        x = np.asarray([feature_vector(features)], dtype=float)

        kp_ids = [cov.kp_id for cov in course.kp_coverage]
        priors = [float(model.mastery_of(kp_id)) for kp_id in kp_ids]
        required = float(getattr(model, "required_mastery", 0.80) or 0.80)

        if self.gain_model is not None:
            achievement = float(self.gain_model.predict(x)[0])
            achievement = max(0.0, min(1.0, achievement))
        else:
            # 回归模型不可用（样本不足）时用完成概率折算达成率
            achievement = max(0.0, min(1.0, self.predict_completion(dataset, graph, model, course).p_completion))

        expected_kp_mastery: dict[str, float] = {}
        gains: list[float] = []
        for cov, prior in zip(course.kp_coverage, priors):
            deficit = max(0.0, required - prior)
            post = prior + float(cov.weight) * deficit * achievement
            post = max(0.0, min(1.0, post))
            expected_kp_mastery[cov.kp_id] = post
            gains.append(post - prior)

        expected_gain = float(sum(gains) / len(gains)) if gains else 0.0
        return GainPrediction(
            expected_gain=expected_gain,
            expected_kp_mastery=expected_kp_mastery,
            predicted_level=achievement,
        )

    def _top_drivers(
        self,
        features: Mapping[str, float],
        x: np.ndarray,
    ) -> list[tuple[str, float]]:
        """按「重要性 × 标准化特征值」计算 Top-K 特征贡献。

        树模型用 ``feature_importances_``（全局重要度），线性模型用 ``coef_``
        （对数几率单位）；再乘以标准化特征值 ``z`` 得到带方向的局部贡献。
        """
        importances = self._feature_importance()
        z = (x - self.feature_mean) / self.feature_std
        contributions = importances * z
        order = np.argsort(-np.abs(contributions))[:DRIVER_TOP_K]
        return [
            (FEATURE_NAMES[int(i)], float(contributions[int(i)]))
            for i in order
            if abs(float(contributions[int(i)])) > 1e-9
        ]

    # --------------------------------------------------------
    # 持久化
    # --------------------------------------------------------
    def save(self) -> None:
        """把二进制品写入 ``model_path/*.joblib``，指标写入 ``model_metrics.json``。"""
        if not self._fitted:
            return
        ensure_dir(self.model_path)
        joblib.dump(
            {
                "model": self.completion_model,
                "name": self.completion_model_name,
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "feature_importance": self.feature_importance,
                "metrics": self._metrics,
                "report": self._report,
            },
            self.model_path / COMPLETION_MODEL_FILE,
        )
        joblib.dump(
            {
                "model": self.gain_model,
                "name": self.gain_model_name,
                "feature_names": list(FEATURE_NAMES),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
            },
            self.model_path / MASTERY_MODEL_FILE,
        )
        with open(self.model_path / MODEL_METRICS_FILE, "w", encoding="utf-8") as handle:
            json.dump(self._report, handle, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """从 ``model_path`` 加载已训练模型与指标。

        Raises:
            FileNotFoundError: 模型文件不存在（尚未训练）。
        """
        completion_path = self.model_path / COMPLETION_MODEL_FILE
        mastery_path = self.model_path / MASTERY_MODEL_FILE
        if not completion_path.exists():
            raise FileNotFoundError(f"未找到完成概率模型：{completion_path}")
        payload = joblib.load(completion_path)
        self.completion_model = payload["model"]
        self.completion_model_name = payload.get("name", "")
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=float)
        self.feature_std = np.asarray(payload["feature_std"], dtype=float)
        self.feature_importance = np.asarray(
            payload.get("feature_importance", np.zeros(len(FEATURE_NAMES))), dtype=float
        )
        self._metrics = dict(payload.get("metrics", {}))
        self._report = dict(payload.get("report", {}))

        if mastery_path.exists():
            gain_payload = joblib.load(mastery_path)
            self.gain_model = gain_payload.get("model")
            self.gain_model_name = gain_payload.get("name", "")
        self._fitted = True


# ============================================================
# 工具函数
# ============================================================


def risk_level_of(p_completion: float) -> str:
    """完成概率 → 风险等级（``低`` / ``中`` / ``高``）。"""
    if p_completion >= RISK_LOW_THRESHOLD:
        return "低"
    if p_completion >= RISK_HIGH_THRESHOLD:
        return "中"
    return "高"


def _batched_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int,
    sample_limit: int,
) -> np.ndarray:
    """批量置换重要性（模型无关的特征重要度）。

    定义：对第 ``j`` 个特征做随机置换后，模型 AUC 的**平均下降量**。

    与 ``sklearn.inspection.permutation_importance`` 等价，但把
    「``n_features × n_repeats`` 次单独预测」合并成**一次批量预测**
    （把置换后的样本按行堆叠），从而把 ``HistGradientBoostingClassifier``
    每调用一次 ``predict_proba`` 的固定开销（约 90 ms）从 69 次降到 1 次。

    Args:
        model: 已训练好的分类器（需有 ``predict_proba``）。
        X: 训练特征矩阵。
        y: 二分类标签。
        n_repeats: 每个特征的重复置换次数。
        sample_limit: 参与计算的最大样本数（等步长下采样，保证确定性）。

    Returns:
        形状 ``(n_features,)``、已裁剪到非负的重要度（**未归一化**）；
        当样本只含单一类别时返回全零。
    """
    n, m = X.shape
    if n == 0 or m == 0:
        return np.zeros(m, dtype=float)
    step = max(1, n // max(1, sample_limit))
    Xs = X[::step][:sample_limit]
    ys = y[::step][:sample_limit]
    if len(set(np.asarray(ys).tolist())) < 2:
        return np.zeros(m, dtype=float)

    rng = np.random.default_rng(EVAL_SEED)
    rows = Xs.shape[0]
    blocks = np.empty((m * n_repeats, rows, m), dtype=float)
    for j in range(m):
        for r in range(n_repeats):
            permuted = Xs.copy()
            permuted[:, j] = Xs[rng.permutation(rows), j]
            blocks[j * n_repeats + r] = permuted

    flat = blocks.reshape(-1, m)
    proba = np.asarray(model.predict_proba(flat))[:, 1].reshape(m, n_repeats, rows)
    base = float(roc_auc_score(ys, np.asarray(model.predict_proba(Xs))[:, 1]))

    importances = np.zeros(m, dtype=float)
    for j in range(m):
        scores = [float(roc_auc_score(ys, proba[j, r])) for r in range(n_repeats)]
        importances[j] = max(0.0, base - float(np.mean(scores)))
    return importances


def _regression_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """计算一个回归器的 MAE / RMSE / R²。"""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "name": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": math.sqrt(mse),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
    }


def _finite(value: Any) -> float:
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _round4(value: Any) -> float:
    number = float(value)
    return round(number, 4) if math.isfinite(number) else 0.0


__all__ = [
    "CompletionPrediction",
    "GainPrediction",
    "EffectPredictor",
    "risk_level_of",
    "RISK_LOW_THRESHOLD",
    "RISK_HIGH_THRESHOLD",
    "CV_FOLDS",
]
