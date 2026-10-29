"""训练学习效果预测模型，并把指标写入 ``models/model_metrics.json``。

用法（仓库根目录）::

    .\\.venv\\Scripts\\python.exe scripts\\train_models.py
    .\\.venv\\Scripts\\python.exe scripts\\train_models.py --max-rows 8000 --quiet

流程
----
1. 用 :func:`learnpath.data.loader.load_dataset` 读取 ``data/`` 下的完整数据集
   （若不存在则回退到 :func:`learnpath.data.fixtures.build_tiny_dataset` 并给出警告）；
2. 构建知识图谱；
3. 调 :meth:`learnpath.prediction.effect.EffectPredictor.fit` 完成
   「完成概率分类」与「掌握度增益回归」两个任务的双模型对比 + 5 折交叉验证；
4. 二进制品写 ``models/completion_model.joblib`` / ``models/mastery_model.joblib``
   （被 ``.gitignore`` 忽略，可复现），指标写 ``models/model_metrics.json``（入库）；
5. 在控制台打印对比表与运行耗时。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许直接 `python scripts/train_models.py` 运行（把仓库根目录加入 sys.path）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，直接打印中文 / 上标（如 R²）会抛 UnicodeEncodeError；
# 这里统一切到 UTF-8 并用 replace 兜底，保证脚本在任何代码页下都能跑完。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover - 非标准流
        pass

from learnpath.data.loader import dataset_exists, load_dataset  # noqa: E402
from learnpath.data.fixtures import build_tiny_dataset  # noqa: E402
from learnpath.knowledge.graph import KnowledgeGraph  # noqa: E402
from learnpath.paths import MODEL_METRICS_FILE, MODELS_DIR  # noqa: E402
from learnpath.prediction.effect import EffectPredictor  # noqa: E402


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    """打印一个对齐的 ASCII 表格。"""
    widths = [len(h) for h in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "+".join("-" * (w + 2) for w in widths)
    print(f"\n{title}")
    print(line)
    print("|".join(f" {h.ljust(widths[i])} " for i, h in enumerate(header)))
    print(line)
    for row in rows:
        print("|".join(f" {cell.ljust(widths[i])} " for i, cell in enumerate(row)))
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 LearnPath 学习效果预测模型")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="训练样本上限（超出按固定步长确定性下采样），默认 20000",
    )
    parser.add_argument("--quiet", action="store_true", help="只打印最终指标，不打印表格")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据集目录（缺省 data/）",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    if dataset_exists(args.data_dir):
        dataset = load_dataset(args.data_dir)
        source = "full"
    else:  # pragma: no cover - 正常仓库里 data/ 一定存在
        dataset = build_tiny_dataset()
        source = "tiny-fixture"
        print(
            "[警告] 未找到完整数据集，已回退到 tiny fixture（仅用于自检，指标不可用于报告）",
            file=sys.stderr,
        )

    graph = KnowledgeGraph.from_dataset(dataset)
    graph.validate()
    stats = dataset.stats()

    print("=" * 78)
    print("LearnPath · 学习效果预测模型训练")
    print("=" * 78)
    print(f"数据源        : {source}")
    print(
        "数据规模      : "
        f"{stats['n_knowledge_points']} 知识点 / {stats['n_courses']} 课程 / "
        f"{stats['n_learners']} 学习者 / {stats['n_interactions']} 交互"
    )
    print(f"整体完成率    : {stats['completion_rate']:.4f}")

    predictor = EffectPredictor()
    metrics = predictor.fit(dataset, graph, max_rows=args.max_rows)
    report = predictor.report

    if not args.quiet:
        effect = report.get("effect_model", {})
        clf_rows = [
            [
                str(row.get("name", "")),
                f"{float(row.get('auc', 0)):.4f}",
                f"{float(row.get('accuracy', 0)):.4f}",
                f"{float(row.get('f1', 0)):.4f}",
                f"{float(row.get('cv_mean', 0)):.4f} ± {float(row.get('cv_std', 0)):.4f}",
            ]
            for row in effect.get("models", [])
        ]
        _print_table(
            "任务 A · 完成概率（二分类，5 折分层交叉验证，折外指标）",
            ["模型", "AUC", "Accuracy", "F1", "CV AUC 均值 ± 标准差"],
            clf_rows,
        )

        gain = report.get("gain_model", {})
        gain_rows = [
            [
                str(row.get("name", "")),
                f"{float(row.get('mae', 0)):.4f}",
                f"{float(row.get('rmse', 0)):.4f}",
                f"{float(row.get('r2', 0)):.4f}",
            ]
            for row in gain.get("models", [])
        ]
        _print_table(
            "任务 B · 掌握度达成水平回归（标签 = quiz_score，5 折交叉验证，折外指标）",
            ["模型", "MAE", "RMSE", "R²"],
            gain_rows,
        )

        feature_rows = [
            [str(item.get("name", "")), f"{float(item.get('importance', 0)):.4f}"]
            for item in effect.get("features", [])
        ]
        _print_table(
            "特征重要性（最佳分类器）",
            ["特征", "重要度"],
            feature_rows,
        )

    print("\n最佳分类器 :", predictor.completion_model_name)
    print("最佳回归器 :", predictor.gain_model_name or "（样本不足，推理期退化为解析式）")
    print("完成概率 AUC : %.4f" % metrics.get("auc", 0.0))
    print("增益回归 R²  : %.4f" % metrics.get("r2", 0.0))
    print("指标文件     :", (MODELS_DIR / MODEL_METRICS_FILE))
    print("总耗时       : %.1f s" % (time.perf_counter() - started))

    if not args.quiet:
        print("\n---- models/model_metrics.json 摘要 ----")
        print(json.dumps(report, ensure_ascii=False, indent=2)[:2600])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
