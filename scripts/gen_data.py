"""数据集生成命令行工具。

用法::

    python scripts/gen_data.py --scale full --out data --force
    python scripts/gen_data.py --scale tiny --out data_tiny
    python scripts/gen_data.py --scale full --out data --seed 20261029

行为：
    1. 先校验参数（规模档位 / 输出目录 / 是否覆盖已有数据）；
    2. 调 :func:`learnpath.data.generator.generate_dataset` 生成数据集；
    3. 用 :func:`learnpath.data.loader.save_dataset` 按实体拆分成 JSON 落盘；
    4. 打印人类可读的统计摘要（实体数量、图谱统计、完成率、目标闭包规模等）。

脚本只依赖 ``learnpath`` 包，不需要安装额外依赖；从仓库任意目录运行都可以。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

# 允许直接 `python scripts/gen_data.py` 运行：把仓库根目录塞进 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learnpath.config import (  # noqa: E402  （必须在 sys.path 调整之后导入）
    DEFAULT_SCALE,
    DOMAINS,
    MIN_LEVEL,
    TINY_SCALE,
    DatasetScale,
)
from learnpath.data.generator import generate_dataset  # noqa: E402
from learnpath.data.loader import dataset_exists, save_dataset  # noqa: E402
from learnpath.data.schema import Dataset  # noqa: E402
from learnpath.knowledge.graph import KnowledgeGraph  # noqa: E402

_SCALES: dict[str, DatasetScale] = {"full": DEFAULT_SCALE, "tiny": TINY_SCALE}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 LearnPath 在线学习数据集（知识点图谱 / 课程库 / 学习者 / 交互日志）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale",
        choices=sorted(_SCALES),
        default="full",
        help="规模档位：full=默认完整规模，tiny=极小规模（用于冒烟测试）",
    )
    parser.add_argument("--out", default="data", help="输出目录（相对路径按仓库根目录解析）")
    parser.add_argument(
        "--force",
        action="store_true",
        help="输出目录已有数据集时仍然覆盖重写",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆盖规模档位自带的随机种子（默认沿用 config 中的 seed）",
    )
    return parser.parse_args(argv)


def _fmt_table(rows: Sequence[tuple[str, str]]) -> str:
    width = max((len(key) for key, _ in rows), default=0) + 2
    return "\n".join(f"  {key.ljust(width)}{value}" for key, value in rows)


def summarize(dataset: Dataset, out_dir: Path, elapsed: float, files: list[Path]) -> str:
    """构造人类可读的统计摘要。"""
    graph = KnowledgeGraph.from_dataset(dataset)
    graph.validate()
    graph_stats = graph.stats()
    stats = dataset.stats()

    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("LearnPath 数据集生成完成")
    lines.append("=" * 68)

    lines.append("[规模]")
    lines.append(
        _fmt_table(
            [
                ("知识点", str(stats["n_knowledge_points"])),
                ("先修边", str(stats["n_kp_edges"])),
                ("课程", str(stats["n_courses"])),
                ("学习者", str(stats["n_learners"])),
                ("职业目标", str(stats["n_goals"])),
                ("交互记录", str(stats["n_interactions"])),
                ("已完成交互", str(stats["n_completed"])),
                ("完成率", f"{stats['completion_rate']:.4f}"),
            ]
        )
    )

    lines.append("[知识图谱]")
    lines.append(
        _fmt_table(
            [
                ("节点 / 边", f"{graph_stats.n_nodes} / {graph_stats.n_edges}"),
                ("起点 / 叶子", f"{graph_stats.n_roots} / {graph_stats.n_leaves}"),
                ("最大先修深度", str(graph_stats.max_depth)),
                ("平均出度", f"{graph_stats.avg_out_degree:.3f}"),
            ]
        )
    )
    isolated = [
        kp_id
        for kp_id in graph.node_ids
        if not graph.predecessors(kp_id) and not graph.successors(kp_id)
    ]
    lines.append(f"  孤立知识点      {len(isolated)}")

    lines.append("[各领域分布]")
    lines.append(f"  {'领域'.ljust(16)}{'知识点':>8}{'课程':>8}{'层级':>8}{'起点':>8}")
    for domain in DOMAINS:
        domain_kps = [kp for kp in dataset.knowledge_points.values() if kp.domain == domain]
        domain_courses = [c for c in dataset.courses.values() if c.domain == domain]
        depths = {graph.depth(kp.kp_id) for kp in domain_kps} if domain_kps else {0}
        roots = [k for k in graph.roots() if graph.domain_of(k) == domain]
        lines.append(
            f"  {domain.ljust(16)}{len(domain_kps):>8}{len(domain_courses):>8}"
            f"{len(depths):>8}{len(roots):>8}"
        )

    lines.append("[课程库]")
    modality_counts: dict[str, int] = {}
    level_counts: dict[int, int] = {}
    for course in dataset.courses.values():
        modality_counts[course.modality] = modality_counts.get(course.modality, 0) + 1
        level_counts[course.level] = level_counts.get(course.level, 0) + 1
    durations = [course.duration_hours for course in dataset.courses.values()]
    ratings = [course.rating for course in dataset.courses.values()]
    covered = {kp_id for course in dataset.courses.values() for kp_id in course.kp_ids}
    lines.append(
        _fmt_table(
            [
                (
                    "模态分布",
                    "  ".join(f"{k}={v}" for k, v in sorted(modality_counts.items())),
                ),
                (
                    "等级分布",
                    "  ".join(
                        f"L{k}={level_counts.get(k, 0)}"
                        for k in range(MIN_LEVEL, MIN_LEVEL + 5)
                    ),
                ),
                ("时长范围", f"{min(durations):.1f} ~ {max(durations):.1f} 小时"),
                (
                    "评分范围",
                    f"{min(ratings):.2f} ~ {max(ratings):.2f}"
                    f"（均值 {sum(ratings) / len(ratings):.2f}）",
                ),
                ("知识点覆盖率", f"{len(covered)}/{len(dataset.knowledge_points)}"),
            ]
        )
    )

    lines.append("[职业目标闭包]")
    lines.append(f"  {'目标'.ljust(10)}{'目标知识点':>12}{'先修闭包':>10}{'比率':>8}  名称")
    for goal in dataset.goals.values():
        closure = graph.prerequisite_closure(goal.target_kp_ids)
        ratio = len(closure) / max(1, len(goal.target_kp_ids))
        lines.append(
            f"  {goal.goal_id.ljust(10)}{len(goal.target_kp_ids):>12}"
            f"{len(closure):>10}{ratio:>8.2f}  {goal.name}"
        )

    lines.append("[学习者]")
    abilities = [learner.ability for learner in dataset.learners.values()]
    mastery_counts = [len(learner.initial_mastery) for learner in dataset.learners.values()]
    goal_counts: dict[str, int] = {}
    for learner in dataset.learners.values():
        goal_counts[learner.goal_id] = goal_counts.get(learner.goal_id, 0) + 1
    budgets = [learner.time_budget_hours_per_week for learner in dataset.learners.values()]
    lines.append(
        _fmt_table(
            [
                (
                    "能力",
                    f"{min(abilities):.2f} ~ {max(abilities):.2f}"
                    f"（均值 {sum(abilities) / len(abilities):.3f}）",
                ),
                (
                    "初始掌握点",
                    f"{min(mastery_counts)} ~ {max(mastery_counts)}"
                    f"（均值 {sum(mastery_counts) / len(mastery_counts):.2f}）",
                ),
                (
                    "每周预算",
                    f"{min(budgets):.0f} ~ {max(budgets):.0f} 小时",
                ),
                (
                    "目标分布",
                    "  ".join(f"{k}={v}" for k, v in sorted(goal_counts.items())),
                ),
            ]
        )
    )

    lines.append("[交互日志]")
    quizzes = [i.quiz_score for i in dataset.interactions if i.quiz_score is not None]
    rated = sum(1 for i in dataset.interactions if i.rating is not None)
    timestamps = [i.timestamp for i in dataset.interactions]
    lines.append(
        _fmt_table(
            [
                (
                    "完成率",
                    f"{stats['completion_rate']:.4f}"
                    f"（{stats['n_completed']}/{stats['n_interactions']}）",
                ),
                ("测验覆盖", f"{len(quizzes)} 条，均值 {sum(quizzes) / max(1, len(quizzes)):.3f}"),
                ("评价覆盖", f"{rated} 条（{rated / max(1, len(timestamps)):.1%}）"),
                (
                    "时间范围",
                    f"{min(timestamps)} ~ {max(timestamps)}" if timestamps else "无",
                ),
            ]
        )
    )

    model = dataset.meta.get("completion_model", {})
    if model.get("auc_test") is not None:
        lines.append("[完成率模型可学习性]")
        lines.append(
            _fmt_table(
                [
                    ("任务", str(model.get("task"))),
                    ("模型", str(model.get("model"))),
                    ("训练 / 测试", f"{model.get('n_train')} / {model.get('n_test')}"),
                    ("正例比例", f"{model.get('positive_rate')}"),
                    ("AUC（测试）", f"{model.get('auc_test')}"),
                    ("AUC（训练）", f"{model.get('auc_train')}"),
                ]
            )
        )
        importance = model.get("feature_importance") or {}
        if importance:
            top = sorted(importance.items(), key=lambda item: -item[1])[:4]
            lines.append("  主要特征      " + "  ".join(f"{k}={v:.3f}" for k, v in top))

    lines.append("[输出文件]")
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        lines.append(f"  {path.name.ljust(24)}{size / 1024:>10.1f} KB")
    lines.append(f"  {'合计'.ljust(24)}{total_bytes / 1024 / 1024:>10.2f} MB")
    lines.append(f"  输出目录：{out_dir}")
    lines.append(f"  耗时：{elapsed:.2f}s")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口，返回进程退出码。"""
    args = _parse_args(argv)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir = out_dir.resolve()

    if dataset_exists(out_dir) and not args.force:
        print(
            f"[跳过] {out_dir} 已存在数据集文件；如需覆盖请加 --force",
            file=sys.stderr,
        )
        return 1

    scale = _SCALES[args.scale]
    if args.seed is not None:
        scale = DatasetScale(
            n_knowledge_points=scale.n_knowledge_points,
            n_courses=scale.n_courses,
            n_learners=scale.n_learners,
            n_interactions=scale.n_interactions,
            seed=int(args.seed),
        )

    print(f"[1/3] 正在生成数据集（scale={args.scale}, seed={scale.seed}）...")
    started = time.perf_counter()
    dataset = generate_dataset(scale)
    elapsed_generate = time.perf_counter() - started

    print(f"[2/3] 正在写入 {out_dir} ...")
    files = save_dataset(dataset, out_dir)

    print("[3/3] 统计摘要：")
    print(summarize(dataset, out_dir, elapsed_generate, files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
