"""报告与 PPT 共用的图表生成器。

把所有可视化产物统一生成到 ``docs/report/figures/``，供
:mod:`docs.report.build_report` 与 :mod:`docs.slides.build_slides` 引用。

设计原则：
    * 每张图都能从仓库内的真实产物（数据集 / 模型指标 / 评测结果）独立复现；
    * 若某项产物尚不存在，对应函数返回 ``None`` 并打印提示，不影响其余图表；
    * 统一视觉风格（同一配色、同一字号），保证报告与 PPT 观感一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG_DIR = Path(__file__).resolve().parent / "figures"

# ---- 统一配色 ----
C_PRIMARY = "#2F6BFF"
C_ACCENT = "#00C48C"
C_WARN = "#FF7A45"
C_PURPLE = "#7C5CFF"
C_GRAY = "#8A94A6"
C_DARK = "#1B2333"
PALETTE = [C_PRIMARY, C_ACCENT, C_WARN, C_PURPLE, C_GRAY]

# ---- 中文字体 ----
for _font in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun"):
    if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.edgecolor"] = "#CCD3E0"
plt.rcParams["axes.labelcolor"] = C_DARK
plt.rcParams["text.color"] = C_DARK
plt.rcParams["xtick.color"] = "#5A6472"
plt.rcParams["ytick.color"] = "#5A6472"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"


def _out(name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / name


def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [figure] {path.relative_to(ROOT)}")
    return path


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        print(f"  [skip] 缺少 {path.relative_to(ROOT)}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 1. 系统架构图
# ============================================================


def fig_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, color, fill):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.6,rounding_size=1.2",
                linewidth=1.6, edgecolor=color, facecolor=fill,
            )
        )
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h - 9.5, body, ha="center", va="top",
                fontsize=8.3, color="#333B4D", linespacing=1.6)

    ax.text(50, 97, "LearnPath 三层架构", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C_DARK)

    box(6, 70, 88, 20, "表现层  web/",
        "index.html · css · js/views/*.js · vendor/echarts.min.js\n"
        "六个视图：学习者画像 / 智能推荐 / 学习路径 / 学习分析 / 社交学习 / 系统评测",
        C_PRIMARY, "#EEF3FF")

    box(6, 40, 88, 20, "服务层  app/",
        "main.py 应用装配 · services.py 单例容器（懒加载 / 缓存 / 降级）\n"
        "routers/*.py 十个路由模块：只做参数解析与 JSON 序列化",
        C_ACCENT, "#E8FBF4")

    box(6, 4, 88, 30, "引擎层  learnpath/",
        "data 数据契约与生成   knowledge 知识点 DAG   learner BKT 画像\n"
        "recommend 内容+协同+混合   planning 路径规划+NSGA-II 多目标\n"
        "prediction 完成概率与增益   adaptive 自适应重规划\n"
        "social 同伴匹配与学习小组   analytics 仪表板与离线评测",
        C_WARN, "#FFF4EC")

    for y0, y1 in ((70, 60.5), (40, 34.5)):
        ax.add_patch(FancyArrowPatch((50, y0), (50, y1),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.8, color=C_GRAY))
    ax.text(52.5, 65.2, "HTTP / JSON", fontsize=8, color=C_GRAY, va="center")
    ax.text(52.5, 37.2, "Python 调用", fontsize=8, color=C_GRAY, va="center")

    return _finish(fig, _out("fig_architecture.png"))


# ============================================================
# 2. 数据集概览
# ============================================================


def fig_dataset_overview(domains: dict[str, int], levels: dict[int, int]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    names = list(domains)
    values = [domains[n] for n in names]
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    bars = ax.barh(names, values, color=C_PRIMARY, alpha=0.88, height=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                str(value), va="center", fontsize=8.5, color="#5A6472")
    ax.set_title("知识点在各领域的分布", pad=10)
    ax.set_xlabel("知识点数量")
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    ax = axes[1]
    ks = sorted(levels)
    vs = [levels[k] for k in ks]
    bars = ax.bar([f"难度 {k}" for k in ks], vs, color=C_ACCENT, alpha=0.9, width=0.58)
    for bar, v in zip(bars, vs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vs) * 0.015,
                str(v), ha="center", fontsize=8.5, color="#5A6472")
    ax.set_title("课程难度等级分布", pad=10)
    ax.set_ylabel("课程数量")
    ax.set_ylim(0, max(vs) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    return _finish(fig, _out("fig_dataset_overview.png"))


# ============================================================
# 3. 帕累托前沿
# ============================================================


def fig_pareto(points: Sequence[dict[str, Any]], selected_index: int = 0) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    hours = [p["hours"] for p in points]
    gains = [p["gain"] for p in points]

    ax.scatter(hours, gains, s=64, color=C_PRIMARY, alpha=0.75,
               edgecolor="white", linewidth=1.0, zorder=3,
               label="4 目标帕累托前沿（投影）")

    front = sorted(range(len(points)), key=lambda i: hours[i])
    fx = [hours[i] for i in front]
    fy = [gains[i] for i in front]
    ax.plot(fx, fy, linestyle="--", color=C_GRAY, linewidth=1.2,
            alpha=0.75, zorder=2, label="投影包络")

    flagged = [i for i, p in enumerate(points) if p.get("is_selected")]
    index = flagged[0] if flagged else selected_index
    if 0 <= index < len(points):
        sel = points[index]
        ax.scatter([sel["hours"]], [sel["gain"]], s=230, marker="*",
                   color=C_WARN, edgecolor="white", linewidth=1.2,
                   zorder=4, label="TOPSIS 妥协解（均衡预设）")

    ax.set_xlabel("路径总时长（小时）")
    ax.set_ylabel("预期掌握度增益")
    ax.set_title("四目标帕累托前沿的（时长—收益）投影", pad=10)
    ax.legend(fontsize=8.2, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)
    return _finish(fig, _out("fig_pareto.png"))


# ============================================================
# 4. 推荐算法对比
# ============================================================


def fig_recommender_comparison(recommenders: Sequence[dict[str, Any]]) -> Path:
    """推荐策略对比。

    拆成两个面板：左侧是「排序质量」类指标（量级 0.01~0.20），右侧是
    「列表形态」类指标（量级 0.03~0.98）。两类指标量级相差一个数量级，
    画在同一纵轴上会让小值指标被压平、大值指标撑爆坐标轴，因此必须分组。
    """
    rank_metrics = [
        ("precision_at_k", "Precision@10"),
        ("recall_at_k", "Recall@10"),
        ("ndcg_at_k", "NDCG@10"),
        ("map", "MAP"),
    ]
    shape_metrics = [
        ("coverage", "Coverage"),
        ("diversity", "Diversity"),
        ("novelty", "Novelty"),
        ("hit_rate", "HitRate"),
    ]

    def short(name: str) -> str:
        table = {
            "Popularity 基线": "热度基线",
            "Content 内容推荐": "内容",
            "CF-SVD 协同过滤": "CF-SVD",
            "Hybrid 混合推荐（本系统）": "混合(本系统)",
        }
        return table.get(name, name)

    # 固定配色，保证「本系统」在任何图里都是橙色高亮
    def color_of(name: str, index: int) -> str:
        if "本系统" in name:
            return C_WARN
        return [C_GRAY, C_PRIMARY, C_PURPLE][index % 3]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, metrics, title, upper in (
        (axes[0], rank_metrics, "排序质量指标", 1.30),
        (axes[1], shape_metrics, "列表形态指标", 1.18),
    ):
        x = np.arange(len(metrics))
        width = 0.8 / max(1, len(recommenders))
        peak = max(float(r.get(key, 0.0)) for key in (m[0] for m in metrics) for r in recommenders)
        for i, rec in enumerate(recommenders):
            values = [float(rec.get(key, 0.0)) for key, _ in metrics]
            bars = ax.bar(x + i * width - 0.4 + width / 2, values, width * 0.9,
                          label=short(rec["name"]), color=color_of(rec["name"], i), alpha=0.92)
            if "本系统" in rec["name"]:
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + peak * 0.03,
                            f"{v:.3f}", ha="center", fontsize=7.4,
                            color=C_WARN, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], fontsize=9)
        ax.set_title(title, pad=10)
        ax.set_ylim(0, peak * upper)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("指标值")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8.6, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("四种推荐策略的离线评测对比（留出法，K = 10）",
                 y=1.10, fontsize=12.5, fontweight="bold")
    return _finish(fig, _out("fig_recommender_comparison.png"))


# ============================================================
# 5. 效果预测模型对比
# ============================================================


def fig_model_comparison(effect: dict[str, Any], gain: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    names = [m["name"] for m in effect["models"]]
    aucs = [float(m["auc"]) for m in effect["models"]]
    colors = [C_GRAY, C_ACCENT] if len(names) == 2 else PALETTE[: len(names)]
    bars = ax.bar(names, aucs, color=colors, width=0.5, alpha=0.92)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.axhline(0.5, linestyle=":", color=C_GRAY, linewidth=1.2)
    ax.text(len(names) - 0.5, 0.512, "随机猜测 0.500", fontsize=8,
            color=C_GRAY, ha="right")
    ax.set_title("课程完成概率预测：AUC 对比", pad=10)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, max(aucs) * 1.14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax = axes[1]
    names = [m["name"] for m in gain["models"]]
    r2 = [float(m["r2"]) for m in gain["models"]]
    colors = [C_ACCENT, C_GRAY] if len(names) == 2 else PALETTE[: len(names)]
    bars = ax.bar(names, r2, color=colors, width=0.5, alpha=0.92)
    for bar, v in zip(bars, r2):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(r2) * 0.03,
                f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_title("学习效果回归：R² 对比", pad=10)
    ax.set_ylabel("R²")
    ax.set_ylim(0, max(r2) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    return _finish(fig, _out("fig_model_comparison.png"))


def fig_feature_importance(features: Sequence[dict[str, Any]], top: int = 8) -> Path:
    items = list(features)[:top]
    names = [f["name"] for f in items][::-1]
    values = [float(f["importance"]) for f in items][::-1]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    bars = ax.barh(names, values, color=C_PURPLE, alpha=0.85, height=0.62)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.012,
                bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8.5, color="#5A6472")
    ax.set_title("完成概率预测的 Top 特征重要性", pad=10)
    ax.set_xlabel("重要性")
    ax.set_xlim(0, max(values) * 1.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    return _finish(fig, _out("fig_feature_importance.png"))


# ============================================================
# 6. 消融实验
# ============================================================


def fig_ablation(ablation: Sequence[dict[str, Any]], baseline_ndcg: float) -> Path:
    labels = ["完整系统"] + [a["variant"] for a in ablation]
    values = [baseline_ndcg] + [float(a["ndcg_at_k"]) for a in ablation]
    colors = [C_ACCENT] + [C_GRAY] * len(ablation)

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    bars = ax.bar(labels, values, color=colors, width=0.52, alpha=0.92)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.012,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("消融实验：移除关键机制后的 NDCG@10 变化", pad=10)
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, max(values) * 1.2)
    plt.setp(ax.get_xticklabels(), fontsize=8.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return _finish(fig, _out("fig_ablation.png"))


# ============================================================
# 7. 掌握度演化曲线
# ============================================================


def fig_mastery_curve(curve: Sequence[dict[str, Any]], domain_radar: Sequence[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    weeks = [c["week"] for c in curve]
    avg = [float(c["avg_mastery"]) for c in curve]
    mastered = [int(c["mastered"]) for c in curve]
    ax.plot(weeks, avg, marker="o", markersize=4.5, color=C_PRIMARY,
            linewidth=2.0, label="平均掌握度")
    ax.fill_between(weeks, 0, avg, color=C_PRIMARY, alpha=0.12)
    ax.set_xlabel("学习周次")
    ax.set_ylabel("平均掌握度")
    ax.set_ylim(0, 1.0)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(weeks, mastered, marker="s", markersize=4, linestyle="--",
             color=C_ACCENT, linewidth=1.6, label="已掌握知识点数")
    ax2.set_ylabel("已掌握知识点数", color=C_ACCENT)
    ax2.tick_params(axis="y", colors=C_ACCENT)
    ax2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5,
              frameon=False, loc="lower right")
    ax.set_title("掌握度随学习周次的演化", pad=10)

    ax = axes[1]
    labels = [d["domain"] for d in domain_radar]
    current = [float(d["mastery"]) for d in domain_radar]
    target = [float(d.get("target", 0.8)) for d in domain_radar]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    current += current[:1]
    target += target[:1]
    ax.remove()
    ax = fig.add_subplot(1, 2, 2, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, current, color=C_PRIMARY, linewidth=1.8, label="当前掌握度")
    ax.fill(angles, current, color=C_PRIMARY, alpha=0.22)
    ax.plot(angles, target, color=C_ACCENT, linewidth=1.2, linestyle="--", label="目标要求")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_short_domain(l) for l in labels], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="#8A94A6")
    ax.set_title("领域掌握度雷达", pad=18)
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18))

    return _finish(fig, _out("fig_mastery_curve.png"))


def _short_domain(name: str) -> str:
    mapping = {
        "数据与人工智能": "数据与AI",
        "网络与安全": "网络与安全",
        "产品与职场": "产品与职场",
    }
    return mapping.get(name, name)


# ============================================================
# 8. 知识图谱结构示意
# ============================================================


def fig_graph_structure(layer_sizes: Sequence[int]) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    xs = np.arange(1, len(layer_sizes) + 1)
    ax.plot(xs, layer_sizes, marker="o", color=C_PRIMARY, linewidth=2.0, markersize=7)
    for x, y in zip(xs, layer_sizes):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color="#5A6472")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{i - 1}" for i in xs], fontsize=8.5)
    ax.set_xlabel("拓扑层级（按先修关系分层）")
    ax.set_ylabel("该层知识点数量")
    ax.set_title("知识点 DAG 的拓扑分层结构", pad=10)
    ax.fill_between(xs, 0, layer_sizes, color=C_PRIMARY, alpha=0.10)
    ax.set_ylim(0, max(layer_sizes) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)
    return _finish(fig, _out("fig_graph_structure.png"))


def fig_nsga_convergence(history: Sequence[dict[str, Any]]) -> Path:
    """NSGA-II 收敛过程：非支配解个数与种群平均收益随代数的变化。"""
    gens = [int(h["generation"]) for h in history]
    front = [float(h["front_size"]) for h in history]
    mean_gain = [float(h["mean_gain"]) for h in history]

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    ax.plot(gens, front, color=C_PRIMARY, linewidth=2.0, label="非支配解个数")
    ax.fill_between(gens, 0, front, color=C_PRIMARY, alpha=0.12)
    ax.set_xlabel("进化代数")
    ax.set_ylabel("非支配解个数", color=C_PRIMARY)
    ax.tick_params(axis="y", colors=C_PRIMARY)
    ax.set_ylim(0, max(front) * 1.25)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(gens, mean_gain, color=C_WARN, linewidth=1.8, linestyle="--",
             label="种群平均收益")
    ax2.set_ylabel("种群平均收益（归一化）", color=C_WARN)
    ax2.tick_params(axis="y", colors=C_WARN)
    ax2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, frameon=False,
              loc="lower right")
    ax.set_title("NSGA-II 的收敛过程", pad=10)
    return _finish(fig, _out("fig_nsga_convergence.png"))


# ============================================================
# 主入口
# ============================================================


def main() -> list[Path]:
    produced: list[Path] = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("生成图表 ...")
    produced.append(fig_architecture())

    # 数据集统计（来自 data/）
    try:
        from collections import Counter

        from learnpath.data.loader import load_dataset
        from learnpath.knowledge.graph import KnowledgeGraph

        ds = load_dataset()
        graph = KnowledgeGraph.from_dataset(ds)
        domain_counts: dict[str, int] = {}
        for kp in ds.knowledge_points.values():
            domain_counts[kp.domain] = domain_counts.get(kp.domain, 0) + 1
        level_counts = Counter(c.level for c in ds.courses.values())
        produced.append(fig_dataset_overview(domain_counts, dict(level_counts)))
        produced.append(fig_graph_structure([len(l) for l in graph.layers()]))
    except Exception as exc:  # pragma: no cover
        print(f"  [skip] 数据集相关图表：{exc}")

    # 模型指标
    metrics = _load_json(ROOT / "models" / "model_metrics.json")
    if metrics:
        try:
            produced.append(fig_model_comparison(metrics["effect_model"], metrics["gain_model"]))
            produced.append(fig_feature_importance(metrics["effect_model"]["features"]))
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] 模型对比图：{exc}")

    # 评测结果
    evaluation = _load_json(ROOT / "models" / "evaluation.json")
    if evaluation:
        try:
            recommenders = evaluation["recommenders"]
            produced.append(fig_recommender_comparison(recommenders))
            hybrid = next((r for r in recommenders if "本系统" in r["name"]), recommenders[-1])
            produced.append(fig_ablation(evaluation.get("ablation", []), float(hybrid["ndcg_at_k"])))
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] 评测图：{exc}")

    # 仪表板样例（用于曲线与雷达）
    sample = _load_json(ROOT / "docs" / "report" / "figures" / "sample_dashboard.json")
    if sample:
        try:
            produced.append(fig_mastery_curve(sample["mastery_curve"], sample["domain_radar"]))
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] 掌握度曲线图：{exc}")

    pareto = _load_json(ROOT / "docs" / "report" / "figures" / "sample_pareto.json")
    if pareto:
        try:
            produced.append(
                fig_pareto(pareto["points"], int(pareto.get("selected_index", 0)))
            )
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] 帕累托图：{exc}")

    convergence = _load_json(ROOT / "docs" / "report" / "figures" / "sample_convergence.json")
    if convergence:
        try:
            produced.append(fig_nsga_convergence(convergence["history"]))
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] NSGA-II 收敛图：{exc}")

    print(f"共生成 {len(produced)} 张图表 → {FIG_DIR.relative_to(ROOT)}")
    return produced


if __name__ == "__main__":
    main()
