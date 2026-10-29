"""端到端离线评测：跑一遍全系统关键指标，输出 ``models/evaluation.json``。

用法（仓库根目录）::

    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py
    .\\.venv\\Scripts\\python.exe scripts\\evaluate.py --learners 20 --skip-recommenders

评测范围
--------
1. **推荐离线指标**（由 :func:`learnpath.analytics.evaluation.build_evaluation` 提供，
   留一法 + 4 个策略对比 + 消融实验）；
2. **学习效果模型指标**（复用 ``models/model_metrics.json``）；
3. **路径规划 / 多目标优化**（本模块新增）：

   * 单次 ``plan`` 平均耗时（性能硬指标 < 3 s）；
   * 帕累托前沿规模；
   * 三种 preset 的差异化程度（``fast ≤ balanced ≤ thorough`` 的时长单调性、
     是否真的产出不同课程组合）；
   * 缺口覆盖率 = 路径覆盖的缺口知识点数 / 全部缺口知识点数；
   * 时间可行度 ``feasibility``。

4. **自适应 / 社交 / 仪表板**的可用性与合理性采样（动作条数、互补小组、KPI 完整性）。

所有数值都来自真实计算，没有任何随机数占位。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，直接打印中文 / 上标会抛 UnicodeEncodeError；
# 统一切到 UTF-8 并用 replace 兜底。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):  # pragma: no cover - 非标准流
        pass

from learnpath.adaptive.engine import AdaptiveEngine, FeedbackOutcome  # noqa: E402
from learnpath.analytics.dashboard import build_dashboard  # noqa: E402
from learnpath.config import ADAPTIVE, PLAN_PRESETS, PLAN_PRESET_NAMES  # noqa: E402
from learnpath.data.fixtures import build_tiny_dataset  # noqa: E402
from learnpath.data.loader import dataset_exists, load_dataset  # noqa: E402
from learnpath.knowledge.graph import KnowledgeGraph  # noqa: E402
from learnpath.learner.bkt import MasteryTracker  # noqa: E402
from learnpath.learner.profile import LearnerModel  # noqa: E402
from learnpath.paths import MODELS_DIR, ensure_dir  # noqa: E402
from learnpath.planning.planner import PathPlanner  # noqa: E402
from learnpath.social.cohort import SocialMatcher  # noqa: E402


# ============================================================
# 路径规划 / 多目标优化评测
# ============================================================


def evaluate_planning(
    dataset: Any,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    learner_ids: list[str],
) -> dict[str, Any]:
    """在若干学习者上量测路径规划的性能与质量。"""
    durations: list[float] = []
    front_sizes: list[int] = []
    coverages: list[float] = []
    feasibilities: list[float] = []
    monotonic_ok = 0
    distinct_ok = 0
    preset_rows: dict[str, dict[str, list[float]]] = {
        key: {"hours": [], "gain": [], "weeks": [], "steps": [], "objective_gain": [], "objective_time": []}
        for key in PLAN_PRESETS
    }

    for learner_id in learner_ids:
        model = LearnerModel(dataset.learners[learner_id], dataset, graph)
        _closure, gaps, _mastered = planner.knowledge_gaps(model)
        gap_count = max(1, len(gaps))

        started = time.perf_counter()
        paths = {key: planner.plan(model, key) for key in PLAN_PRESETS}
        durations.append(time.perf_counter() - started)

        front = planner.pareto_front(model)
        front_sizes.append(len(front))

        ordered = sorted(paths.items(), key=lambda kv: kv[1].total_hours)
        if ordered[0][0] == "fast" and ordered[-1][0] == "thorough":
            monotonic_ok += 1
        if len({tuple(p.course_ids) for p in paths.values()}) > 1:
            distinct_ok += 1

        for key, path in paths.items():
            row = preset_rows[key]
            row["hours"].append(float(path.total_hours))
            row["gain"].append(float(path.total_gain))
            row["weeks"].append(float(path.weeks))
            row["steps"].append(float(len(path.steps)))
            row["objective_gain"].append(float(path.objectives.get("gain", 0.0)))
            row["objective_time"].append(float(path.objectives.get("time", 0.0)))
            if key == "balanced":
                coverages.append(min(1.0, len(path.kp_ids) / gap_count))
                feasibilities.append(float(path.feasibility))

    sample = len(learner_ids) or 1
    return {
        "n_learners": len(learner_ids),
        "plan_seconds": {
            "mean": round(statistics.fmean(durations), 3) if durations else 0.0,
            "max": round(max(durations), 3) if durations else 0.0,
            "per_preset": round(statistics.fmean(durations) / max(1, len(PLAN_PRESETS)), 3)
            if durations
            else 0.0,
        },
        "pareto_front_size": {
            "mean": round(statistics.fmean(front_sizes), 1) if front_sizes else 0.0,
            "max": max(front_sizes) if front_sizes else 0,
        },
        "presets": {
            key: {
                "name": PLAN_PRESET_NAMES.get(key, key),
                "mean_hours": round(statistics.fmean(rows["hours"]), 1) if rows["hours"] else 0.0,
                "mean_total_gain": round(statistics.fmean(rows["gain"]), 3) if rows["gain"] else 0.0,
                "mean_weeks": round(statistics.fmean(rows["weeks"]), 1) if rows["weeks"] else 0.0,
                "mean_steps": round(statistics.fmean(rows["steps"]), 1) if rows["steps"] else 0.0,
                "mean_objective_gain": round(statistics.fmean(rows["objective_gain"]), 3)
                if rows["objective_gain"]
                else 0.0,
                "mean_objective_time": round(statistics.fmean(rows["objective_time"]), 3)
                if rows["objective_time"]
                else 0.0,
            }
            for key, rows in preset_rows.items()
        },
        "fast_lt_balanced_lt_thorough_hours": round(monotonic_ok / sample, 3),
        "presets_distinct_rate": round(distinct_ok / sample, 3),
        "gap_coverage_mean": round(statistics.fmean(coverages), 3) if coverages else 0.0,
        "feasibility_mean": round(statistics.fmean(feasibilities), 3) if feasibilities else 0.0,
        "target_plan_seconds": 3.0,
        "pass_performance": bool(durations and statistics.fmean(durations) / 3.0 < 3.0),
    }


# ============================================================
# 自适应评测
# ============================================================


def evaluate_adaptive(
    dataset: Any,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    learner_ids: list[str],
) -> dict[str, Any]:
    """采样若干学习者，模拟「低分 / 高分」两类反馈，检查规则是否按预期触发。"""
    engine = AdaptiveEngine(dataset, graph, planner, MasteryTracker())
    low_triggered = 0
    high_triggered = 0
    replanned = 0
    action_counts: list[int] = []

    for learner_id in learner_ids:
        model = LearnerModel(dataset.learners[learner_id], dataset, graph)
        path = planner.plan(model, "balanced")
        if not path.steps:
            continue
        target = path.steps[0].course

        before = float(model.mastery_of(target.kp_ids[0]))
        result = engine.apply_feedback(
            model,
            FeedbackOutcome(
                learner_id=learner_id,
                course_id=target.course_id,
                progress=0.6,
                quiz_score=0.45,
                rating=3.0,
                hours_spent=float(target.duration_hours) * 0.6,
            ),
        )
        after = float(model.mastery_of(target.kp_ids[0]))
        if "未掌握" in result.triggers:
            low_triggered += 1
        if result.replanned:
            replanned += 1
        action_counts.append(len(result.actions))
        assert after >= before - 1e-9, "掌握度更新不应降低掌握度"

        # 高分反馈：换一个学习者，避免与上面的状态耦合
        model2 = LearnerModel(dataset.learners[learner_id], dataset, graph)
        path2 = planner.plan(model2, "balanced")
        if len(path2.steps) >= 2:
            target2 = path2.steps[0].course
            result2 = engine.apply_feedback(
                model2,
                FeedbackOutcome(
                    learner_id=learner_id,
                    course_id=target2.course_id,
                    progress=1.0,
                    quiz_score=0.98,
                    rating=5.0,
                    hours_spent=float(target2.duration_hours),
                ),
            )
            if any(k in result2.triggers for k in ("已掌握跳过", "已掌握")):
                high_triggered += 1

    sample = max(1, len(learner_ids))
    return {
        "n_learners": len(learner_ids),
        "low_score_trigger_rate": round(low_triggered / sample, 3),
        "high_score_trigger_rate": round(high_triggered / sample, 3),
        "replanned_rate": round(replanned / sample, 3),
        "mean_actions": round(statistics.fmean(action_counts), 2) if action_counts else 0.0,
        "thresholds": {
            "low_quiz_score": ADAPTIVE.low_quiz_score,
            "high_quiz_score": ADAPTIVE.high_quiz_score,
            "behind_schedule": ADAPTIVE.behind_schedule,
            "ahead_schedule": ADAPTIVE.ahead_schedule,
        },
    }


# ============================================================
# 社交 / 仪表板评测
# ============================================================


def evaluate_social(
    dataset: Any,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    learner_ids: list[str],
) -> dict[str, Any]:
    """量测同伴匹配与小组编组。"""
    matcher = SocialMatcher(dataset, graph)
    peer_counts: list[int] = []
    similarities: list[float] = []
    group_counts: list[int] = []
    synergies: list[float] = []
    signals: list[float] = []

    for learner_id in learner_ids:
        model = LearnerModel(dataset.learners[learner_id], dataset, graph)
        peers = matcher.find_peers(model, k=6)
        peer_counts.append(len(peers))
        similarities.extend(float(p.similarity) for p in peers)

        path = planner.plan(model, "balanced")
        kp_ids = path.kp_ids[:3] or list(model.target_kps)[:3]
        groups = matcher.form_study_groups(kp_ids, size=4)
        group_counts.append(len(groups))
        synergies.extend(float(g.synergy) for g in groups)

        if path.steps:
            signals.append(float(matcher.peer_signal(model, path.steps[0].course)))

    return {
        "n_learners": len(learner_ids),
        "mean_peers": round(statistics.fmean(peer_counts), 1) if peer_counts else 0.0,
        "mean_similarity": round(statistics.fmean(similarities), 3) if similarities else 0.0,
        "mean_groups": round(statistics.fmean(group_counts), 1) if group_counts else 0.0,
        "mean_synergy": round(statistics.fmean(synergies), 3) if synergies else 0.0,
        "mean_peer_signal": round(statistics.fmean(signals), 3) if signals else 0.0,
    }


def evaluate_dashboard(
    dataset: Any,
    graph: KnowledgeGraph,
    planner: PathPlanner,
    learner_ids: list[str],
) -> dict[str, Any]:
    """检查仪表板九大字段的完整性与掌握度曲线的单调性。"""
    required_keys = (
        "kpi",
        "mastery_curve",
        "domain_radar",
        "difficulty_distribution",
        "modality_mix",
        "time_allocation",
        "completion_forecast",
        "gap_table",
        "history_activity",
    )
    complete = 0
    monotone = 0
    risk_courses: list[int] = []
    for learner_id in learner_ids:
        model = LearnerModel(dataset.learners[learner_id], dataset, graph)
        path = planner.plan(model, "balanced")
        payload = build_dashboard(dataset, graph, model, path)
        if all(key in payload for key in required_keys) and all(payload[key] for key in required_keys[:1] + required_keys[2:]):
            complete += 1
        curve = payload["mastery_curve"]
        values = [float(point["avg_mastery"]) for point in curve]
        if all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1)):
            monotone += 1
        risk_courses.append(int(payload["kpi"]["risk_courses"]))

    sample = max(1, len(learner_ids))
    return {
        "n_learners": len(learner_ids),
        "complete_rate": round(complete / sample, 3),
        "mastery_curve_monotone_rate": round(monotone / sample, 3),
        "mean_risk_courses": round(statistics.fmean(risk_courses), 2) if risk_courses else 0.0,
        "required_keys": list(required_keys),
    }


# ============================================================
# 主流程
# ============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LearnPath 端到端离线评测")
    parser.add_argument("--learners", type=int, default=12, help="参与采样的学习者数量")
    parser.add_argument("--skip-recommenders", action="store_true", help="跳过推荐离线评测（较慢）")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集目录（缺省 data/）")
    parser.add_argument("--output", type=str, default=None, help="输出文件（缺省 models/evaluation.json）")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    if dataset_exists(args.data_dir):
        dataset = load_dataset(args.data_dir)
        source = "full"
    else:  # pragma: no cover
        dataset = build_tiny_dataset()
        source = "tiny-fixture"
        print("[警告] 未找到完整数据集，已回退到 tiny fixture", file=sys.stderr)

    graph = KnowledgeGraph.from_dataset(dataset)
    graph.validate()
    planner = PathPlanner(dataset, graph)
    learner_ids = sorted(dataset.learners)[: max(1, args.learners)]

    print("=" * 78)
    print("LearnPath · 端到端离线评测")
    print("=" * 78)
    print(f"数据源 : {source}（{dataset.stats()['n_interactions']} 条交互）")
    print(f"采样学习者 : {len(learner_ids)}")

    payload: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if args.skip_recommenders:
        payload["recommenders"] = []
        payload["ablation"] = []
        payload["notes"] = ["本次评测跳过了推荐离线指标（--skip-recommenders）"]
    else:
        from learnpath.analytics.evaluation import build_evaluation

        recommend_payload = build_evaluation(dataset, graph)
        payload["recommenders"] = recommend_payload.get("recommenders", [])
        payload["ablation"] = recommend_payload.get("ablation", [])
        payload["notes"] = recommend_payload.get("notes", [])
        payload["effect_model"] = recommend_payload.get("effect_model")
        payload["gain_model"] = recommend_payload.get("gain_model")

    print("\n[1/4] 路径规划 / 多目标优化 ...")
    payload["path_planning"] = evaluate_planning(dataset, graph, planner, learner_ids)
    print("[2/4] 自适应调整 ...")
    payload["adaptive"] = evaluate_adaptive(dataset, graph, planner, learner_ids)
    print("[3/4] 社交学习 ...")
    payload["social"] = evaluate_social(dataset, graph, planner, learner_ids)
    print("[4/4] 学习分析仪表板 ...")
    payload["dashboard"] = evaluate_dashboard(dataset, graph, planner, learner_ids)

    payload["dataset"] = dataset.stats()
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 1)

    output = Path(args.output) if args.output else (ensure_dir(MODELS_DIR) / "evaluation.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    # ---------------- 控制台摘要 ----------------
    planning = payload["path_planning"]
    print("\n---------- 摘要 ----------")
    print(
        "路径规划：plan 平均 %.2f s / 单 preset %.2f s（目标 < 3 s）｜帕累托前沿平均 %.1f 个"
        % (
            planning["plan_seconds"]["mean"],
            planning["plan_seconds"]["per_preset"],
            planning["pareto_front_size"]["mean"],
        )
    )
    for key, row in planning["presets"].items():
        print(
            "  %-8s %-6s 平均 %.1f 小时 / %.1f 周 / %d 门 / total_gain %.2f / objective.gain %.3f"
            % (
                key,
                row["name"],
                row["mean_hours"],
                row["mean_weeks"],
                row["mean_steps"],
                row["mean_total_gain"],
                row["mean_objective_gain"],
            )
        )
    print(
        "  fast≤balanced≤thorough 时长单调率 %.0f%%，三方案互异率 %.0f%%，缺口覆盖率 %.0f%%，可行度 %.2f"
        % (
            planning["fast_lt_balanced_lt_thorough_hours"] * 100,
            planning["presets_distinct_rate"] * 100,
            planning["gap_coverage_mean"] * 100,
            planning["feasibility_mean"],
        )
    )
    adaptive = payload["adaptive"]
    print(
        "自适应：低分触发补强 %.0f%% / 高分触发跳过 %.0f%% / 重排率 %.0f%% / 平均动作 %.1f 条"
        % (
            adaptive["low_score_trigger_rate"] * 100,
            adaptive["high_score_trigger_rate"] * 100,
            adaptive["replanned_rate"] * 100,
            adaptive["mean_actions"],
        )
    )
    social = payload["social"]
    print(
        "社交：平均同伴 %.1f 人 / 平均相似度 %.3f / 平均小组 %.1f 个 / 平均互补度 %.3f"
        % (
            social["mean_peers"],
            social["mean_similarity"],
            social["mean_groups"],
            social["mean_synergy"],
        )
    )
    dashboard = payload["dashboard"]
    print(
        "仪表板：九大字段完整率 %.0f%% / 掌握度曲线单调率 %.0f%% / 平均高风险课 %.1f 门"
        % (
            dashboard["complete_rate"] * 100,
            dashboard["mastery_curve_monotone_rate"] * 100,
            dashboard["mean_risk_courses"],
        )
    )
    if payload.get("effect_model"):
        print(
            "效果模型：最佳分类器 %s AUC=%.4f｜最佳回归器 %s R²=%.4f"
            % (
                payload["effect_model"].get("best"),
                float(
                    next(
                        (
                            row["auc"]
                            for row in payload["effect_model"]["models"]
                            if row["name"] == payload["effect_model"]["best"]
                        ),
                        0.0,
                    )
                ),
                (payload.get("gain_model") or {}).get("best"),
                float(
                    next(
                        (
                            row["r2"]
                            for row in (payload.get("gain_model") or {}).get("models", [])
                            if row["name"] == (payload.get("gain_model") or {}).get("best")
                        ),
                        0.0,
                    )
                ),
            )
        )
    print("\n结果已写入 :", output)
    print("总耗时     : %.1f s" % payload["elapsed_seconds"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
