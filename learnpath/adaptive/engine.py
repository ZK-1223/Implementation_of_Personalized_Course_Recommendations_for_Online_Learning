"""自适应调整引擎（实现要求 ⑥「根据学习进度和反馈动态调整推荐策略」）。

一次反馈要经过的完整链路
------------------------
::

    学习反馈 (FeedbackOutcome)
        │
        ├─(1) 折算出「观测表现」score 与「努力程度」effort
        ├─(2) BKT 状态更新：model.apply_outcome(course, score, effort)  ← 原地改掌握度
        ├─(3) 规则引擎：按 config.ADAPTIVE 的四个阈值判定四类触发
        └─(4) 路径重排：插入补强课 / 跳过冗余课 / 换 preset / 追加进阶课
                    ↓
            AdaptationResult(mastery_updates, delta, actions, replanned, new_path, reason)

规则（阈值全部来自 :class:`learnpath.config.AdaptiveThresholds`）
----------------------------------------------------------------
=====================  ==========================  ==============================================
触发                   阈值                         动作
=====================  ==========================  ==============================================
测验未达标             ``score < 0.60``            判定「未掌握」：在同知识点范围内挑一门
                                                    **更基础、更短**的课程插入路径首位，
                                                    action 文案写明知识点名、掌握度、补强课名
测验优秀               ``score > 0.90``            判定「已掌握」：跳过原路径中**后续覆盖同一
                                                    知识点**的课程（这些课已冗余）
进度落后               ``实际/计划 < 0.70``         切换更紧的 preset（``fast``）并消减收益最低的
                                                    课程，同时给出落后幅度
进度超前               ``实际/计划 > 1.30``         追加**进阶课程**（同领域、难度更高、先修已就绪）
=====================  ==========================  ==============================================

「未掌握」的判定不只看分数，还要求更新后的掌握度仍低于 ``MASTERY_GAP``——
避免「一次失手但历史基础扎实」被误判为需要补强课堂。

``score`` 的折算
----------------
``quiz_score`` 有值时直接用它；无值时按进度折算
``score = 0.20 + 0.70 × progress``（进度 0 → 0.20，进度 1 → 0.90），
这样「看完了但没测验」不会被误判为「完全未掌握」，也不会被判成「优秀」。

``effort_ratio = hours_spent / course.duration_hours``（裁剪到 ``[0, 1]``），
它只调制 BKT 的有效观测分数，不额外改变掌握度上界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Iterable, Mapping, Sequence

from learnpath.config import (
    ADAPTIVE,
    MASTERY_GAP,
    MASTERY_MASTERED,
    PLAN_PRESET_NAMES,
)
from learnpath.data.schema import Course, Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.planning.planner import LearningPath, PathPlanner, PathStep

# ============================================================
# 超参数（报告可直接引用）
# ============================================================

#: 无测验成绩时的分数折算：``score = SCORE_FLOOR + SCORE_SLOPE × progress``
SCORE_FLOOR: Final[float] = 0.20
SCORE_SLOPE: Final[float] = 0.70

#: 插入补强课时候选课程的难度上界相对原课程的折扣（更基础的课）
REMEDIAL_LEVEL_SLACK: Final[int] = 1

#: 追加进阶课时，难度至少高于当前路径最高难度多少级
ADVANCE_LEVEL_DELTA: Final[int] = 0

#: 单次反馈最多追加的进阶课程数
MAX_ADVANCE_COURSES: Final[int] = 3

#: 一次反馈最多消减的低收益课程数
MAX_DROP_COURSES: Final[int] = 2

#: 进度落后时切换的目标 preset
BEHIND_PRESET: Final[str] = "fast"

_EPS: Final[float] = 1e-9


# ============================================================
# 数据结构
# ============================================================


@dataclass(slots=True)
class FeedbackOutcome:
    """一次学习反馈（对应 ``POST /api/feedback`` 的请求体）。

    Attributes:
        learner_id: 学习者 id。
        course_id: 课程 id。
        progress: 学习进度 ``0~1``。
        quiz_score: 结课测验得分 ``0~1``，未测为 ``None``。
        rating: 学习者评分 ``1~5``，未评为 ``None``。
        hours_spent: 本次实际投入小时数。
    """

    learner_id: str
    course_id: str
    progress: float
    quiz_score: float | None = None
    rating: float | None = None
    hours_spent: float = 0.0

    def observed_score(self) -> tuple[float, str]:
        """折算成 BKT 观测分数，并返回折算依据说明。"""
        if self.quiz_score is not None:
            return max(0.0, min(1.0, float(self.quiz_score))), "结课测验分"
        value = SCORE_FLOOR + SCORE_SLOPE * max(0.0, min(1.0, float(self.progress)))
        return max(0.0, min(1.0, value)), "学习进度折算"


@dataclass(slots=True)
class AdaptationResult:
    """一次自适应调整的结果（对应 ``POST /api/feedback`` 的响应体）。

    Attributes:
        mastery_updates: ``kp_id -> 新掌握度``（BKT 更新之后的值）。
        delta: ``kp_id -> 掌握度增量``。
        actions: 中文动作说明列表（前端逐条展示）。
        replanned: 是否触发了路径重排。
        new_path: 重排后的路径（未重排时为 ``None``）。
        reason: 一句话说明「为什么这样调整」。
        triggers: 触发的规则名列表，便于前端打标签与报告统计。
        score: 本次折算出的观测表现。
        course_id / learner_id: 回显，便于前端匹配请求。
    """

    mastery_updates: dict[str, float] = field(default_factory=dict)
    delta: dict[str, float] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    replanned: bool = False
    new_path: LearningPath | None = None
    reason: str = ""
    triggers: list[str] = field(default_factory=list)
    score: float = 0.0
    course_id: str = ""
    learner_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 ``docs/API.md`` 第 5 节的响应体（不含 ``path``，由服务层附加）。"""
        return {
            "learner_id": self.learner_id,
            "course_id": self.course_id,
            "mastery_updates": {k: round(float(v), 3) for k, v in self.mastery_updates.items()},
            "delta": {k: round(float(v), 3) for k, v in self.delta.items()},
            "actions": list(self.actions),
            "replanned": bool(self.replanned),
            "reason": self.reason,
            "triggers": list(self.triggers),
            "score": round(float(self.score), 3),
        }


# ============================================================
# 引擎
# ============================================================


class AdaptiveEngine:
    """反馈驱动的自适应调整引擎。

    Args:
        dataset: 全量数据集。
        graph: 知识点图。
        planner: 路径规划器（用于重规划 / 重建路径）。
        tracker: 掌握度追踪器（用于估计遗忘衰减）。

    Note:
        :meth:`apply_feedback` **会原地更新传入的 ``model``** 的掌握度状态
        （通过 ``model.apply_outcome``），因此调用方缓存同一个 ``LearnerModel``
        实例时，后续推荐 / 规划会自动看到最新的掌握度——这是设计要求。
    """

    def __init__(
        self,
        dataset: Dataset,
        graph: KnowledgeGraph,
        planner: PathPlanner,
        tracker: Any,
    ) -> None:
        self.dataset = dataset
        self.graph = graph
        self.planner = planner
        self.tracker = tracker

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------
    def apply_feedback(
        self,
        model: Any,
        outcome: FeedbackOutcome,
        preset: str = "balanced",
    ) -> AdaptationResult:
        """应用一次学习反馈：更新掌握度 + 规则判定 + 路径重排。

        Args:
            model: 学习者模型（**会被原地更新**）。
            outcome: 学习反馈。
            preset: 当前路径的偏好预设。

        Returns:
            :class:`AdaptationResult`；``replanned=True`` 时 ``new_path`` 为
            调整后的 :class:`~learnpath.planning.planner.LearningPath`。

        Raises:
            KeyError: ``outcome.course_id`` 不在数据集中。
        """
        course = self.dataset.courses[outcome.course_id]

        # ---- (0) 更新前的路径快照（用于识别「后续冗余课程」） ----
        try:
            path_before: LearningPath | None = self.planner.plan(model, preset)
        except Exception:  # pragma: no cover - 规划失败不应阻塞反馈落库
            path_before = None

        # ---- (1) 折算观测表现与努力程度 ----
        score, score_source = outcome.observed_score()
        duration = max(float(course.duration_hours), _EPS)
        effort = max(0.0, min(1.0, float(outcome.hours_spent) / duration))

        # ---- (2) BKT 状态更新（原地） ----
        before = {cov.kp_id: float(model.mastery_of(cov.kp_id)) for cov in course.kp_coverage}
        model.apply_outcome(course, score, effort)
        after = {cov.kp_id: float(model.mastery_of(cov.kp_id)) for cov in course.kp_coverage}
        delta = {kp_id: after[kp_id] - before[kp_id] for kp_id in after}

        actions: list[str] = []
        triggers: list[str] = []
        weakest = min(after.items(), key=lambda item: item[1]) if after else ("", 0.0)

        actions.append(
            "按 BKT 更新 %d 个知识点掌握度，平均提升 %.3f（依据：%s %.2f）"
            % (
                len(after),
                (sum(delta.values()) / len(delta)) if delta else 0.0,
                score_source,
                score,
            )
        )

        # ---- (3) 规则判定 ----
        remediation: list[str] = []
        skip_ids: set[str] = set()
        advance: list[str] = []
        drop_ids: set[str] = set()
        target_preset = preset

        # 规则 1：测验未达标 → 插入更基础的补强课
        # 判定条件：分数低于阈值，且更新后的掌握度仍**未达到「精通」**
        # （MASTERY_MASTERED = 0.85）。用「精通」而不是「缺口阈值 0.60」作为判据，
        # 是因为新标定下 BKT 单步提升较大，60 分左右的观测仍可能把掌握度推到 0.7+，
        # 但「没考好」本身就说明还没真正掌握，仍应给出补强建议。
        if score < ADAPTIVE.low_quiz_score and weakest[1] < MASTERY_MASTERED:
            triggers.append("未掌握")
            remedial = self._find_remedial_course(model, weakest[0], course)
            if remedial is not None:
                remediation.append(remedial.course_id)
                actions.append(
                    "测验 %.2f 低于阈值 %.2f，判定「%s」未达标（掌握度 %.2f < 精通线 %.2f）；"
                    "已插入更基础的补强课《%s》（难度 %d，时长 %.1f 小时）"
                    % (
                        score,
                        ADAPTIVE.low_quiz_score,
                        self.graph.name_of(weakest[0]),
                        weakest[1],
                        MASTERY_MASTERED,
                        remedial.title,
                        int(remedial.level),
                        float(remedial.duration_hours),
                    )
                )
            else:
                actions.append(
                    "测验 %.2f 低于阈值 %.2f，「%s」掌握度仅 %.2f，但找不到更基础的补强课，"
                    "建议原课重学或延长学习时长"
                    % (
                        score,
                        ADAPTIVE.low_quiz_score,
                        self.graph.name_of(weakest[0]),
                        weakest[1],
                    )
                )

        # 规则 2：测验优秀 → 跳过后续覆盖同一知识点的课程
        if score > ADAPTIVE.high_quiz_score and path_before is not None:
            covered = set(course.kp_ids)
            skipped: list[PathStep] = [
                step
                for step in path_before.steps
                if step.course.course_id != course.course_id and set(step.kp_ids) & covered
            ]
            if skipped:
                triggers.append("已掌握跳过")
                skip_ids = {step.course.course_id for step in skipped}
                actions.append(
                    "测验 %.2f 高于阈值 %.2f，判定「%s」已掌握（掌握度 %.2f）；"
                    "已跳过后续覆盖同一知识点的 %d 门课程：%s"
                    % (
                        score,
                        ADAPTIVE.high_quiz_score,
                        "、".join(self.graph.names_of(course.kp_ids)[:2]),
                        weakest[1] if weakest[0] else max(after.values(), default=0.0),
                        len(skipped),
                        "、".join(step.course.title for step in skipped[:3]),
                    )
                )
            else:
                triggers.append("已掌握")
                actions.append(
                    "测验 %.2f 高于阈值 %.2f，判定已掌握；当前路径中已没有覆盖同一知识点的冗余课程"
                    % (score, ADAPTIVE.high_quiz_score)
                )

        # 规则 3 / 4：进度落后 / 超前（需要「当前路径」作基准）
        report = (
            self.progress_report(model, path_before)
            if path_before is not None
            else {"ratio": 1.0, "status": "正常", "actual_progress": 0.0, "plan_progress": 0.0}
        )
        ratio = float(report.get("ratio", 1.0))

        if ratio < ADAPTIVE.behind_schedule and path_before is not None and path_before.steps:
            triggers.append("进度落后")
            if preset != BEHIND_PRESET:
                target_preset = BEHIND_PRESET
            dropped = self._drop_low_benefit(model, path_before, MAX_DROP_COURSES)
            drop_ids = {c.course_id for c in dropped}
            detail = (
                "已消减 %d 门低收益课程：%s" % (len(dropped), "、".join(c.title for c in dropped))
                if dropped
                else "路径已无可消减的低收益课程"
            )
            actions.append(
                "实际进度 %.0f%% / 计划进度 %.0f%% = %.2f，低于阈值 %.2f；"
                "已切换「%s」并%s"
                % (
                    float(report.get("actual_progress", 0.0)) * 100,
                    float(report.get("plan_progress", 0.0)) * 100,
                    ratio,
                    ADAPTIVE.behind_schedule,
                    PLAN_PRESET_NAMES.get(target_preset, target_preset),
                    detail,
                )
            )
        elif ratio > ADAPTIVE.ahead_schedule and path_before is not None:
            triggers.append("进度超前")
            advanced = self._find_advance_courses(model, path_before)
            advance = [c.course_id for c in advanced]
            if advanced:
                actions.append(
                    "实际进度 / 计划进度 = %.2f，高于阈值 %.2f；已追加 %d 门进阶课程：%s"
                    % (
                        ratio,
                        ADAPTIVE.ahead_schedule,
                        len(advanced),
                        "、".join(c.title for c in advanced),
                    )
                )
            else:
                actions.append(
                    "实际进度 / 计划进度 = %.2f，高于阈值 %.2f；暂时没有可追加的进阶课程"
                    % (ratio, ADAPTIVE.ahead_schedule)
                )

        # ---- (4) 路径重排 ----
        changed = bool(remediation or skip_ids or advance or drop_ids or target_preset != preset)
        new_path: LearningPath | None = None
        if changed:
            new_path = self._rebuild_path(
                model,
                path_before=path_before,
                preset=target_preset,
                remediation=remediation,
                skip_ids=skip_ids,
                advance=advance,
                drop_ids=drop_ids,
            )
            if new_path is not None:
                actions.append(
                    "路径已自动重排：%d 门课程 / 共 %.1f 小时 / %d 周（可行度 %.2f）"
                    % (
                        len(new_path.steps),
                        new_path.total_hours,
                        new_path.weeks,
                        new_path.feasibility,
                    )
                )

        reason = self._build_reason(
            score=score,
            score_source=score_source,
            triggers=triggers,
            delta=delta,
            ratio=ratio,
        )

        return AdaptationResult(
            mastery_updates=after,
            delta=delta,
            actions=actions,
            replanned=new_path is not None,
            new_path=new_path,
            reason=reason,
            triggers=triggers,
            score=score,
            course_id=course.course_id,
            learner_id=str(getattr(model.learner, "learner_id", outcome.learner_id)),
        )

    # --------------------------------------------------------
    # 规则辅助
    # --------------------------------------------------------
    def _find_remedial_course(
        self,
        model: Any,
        kp_id: str,
        current: Course,
    ) -> Course | None:
        """为某个未掌握的知识点挑一门「更基础、更短」的补强课。

        筛选与排序规则（全部确定性）：
            1. 必须覆盖 ``kp_id``，且不能是刚学完的那门课；
            2. 难度 ``level ≤ current.level``（更基础或同级），且先修就绪度高；
            3. 优先「读者还没学过」的课程；
            4. 综合分数 = 就绪度 × 0.5 + 掌握度缺口 × 0.3 + 短课时加成 × 0.2。
        """
        candidates: list[tuple[float, Course]] = []
        for cov_course in self.dataset.courses.values():
            if cov_course.course_id == current.course_id:
                continue
            if kp_id not in cov_course.kp_ids:
                continue
            if int(cov_course.level) > int(current.level) + REMEDIAL_LEVEL_SLACK:
                continue
            readiness = float(model.readiness_for(cov_course))
            gap = max(0.0, MASTERY_GAP - float(model.mastery_of(kp_id)))
            brevity = 1.0 / (1.0 + float(cov_course.duration_hours) / 10.0)
            seen_penalty = 0.5 if cov_course.course_id in model.courses_covered() else 1.0
            score = (0.5 * readiness + 0.3 * gap + 0.2 * brevity) * seen_penalty
            candidates.append((score, cov_course))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1].course_id))
        return candidates[0][1]

    def _drop_low_benefit(self, model: Any, path: LearningPath, limit: int) -> list[Course]:
        """从路径中挑出「收益 / 时长」最低的课程作为可消减项（至少保留 1 门）。"""
        if len(path.steps) <= 1:
            return []
        scored: list[tuple[float, Course]] = []
        for step in path.steps:
            hours = max(0.5, float(step.hours))
            scored.append((float(step.expected_gain) / hours, step.course))
        scored.sort(key=lambda item: (item[0], item[1].course_id))
        keep = max(1, len(path.steps) - max(0, int(limit)))
        removable = scored[: len(path.steps) - keep]
        return [course for _score, course in removable]

    def _find_advance_courses(self, model: Any, path: LearningPath) -> list[Course]:
        """追加进阶课程：同目标领域、难度更高、先修已就绪、且不在当前路径内。

        排序基准 = 就绪度 × 0.6 + 预期收益归一 × 0.4，取前
        ``MAX_ADVANCE_COURSES`` 门。
        """
        goal = getattr(model, "goal", None)
        if goal is None:  # pragma: no cover - 契约保证 LearnerModel.goal 存在
            learner = model.learner
            goal = self.dataset.goals[learner.goal_id]
        target_domains = {self.graph.domain_of(kp_id) for kp_id in goal.target_kp_ids if kp_id in self.graph}
        in_path = {step.course.course_id for step in path.steps}
        covered = model.courses_covered()
        max_level = max((int(step.level) for step in path.steps), default=1)

        scored: list[tuple[float, Course]] = []
        for course in self.dataset.courses.values():
            if course.course_id in in_path or course.course_id in covered:
                continue
            if course.domain not in target_domains:
                continue
            if int(course.level) < max_level + ADVANCE_LEVEL_DELTA:
                continue
            readiness = float(model.readiness_for(course))
            if readiness < 0.45:
                continue
            gain = float(model.expected_gain(course))
            scored.append((0.6 * readiness + 0.4 * min(1.0, gain), course))
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1].course_id))
        return [course for _score, course in scored[:MAX_ADVANCE_COURSES]]

    def _rebuild_path(
        self,
        model: Any,
        path_before: LearningPath | None,
        preset: str,
        remediation: Sequence[str],
        skip_ids: Iterable[str],
        advance: Sequence[str],
        drop_ids: Iterable[str],
    ) -> LearningPath | None:
        """把「补强 / 跳过 / 追加 / 消减」四类改动落实到一条新路径上。

        基准集合的选取：有旧路径就用旧路径的课程（保证学习连续性），
        否则重新规划一次。
        """
        if path_before is not None and path_before.steps:
            base = [step.course.course_id for step in path_before.steps]
        else:
            base = list(self.planner.plan(model, preset).course_ids)

        skip = set(skip_ids) | set(drop_ids)
        ordered: list[str] = list(remediation)
        ordered.extend(cid for cid in base if cid not in skip)
        ordered.extend(advance)

        # 补强 / 进阶课程若与基准集合重复，由 planner.rebuild 去重
        new_path = self.planner.rebuild(model, ordered, preset)

        # 若重排后没有任何变化（例如补强课本来就在路径里），再跑一次完整重规划兜底
        if path_before is not None and (
            new_path.course_ids == path_before.course_ids and not advance
        ):
            new_path = self.planner.plan(model, preset)
        return new_path

    @staticmethod
    def _build_reason(
        score: float,
        score_source: str,
        triggers: Sequence[str],
        delta: Mapping[str, float],
        ratio: float,
    ) -> str:
        """生成一句话的原因说明。"""
        if not triggers:
            return (
                "表现（%s %.2f）位于阈值区间内（%.2f~%.2f），掌握度平均提升 %.3f，"
                "进度比 %.2f，保持原路径不变"
                % (
                    score_source,
                    score,
                    ADAPTIVE.low_quiz_score,
                    ADAPTIVE.high_quiz_score,
                    (sum(delta.values()) / len(delta)) if delta else 0.0,
                    ratio,
                )
            )
        return (
            "触发规则：%s；依据 %s %.2f、掌握度平均变化 %.3f、进度比 %.2f，"
            "按 BKT 更新状态并自动调整路径"
            % (
                "、".join(triggers),
                score_source,
                score,
                (sum(delta.values()) / len(delta)) if delta else 0.0,
                ratio,
            )
        )

    # --------------------------------------------------------
    # 进度报告
    # --------------------------------------------------------
    def progress_report(self, model: Any, path: LearningPath) -> dict[str, Any]:
        """计划 vs 实际的进度对比。

        两个进度都归一到 ``[0, 1]``，可以直接相除：

        * **计划进度** ``plan_progress``
          ``= Σ_{step.week ≤ 已学习周数} step.hours / path.total_hours``
          ——「按路径排期，到现在本应学完多少」；
        * **实际进度** ``actual_progress``
          ``= Σ min(实际投入, 计划时长) / path.total_hours``
          ——「学习者实际投入了多少」。

        「已学习周数」由该学习者**最早一次交互的时间戳**起算（日历时间，不循环依赖），
        没有历史记录时为 0，此时两者都为 0、比值取 1.0（中性）。

        Returns:
            ``{"learner_id", "status", "ratio", "actual_progress", "plan_progress",
            "planned_hours", "actual_hours", "path_total_hours", "done_courses",
            "remaining_courses", "elapsed_weeks", "next_course", "retention_estimate",
            "weekly": [...]}``
        """
        learner_id = str(getattr(model.learner, "learner_id", ""))
        path_courses = {step.course.course_id: step for step in path.steps}
        records = [r for r in self.dataset.interactions if r.learner_id == learner_id]

        elapsed_weeks = 0.0
        timestamps = sorted(r.timestamp for r in records if r.timestamp)
        if len(timestamps) >= 2:
            first = _parse_ts(timestamps[0])
            last = _parse_ts(timestamps[-1])
            if first is not None and last is not None:
                elapsed_weeks = max(0.0, (last - first).total_seconds() / (7.0 * 86400.0))

        total_hours = float(path.total_hours)
        if total_hours <= _EPS:
            return {
                "learner_id": learner_id,
                "status": "正常",
                "ratio": 1.0,
                "actual_progress": 0.0,
                "plan_progress": 0.0,
                "planned_hours": 0.0,
                "actual_hours": 0.0,
                "path_total_hours": 0.0,
                "done_courses": 0,
                "remaining_courses": 0,
                "elapsed_weeks": round(elapsed_weeks, 2),
                "next_course": None,
                "retention_estimate": 1.0,
                "weekly": [],
            }

        planned_hours = sum(
            float(step.hours) for step in path.steps if float(step.week) <= elapsed_weeks + _EPS
        )
        actual_hours = 0.0
        done = 0
        for record in records:
            step = path_courses.get(record.course_id)
            if step is None:
                continue
            actual_hours += min(float(record.hours_spent), float(step.hours))
            if record.completed:
                done += 1

        plan_progress = max(0.0, min(1.0, planned_hours / total_hours))
        actual_progress = max(0.0, min(1.0, actual_hours / total_hours))
        if plan_progress <= _EPS:
            ratio = 1.0 if actual_progress <= _EPS else min(2.0, actual_progress / 0.05)
        else:
            ratio = actual_progress / plan_progress

        if ratio < ADAPTIVE.behind_schedule:
            status = "落后"
        elif ratio > ADAPTIVE.ahead_schedule:
            status = "超前"
        else:
            status = "正常"

        # 遗忘衰减估计：空闲周数取「距最后一次交互的周数」
        idle_weeks = 0.0
        if timestamps:
            last_ts = _parse_ts(timestamps[-1])
            if last_ts is not None:
                idle_weeks = max(0.0, (datetime.now() - last_ts).total_seconds() / (7.0 * 86400.0))
        avg_mastery = (
            sum(float(model.mastery_of(kp_id)) for kp_id in path.kp_ids) / len(path.kp_ids)
            if path.kp_ids
            else 0.0
        )
        try:
            retention = float(self.tracker.decay(avg_mastery, idle_weeks))
        except Exception:  # pragma: no cover - tracker 接口异常时的兜底
            retention = avg_mastery

        next_step = next(
            (step for step in path.steps if step.course.course_id not in
             {r.course_id for r in records if r.completed}),
            None,
        )
        return {
            "learner_id": learner_id,
            "status": status,
            "ratio": round(float(ratio), 3),
            "actual_progress": round(float(actual_progress), 3),
            "plan_progress": round(float(plan_progress), 3),
            "planned_hours": round(float(planned_hours), 2),
            "actual_hours": round(float(actual_hours), 2),
            "path_total_hours": round(total_hours, 2),
            "done_courses": int(done),
            "remaining_courses": max(0, len(path.steps) - done),
            "elapsed_weeks": round(elapsed_weeks, 2),
            "next_course": None
            if next_step is None
            else {
                "course_id": next_step.course.course_id,
                "title": next_step.course.title,
                "week": int(next_step.week),
            },
            "retention_estimate": round(float(retention), 4),
            "weekly": [
                {
                    "week": int(item["week"]),
                    "planned_hours": round(float(item["hours"]), 2),
                    "done": bool(float(item["week"]) <= elapsed_weeks),
                }
                for item in path.weekly_plan
            ],
        }


def _parse_ts(text: str) -> datetime | None:
    """解析 ISO 8601 时间戳，失败返回 ``None``（脏数据不致命）。"""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


__all__ = [
    "AdaptiveEngine",
    "FeedbackOutcome",
    "AdaptationResult",
    "SCORE_FLOOR",
    "SCORE_SLOPE",
    "MAX_ADVANCE_COURSES",
    "MAX_DROP_COURSES",
]
