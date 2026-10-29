"""服务容器（Service Container）——引擎层与 HTTP 层之间的唯一粘合点。

职责边界（分层架构中的「服务层」）：

    * **对象构造**：惰性加载数据集、构建知识图谱、训练/加载效果预测模型、
      构建推荐器 / 规划器 / 社交匹配器 / 自适应引擎，并全部缓存为进程级单例；
    * **缓存与失效**：按 ``(learner_id, 参数)`` 缓存推荐、路径、仪表板等结果；
      反馈导致掌握度变化时精确失效该学习者的相关缓存；
    * **契约映射**：把引擎层 dataclass 序列化为 ``docs/API.md`` 规定的 JSON 结构；
    * **降级**：某个算法子模块尚未就绪（``ImportError`` / 缺少符号）时抛
      :class:`ModuleNotReady`，由 HTTP 层映射为 ``503``，**绝不阻断应用启动**。

业务计算（推荐打分、规划、预测、聚合指标）一律不在此实现，只做编排，
以便报告中说明「引擎层 / 服务层 / 表现层」三层职责清晰。
"""

from __future__ import annotations

import importlib
import threading
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Mapping

from learnpath.config import (
    DOMAIN_ICONS,
    DOMAINS,
    MASTERY_GAP,
    MASTERY_MASTERED,
    MODALITIES,
    MODALITY_NAMES,
    MOTIVATIONS,
    PLAN_PRESETS,
    PLAN_PRESET_NAMES,
)
from learnpath.data.fixtures import build_tiny_dataset
from learnpath.data.schema import Course, Dataset, Goal, Learner
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.paths import DATA_DIR

#: 服务版本号（与 ``docs/API.md`` 中的 ``/api/health`` 约定一致）
VERSION = "1.0.0"


# ============================================================
# 异常
# ============================================================


class NotFound(LookupError):
    """请求的资源（学习者 / 课程 / 领域）不存在 → HTTP 404。"""


class ModuleNotReady(RuntimeError):
    """某个算法子模块尚未就绪 → HTTP 503（服务本身仍然可用）。"""

    def __init__(self, module: str, detail: str = "") -> None:
        self.module = module
        self.detail = detail or f"模块尚未就绪：{module}"
        super().__init__(self.detail)


# ============================================================
# 小工具
# ============================================================


def _import_attr(module: str, attr: str) -> Any:
    """惰性导入 ``module.attr``；模块或符号缺失统一抛 :class:`ModuleNotReady`。

    使用惰性导入（而非文件顶部 import）的原因：算法子包由其他同学并行开发，
    服务必须在它们缺席时仍能启动并通过不依赖算法的接口。
    """
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # ImportError 及其内部依赖错误
        raise ModuleNotReady(module, f"模块尚未就绪：{module}（{type(exc).__name__}）") from exc
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise ModuleNotReady(
            f"{module}.{attr}", f"模块尚未就绪：{module} 缺少符号 {attr}"
        ) from exc


def r2(value: Any) -> float:
    """保留 2 位小数（时长）。"""
    return round(float(value), 2)


def r3(value: Any) -> float:
    """保留 3 位小数（分数 / 概率，落在 [0, 1] 的会被截断到区间内）。"""
    number = float(value)
    if number != number:  # NaN
        return 0.0
    if -1e-9 <= number <= 1.0 + 1e-9:
        number = min(1.0, max(0.0, number))
    return round(number, 3)


def risk_of(p_completion: float) -> str:
    """完成概率 → 风险等级（阈值与 ``learnpath.prediction.effect`` 保持一致）。"""
    if p_completion >= 0.70:
        return "低"
    if p_completion >= 0.40:
        return "中"
    return "高"


def _mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else 0.0


def _plain_mapping(value: Any) -> dict[str, Any]:
    """把引擎返回的诊断字典（如 ``coverage``）转成 JSON 安全结构。

    只做类型归一（numpy 标量 → Python 标量）+ 数值定点，**不做任何业务换算**，
    保证「算法侧权威值原样透传」。
    """
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            result[str(key)] = bool(item)
        elif isinstance(item, Integral):
            result[str(key)] = int(item)
        elif isinstance(item, Real):
            result[str(key)] = round(float(item), 4)
        else:
            result[str(key)] = item
    return result


# ============================================================
# 服务容器
# ============================================================


class ServiceContainer:
    """进程级单例容器：所有引擎对象的构造入口与结果缓存。

    线程安全策略：所有「首次初始化」与「缓存读写」都在 ``self._lock`` 内完成，
    采用双检锁（double-checked locking）避免并发请求重复加载/训练。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # ---- 基础数据（进程级永久缓存）----
        self._dataset: Dataset | None = None
        self._data_source: str = "unknown"
        self._graph: KnowledgeGraph | None = None
        self._completed_count: dict[str, int] | None = None

        # ---- 引擎对象（惰性构造 + 缓存）----
        self._predictor: Any = None
        self._predictor_failed: bool = False
        self._recommender: Any = None
        self._planner: Any = None
        self._matcher: Any = None
        self._adaptive: Any = None
        self._tracker: Any = None
        self._learner_models: dict[str, Any] = {}

        # ---- 业务结果缓存 ----
        self._meta_cache: dict[str, Any] | None = None
        self._graph_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
        self._recommend_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._path_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._pareto_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._plan_object_cache: dict[tuple[str, str], Any] = {}
        self._social_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._analytics_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._evaluation_cache: dict[str, Any] | None = None
        self._prediction_cache: dict[tuple[str, str], Any] = {}

    # ========================================================
    # 基础数据 / 图谱
    # ========================================================
    @property
    def dataset(self) -> Dataset:
        """数据集单例：优先加载 ``data/``，不可用时退化为微型 fixture。"""
        if self._dataset is not None:
            return self._dataset
        with self._lock:
            if self._dataset is None:
                dataset, source = self._load_dataset()
                self._dataset = dataset
                self._data_source = source
        return self._dataset

    @property
    def data_source(self) -> str:
        """``full``（来自 ``data/``）或 ``tiny``（内置 fixture）。"""
        _ = self.dataset
        return self._data_source

    def _load_dataset(self) -> tuple[Dataset, str]:
        """尽力加载完整数据集，任何失败都退化为 :func:`build_tiny_dataset`。"""
        loader = None
        try:
            loader = _import_attr("learnpath.data.loader", "load_or_build")
        except ModuleNotReady:
            loader = None

        if loader is not None:
            try:
                dataset = loader()
                has_files = DATA_DIR.is_dir() and any(DATA_DIR.glob("*.json"))
                return dataset, "full" if has_files else "tiny"
            except Exception:  # noqa: BLE001 - 数据层未就绪时静默降级
                pass

        # 退一步：尝试纯粹的 load_dataset()
        try:
            load_dataset = _import_attr("learnpath.data.loader", "load_dataset")
            if DATA_DIR.is_dir() and any(DATA_DIR.glob("*.json")):
                return load_dataset(DATA_DIR), "full"
        except Exception:  # noqa: BLE001
            pass

        return build_tiny_dataset(), "tiny"

    @property
    def graph(self) -> KnowledgeGraph:
        """知识图谱单例（首次构造时校验无环）。"""
        if self._graph is not None:
            return self._graph
        with self._lock:
            if self._graph is None:
                graph = KnowledgeGraph.from_dataset(self.dataset)
                try:
                    graph.validate()
                except Exception:  # noqa: BLE001 - 环会让统计接口失效，但不阻止服务
                    pass
                self._graph = graph
        return self._graph

    # ========================================================
    # 引擎对象（惰性 + 缓存）
    # ========================================================
    @property
    def effect_predictor(self) -> Any:
        """效果预测模型：优先加载 ``models/``，否则用小数据现场训练。

        返回 ``None`` 表示预测能力不可用（不影响其它接口）。
        """
        if self._predictor is not None or self._predictor_failed:
            return self._predictor
        with self._lock:
            if self._predictor is not None or self._predictor_failed:
                return self._predictor
            try:
                cls = _import_attr("learnpath.prediction.effect", "EffectPredictor")
                predictor = cls()
                try:
                    predictor.load()
                except Exception:  # noqa: BLE001 - 无模型文件则现场训练
                    predictor.fit(self.dataset, self.graph)
                self._predictor = predictor
            except ModuleNotReady:
                self._predictor_failed = True
            except Exception:  # noqa: BLE001 - 训练失败不影响服务可用性
                self._predictor_failed = True
        return self._predictor

    @property
    def recommender(self) -> Any:
        """混合推荐器单例。"""
        if self._recommender is not None:
            return self._recommender
        with self._lock:
            if self._recommender is None:
                cls = _import_attr("learnpath.recommend.hybrid", "HybridRecommender")
                self._recommender = cls(self.dataset, self.graph, effect_predictor=self.effect_predictor)
        return self._recommender

    @property
    def planner(self) -> Any:
        """路径规划器单例。"""
        if self._planner is not None:
            return self._planner
        with self._lock:
            if self._planner is None:
                cls = _import_attr("learnpath.planning.planner", "PathPlanner")
                self._planner = cls(self.dataset, self.graph)
        return self._planner

    @property
    def matcher(self) -> Any:
        """社交学习匹配器单例。"""
        if self._matcher is not None:
            return self._matcher
        with self._lock:
            if self._matcher is None:
                cls = _import_attr("learnpath.social.cohort", "SocialMatcher")
                self._matcher = cls(self.dataset, self.graph)
        return self._matcher

    @property
    def tracker(self) -> Any:
        """贝叶斯知识追踪器（自适应引擎依赖）。"""
        if self._tracker is not None:
            return self._tracker
        with self._lock:
            if self._tracker is None:
                cls = _import_attr("learnpath.learner.bkt", "MasteryTracker")
                self._tracker = cls()
        return self._tracker

    @property
    def adaptive_engine(self) -> Any:
        """自适应调整引擎单例。"""
        if self._adaptive is not None:
            return self._adaptive
        with self._lock:
            if self._adaptive is None:
                cls = _import_attr("learnpath.adaptive.engine", "AdaptiveEngine")
                self._adaptive = cls(self.dataset, self.graph, self.planner, self.tracker)
        return self._adaptive

    def learner_model(self, learner_id: str) -> Any:
        """取（或构造）学习者的 ``LearnerModel`` 单例。

        必须是单例：``POST /api/feedback`` 会原地更新其掌握度，
        单例才能让后续推荐 / 路径 / 仪表板看到更新后的画像。
        """
        learner = self._require_learner(learner_id)
        model = self._learner_models.get(learner_id)
        if model is not None:
            return model
        with self._lock:
            model = self._learner_models.get(learner_id)
            if model is None:
                cls = _import_attr("learnpath.learner.profile", "LearnerModel")
                model = cls(learner, self.dataset, self.graph)
                self._learner_models[learner_id] = model
        return model

    # ========================================================
    # 查找与校验
    # ========================================================
    def _require_learner(self, learner_id: str) -> Learner:
        learner = self.dataset.learners.get(learner_id)
        if learner is None:
            raise NotFound(f"学习者不存在：{learner_id}")
        return learner

    def _require_course(self, course_id: str) -> Course:
        course = self.dataset.courses.get(course_id)
        if course is None:
            raise NotFound(f"课程不存在：{course_id}")
        return course

    def _goal_of(self, learner: Learner) -> Goal:
        return self.dataset.goals[learner.goal_id]

    def _completed_counts(self) -> dict[str, int]:
        """``learner_id -> 已完成课程数``（首次使用时统计一次）。"""
        if self._completed_count is None:
            with self._lock:
                if self._completed_count is None:
                    counts: dict[str, int] = {}
                    for inter in self.dataset.interactions:
                        if inter.completed:
                            counts[inter.learner_id] = counts.get(inter.learner_id, 0) + 1
                    self._completed_count = counts
        return self._completed_count

    def _completion_rate_of(self, learner_id: str) -> float:
        """某学习者的交互完成率（历史行为统计）。"""
        interactions = self.dataset.interactions_of(learner_id)
        if not interactions:
            return 0.0
        return r3(sum(1 for i in interactions if i.completed) / len(interactions))

    # ========================================================
    # 1. 系统信息
    # ========================================================
    def health(self) -> dict[str, Any]:
        """健康检查：``dataset_loaded`` 表示数据层是否已就绪（首次调用会触发加载）。

        这里主动触发一次数据加载：健康检查的语义是「我现在能否对外服务」，
        而不是「我之前有没有被调用过」。
        """
        _ = self.dataset
        return {
            "status": "ok",
            "version": VERSION,
            "dataset_loaded": self._dataset is not None,
        }

    def meta(self) -> dict[str, Any]:
        """前端首屏所需的全部枚举与统计（含附加字段 ``data_source``）。"""
        if self._meta_cache is not None:
            return self._meta_cache
        with self._lock:
            if self._meta_cache is not None:
                return self._meta_cache

            dataset = self.dataset
            graph = self.graph
            stats = dataset.stats()

            domains = []
            for domain in DOMAINS:
                domains.append(
                    {
                        "name": domain,
                        "icon": DOMAIN_ICONS.get(domain, "code"),
                        "n_kps": sum(1 for kp in dataset.knowledge_points.values() if kp.domain == domain),
                        "n_courses": sum(1 for c in dataset.courses.values() if c.domain == domain),
                    }
                )

            goals = [
                {
                    "goal_id": goal.goal_id,
                    "name": goal.name,
                    "target_role": goal.target_role,
                    "description": goal.description,
                    "n_target_kps": len(goal.target_kp_ids),
                    "default_weeks": int(goal.default_weeks),
                }
                for goal in dataset.goals.values()
            ]

            presets = [
                {
                    "key": key,
                    "name": PLAN_PRESET_NAMES.get(key, key),
                    "weights": {
                        "gain": r3(weights.gain),
                        "time": r3(weights.time),
                        "smoothness": r3(weights.smoothness),
                        "fit": r3(weights.fit),
                    },
                }
                for key, weights in PLAN_PRESETS.items()
            ]

            learners = [
                self._learner_summary(learner) for learner in dataset.learners.values()
            ]

            self._meta_cache = {
                "stats": stats,
                "graph": graph.stats().to_dict(),
                "domains": domains,
                "goals": goals,
                "presets": presets,
                "learners": learners,
                "modalities": [{"key": key, "name": MODALITY_NAMES.get(key, key)} for key in MODALITIES],
                "motivations": list(MOTIVATIONS),
                # 附加字段（非契约）：便于排查当前用的是完整数据集还是 fixture
                "data_source": self.data_source,
            }
            return self._meta_cache

    def _learner_summary(self, learner: Learner) -> dict[str, Any]:
        """``/api/meta`` 与 ``/api/learners`` 的列表项结构。"""
        goal = self._goal_of(learner)
        return {
            "learner_id": learner.learner_id,
            "nickname": learner.nickname,
            "goal_id": learner.goal_id,
            "goal_name": goal.name,
            "ability": r3(learner.ability),
            "time_budget_hours_per_week": r2(learner.time_budget_hours_per_week),
        }

    # ========================================================
    # 2. 课程库
    # ========================================================
    def list_courses(
        self,
        q: str | None = None,
        domain: str | None = None,
        level: int | None = None,
        modality: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """课程库分页检索：支持关键词、领域、难度、模态过滤。"""
        dataset = self.dataset
        keyword = (q or "").strip().lower()

        matched: list[Course] = []
        for course in dataset.courses.values():
            if domain and course.domain != domain:
                continue
            if level is not None and int(course.level) != int(level):
                continue
            if modality and course.modality != modality:
                continue
            if keyword:
                haystack = " ".join(
                    [course.title, course.instructor, course.domain, *course.tags]
                ).lower()
                kp_names = " ".join(
                    dataset.knowledge_points[kp].name
                    for kp in course.kp_ids
                    if kp in dataset.knowledge_points
                ).lower()
                if keyword not in haystack and keyword not in kp_names:
                    continue
            matched.append(course)

        matched.sort(key=lambda c: (-c.rating, -c.enroll_count, c.course_id))
        return self._paginate(matched, page, page_size, self._course_item)

    def _course_item(self, course: Course) -> dict[str, Any]:
        leak = dict(course.to_dict())
        leak["kp_names"] = [
            self.dataset.knowledge_points[kp].name
            for kp in course.kp_ids
            if kp in self.dataset.knowledge_points
        ]
        return leak

    # ========================================================
    # 3. 学习者
    # ========================================================
    def list_learners(
        self,
        q: str | None = None,
        goal_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """学习者分页检索：支持昵称 / id 关键词与目标过滤。"""
        keyword = (q or "").strip().lower()
        learners = [
            learner
            for learner in self.dataset.learners.values()
            if (not goal_id or learner.goal_id == goal_id)
            and (
                not keyword
                or keyword in learner.nickname.lower()
                or keyword in learner.learner_id.lower()
            )
        ]
        learners.sort(key=lambda item: item.learner_id)
        return self._paginate(learners, page, page_size, self._learner_summary)

    def learner_detail(self, learner_id: str) -> dict[str, Any]:
        """学习者画像：基础信息 + 目标 + 掌握度画像 + 学习历史。"""
        learner = self._require_learner(learner_id)
        goal = self._goal_of(learner)
        model = self.learner_model(learner_id)
        return {
            "learner": learner.to_dict(),
            "goal": goal.to_dict(),
            "profile": self.profile_payload(learner, model),
            "history": self._history_of(learner_id),
        }

    def profile_payload(self, learner: Learner, model: Any) -> dict[str, Any]:
        """把 ``ProfileSnapshot`` 映射为 API 文档第 2 节的 ``profile`` 结构。"""
        graph = self.graph
        snapshot = model.snapshot()

        mastery = {kp: r3(v) for kp, v in snapshot.mastery.items()}
        mastery_names = sorted(
            (
                {"kp_id": kp, "name": graph.name_of(kp), "value": value}
                for kp, value in mastery.items()
                if kp in graph
            ),
            key=lambda item: (-item["value"], item["kp_id"]),
        )
        gaps = {kp: r3(v) for kp, v in snapshot.gaps.items()}
        target_kps = [kp for kp in snapshot.target_kps]
        target_set = set(target_kps)
        # 契约示例中 mastered_count + gap_count == total_target_kps，
        # 因此这两个计数都以「目标知识点」为口径；掌握度档案仍返回全量 mastery。
        mastered_count = sum(
            1 for kp in target_kps if mastery.get(kp, 0.0) >= MASTERY_MASTERED
        )
        gap_count = sum(1 for kp in target_kps if mastery.get(kp, 0.0) < MASTERY_GAP)
        if gap_count == 0 and gaps:  # 引擎给出的缺口已做过滤时以其为准
            gap_count = len(gaps)

        return {
            "learner_id": learner.learner_id,
            "mastery": mastery,
            "mastery_names": mastery_names[:80],
            "domain_mastery": {domain: r3(v) for domain, v in snapshot.domain_mastery.items()},
            "target_kps": target_kps,
            "target_kp_names": graph.names_of(target_kps),
            "gaps": gaps,
            "gap_names": graph.names_of(gaps),
            "readiness": r3(snapshot.readiness),
            "style_vector": {key: r3(v) for key, v in snapshot.style_vector.items()},
            "time_budget": r2(snapshot.time_budget),
            "total_available_hours": r2(learner.total_available_hours),
            "mastered_count": mastered_count,
            "gap_count": gap_count,
            "completion_rate": self._completion_rate_of(learner.learner_id),
        }

    def _history_of(self, learner_id: str, limit: int = 60) -> list[dict[str, Any]]:
        """学习历史（最近在前）。"""
        dataset = self.dataset
        rows: list[dict[str, Any]] = []
        for inter in dataset.interactions_of(learner_id):
            course = dataset.courses.get(inter.course_id)
            rows.append(
                {
                    "course_id": inter.course_id,
                    "title": course.title if course else inter.course_id,
                    "domain": course.domain if course else "",
                    "progress": r3(inter.progress),
                    "completed": bool(inter.completed),
                    "rating": None if inter.rating is None else round(float(inter.rating), 1),
                    "hours_spent": r2(inter.hours_spent),
                    "quiz_score": None if inter.quiz_score is None else r3(inter.quiz_score),
                    "timestamp": inter.timestamp,
                }
            )
        rows.sort(key=lambda row: (row["timestamp"], row["course_id"]), reverse=True)
        return rows[:limit]

    # ========================================================
    # 4. 智能推荐
    # ========================================================
    def recommend(self, learner_id: str, k: int = 10) -> dict[str, Any]:
        """混合推荐 Top-k（含分项得分、中文理由、预期增益与完成概率）。"""
        self._require_learner(learner_id)
        cache_key = (learner_id, int(k))
        cached = self._recommend_cache.get(cache_key)
        if cached is not None:
            return cached

        recommender = self.recommender
        model = self.learner_model(learner_id)
        raw_items = list(recommender.recommend(model, k=int(k)))
        items = [self._recommend_item(learner_id, model, rec, rank) for rank, rec in enumerate(raw_items, 1)]

        # 对外契约要求「rank 与 score 一致且 score 单调不增」：
        # 推荐器内部会按 MMR 做多样性重排，其顺序未必与融合分同序；
        # 这里以「序列化后的 3 位小数分」为唯一权威重排一次（稳定排序，
        # 同分时保留推荐器给出的原始先后），再重新编号 rank。
        items.sort(key=lambda item: (-float(item["score"]), int(item["rank"])))
        for index, item in enumerate(items, 1):
            item["rank"] = index

        payload = {
            "learner_id": learner_id,
            "k": int(k),
            "count": len(items),
            "items": items,
        }
        with self._lock:
            self._recommend_cache[cache_key] = payload
        return payload

    def _recommend_item(self, learner_id: str, model: Any, rec: Any, rank: int) -> dict[str, Any]:
        """单条推荐结果 → 契约结构（缺失的分项稳健兜底为 0）。"""
        course: Course = rec.course
        components = dict(getattr(rec, "components", None) or {})
        p_completion = self._p_completion_for(learner_id, model, course, getattr(rec, "p_completion", None))

        return {
            "rank": rank,
            "course_id": course.course_id,
            "title": course.title,
            "domain": course.domain,
            "level": int(course.level),
            "duration_hours": r2(course.duration_hours),
            "modality": course.modality,
            "modality_name": MODALITY_NAMES.get(course.modality, course.modality),
            "rating": round(float(course.rating), 2),
            "enroll_count": int(course.enroll_count),
            "price": r2(course.price),
            "kp_names": self.graph.names_of(course.kp_ids),
            "score": r3(rec.score),
            "components": {
                "content": r3(components.get("content", 0.0)),
                "collaborative": r3(components.get("collaborative", 0.0)),
                "quality": r3(components.get("quality", 0.0)),
                "readiness": r3(components.get("readiness", 0.0)),
                "social": r3(components.get("social", 0.0)),
            },
            "reasons": list(getattr(rec, "reasons", None) or []),
            "expected_gain": r3(getattr(rec, "expected_gain", 0.0)),
            "p_completion": r3(p_completion),
            "risk_level": risk_of(p_completion),
            "readiness": r3(self._call(model, "readiness_for", course, default=0.0)),
            "style_fit": r3(self._call(model, "style_fit", course, default=0.0)),
            "time_fit": r3(self._call(model, "time_fit", course, default=0.0)),
        }

    def _p_completion_for(
        self,
        learner_id: str,
        model: Any,
        course: Course,
        fallback: float | None = None,
    ) -> float:
        """完成概率：优先效果预测模型，其次推荐器给出的估计，最后 0.5 兜底。"""
        cache_key = (learner_id, course.course_id)
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            return float(cached)

        value: float | None = None
        predictor = self.effect_predictor
        if predictor is not None:
            try:
                prediction = predictor.predict_completion(self.dataset, self.graph, model, course)
                value = float(prediction.p_completion)
            except Exception:  # noqa: BLE001 - 预测失败不影响推荐主流程
                value = None
        if value is None and fallback is not None:
            value = float(fallback)
        if value is None:
            value = 0.5

        with self._lock:
            self._prediction_cache[cache_key] = value
        return value

    def _call(self, obj: Any, name: str, *args: Any, default: Any = 0.0) -> Any:
        """安全调用引擎对象的可选方法（不存在或抛错时返回默认值）。"""
        func = getattr(obj, name, None)
        if not callable(func):
            return default
        try:
            return func(*args)
        except Exception:  # noqa: BLE001
            return default

    # ========================================================
    # 5. 学习路径
    # ========================================================
    def _plan(self, learner_id: str, preset: str) -> Any:
        """取（或计算）``LearningPath`` 对象，按 (learner, preset) 缓存。"""
        preset = preset if preset in PLAN_PRESETS else "balanced"
        cache_key = (learner_id, preset)
        plan = self._plan_object_cache.get(cache_key)
        if plan is None:
            model = self.learner_model(learner_id)
            plan = self.planner.plan(model, preset=preset)
            with self._lock:
                self._plan_object_cache[cache_key] = plan
        return plan

    def path(self, learner_id: str, preset: str = "balanced") -> dict[str, Any]:
        """学习路径方案（含三种预设对比）。"""
        self._require_learner(learner_id)
        preset = preset if preset in PLAN_PRESETS else "balanced"
        cache_key = (learner_id, preset)
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return cached

        plan = self._plan(learner_id, preset)
        payload = self._path_payload(learner_id, plan, preset)
        with self._lock:
            self._path_cache[cache_key] = payload
        return payload

    def _path_payload(self, learner_id: str, plan: Any, preset: str) -> dict[str, Any]:
        """``LearningPath`` → API 文档第 4 节结构。

        预算 / 覆盖率相关字段（``budget_hours`` / ``over_budget`` / ``coverage_ratio`` /
        ``covered_target_kp_*`` / ``uncovered_target_kp_*`` / ``coverage``）由算法侧
        唯一权威给出，服务层**只做透传**（仅对数值做 JSON 安全的定点化），不重算、不裁剪。
        """
        graph = self.graph
        goal: Goal = self._goal_of(self._require_learner(learner_id))

        steps: list[dict[str, Any]] = []
        for step in getattr(plan, "steps", []):
            course: Course = step.course
            steps.append(
                {
                    "order": int(step.order),
                    "course_id": course.course_id,
                    "title": course.title,
                    "domain": course.domain,
                    "level": int(step.level if step.level is not None else course.level),
                    "hours": r2(step.hours),
                    "cumulative_hours": r2(step.cumulative_hours),
                    "week": int(step.week),
                    "kp_names": graph.names_of(step.kp_ids or course.kp_ids),
                    "expected_gain": r3(step.expected_gain),
                    "modality": course.modality,
                }
            )

        objectives = dict(getattr(plan, "objectives", None) or {})

        # 覆盖率相关字段（含 id 与中文名）全部由算法侧给出，服务层只做列表拷贝透传，
        # 既不重算覆盖判定、也不重新翻译名称（``LearningPath`` 保证两个名单一一对应）。
        covered_ids = list(getattr(plan, "covered_target_kp_ids", None) or [])
        uncovered_ids = list(getattr(plan, "uncovered_target_kp_ids", None) or [])
        covered_names = list(getattr(plan, "covered_target_kp_names", None) or [])
        uncovered_names = list(getattr(plan, "uncovered_target_kp_names", None) or [])

        return {
            "learner_id": learner_id,
            "preset": preset,
            "preset_name": PLAN_PRESET_NAMES.get(preset, preset),
            "goal": {"goal_id": goal.goal_id, "name": goal.name},
            "total_hours": r2(getattr(plan, "total_hours", 0.0)),
            "weeks": int(getattr(plan, "weeks", 0)),
            "total_gain": r3(getattr(plan, "total_gain", 0.0)),
            "feasibility": r3(getattr(plan, "feasibility", 0.0)),
            "objectives": {
                "gain": r3(objectives.get("gain", 0.0)),
                "time": r3(objectives.get("time", 0.0)),
                "smoothness": r3(objectives.get("smoothness", 0.0)),
                "fit": r3(objectives.get("fit", 0.0)),
            },
            # ---- 预算与覆盖率（算法侧权威值透传）----
            "budget_hours": r2(getattr(plan, "budget_hours", 0.0)),
            "over_budget": bool(getattr(plan, "over_budget", False)),
            "coverage_ratio": r3(getattr(plan, "coverage_ratio", 0.0)),
            "covered_target_kp_ids": covered_ids,
            "covered_target_kp_names": covered_names,
            "uncovered_target_kp_ids": uncovered_ids,
            "uncovered_target_kp_names": uncovered_names,
            "coverage": _plain_mapping(getattr(plan, "coverage", None)),
            # ---- 路径本体 ----
            "steps": steps,
            "weekly_plan": self._weekly_plan(plan, steps),
            "milestones": self._milestones(plan, steps),
            "alternatives": self._alternatives(learner_id, preset),
        }

    def _weekly_plan(self, plan: Any, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """周计划：优先用规划器给出的，缺失时按步骤周次聚合。"""
        provided = getattr(plan, "weekly_plan", None)
        if provided:
            rows = []
            for row in provided:
                if isinstance(row, Mapping):
                    rows.append(
                        {
                            "week": int(row.get("week", 0)),
                            "hours": r2(row.get("hours", 0.0)),
                            "course_ids": [str(c) for c in row.get("course_ids", [])],
                        }
                    )
                else:  # 兼容 dataclass 形态
                    rows.append(
                        {
                            "week": int(getattr(row, "week", 0)),
                            "hours": r2(getattr(row, "hours", 0.0)),
                            "course_ids": [str(c) for c in getattr(row, "course_ids", [])],
                        }
                    )
            if rows:
                return rows

        buckets: dict[int, dict[str, Any]] = {}
        for step in steps:
            week = int(step["week"]) or 1
            bucket = buckets.setdefault(week, {"week": week, "hours": 0.0, "course_ids": []})
            bucket["hours"] = r2(bucket["hours"] + float(step["hours"]))
            bucket["course_ids"].append(step["course_id"])
        return [buckets[week] for week in sorted(buckets)]

    def _milestones(self, plan: Any, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """阶段里程碑：优先用规划器的，缺失时按「首次出现的新知识点」合成。"""
        provided = getattr(plan, "milestones", None)
        if provided:
            rows = []
            for row in provided:
                if isinstance(row, Mapping):
                    rows.append(
                        {
                            "week": int(row.get("week", 0)),
                            "label": str(row.get("label", "")),
                            "kp_names": [str(n) for n in row.get("kp_names", [])],
                        }
                    )
                else:
                    rows.append(
                        {
                            "week": int(getattr(row, "week", 0)),
                            "label": str(getattr(row, "label", "")),
                            "kp_names": [str(n) for n in getattr(row, "kp_names", [])],
                        }
                    )
            if rows:
                return rows

        seen: set[str] = set()
        rows = []
        for step in steps:
            fresh = [name for name in step["kp_names"] if name not in seen]
            if not fresh:
                continue
            seen.update(fresh)
            rows.append(
                {
                    "week": int(step["week"]),
                    "label": "掌握 " + "、".join(fresh[:2]),
                    "kp_names": fresh[:4],
                }
            )
        return rows[:8]

    def _alternatives(self, learner_id: str, preset: str) -> list[dict[str, Any]]:
        """另外两种预设方案的对比摘要（契约补充：仅额外带 ``coverage_ratio``）。"""
        rows = []
        for key in PLAN_PRESETS:
            if key == preset:
                continue
            try:
                other = self._plan(learner_id, key)
            except ModuleNotReady:
                continue
            objectives = dict(getattr(other, "objectives", None) or {})
            coverage_ratio = getattr(other, "coverage_ratio", None)
            rows.append(
                {
                    "preset": key,
                    "preset_name": PLAN_PRESET_NAMES.get(key, key),
                    "total_hours": r2(getattr(other, "total_hours", 0.0)),
                    "weeks": int(getattr(other, "weeks", 0)),
                    "total_gain": r3(getattr(other, "total_gain", 0.0)),
                    "feasibility": r3(getattr(other, "feasibility", 0.0)),
                    "coverage_ratio": None if coverage_ratio is None else r3(coverage_ratio),
                    "objectives": {
                        "gain": r3(objectives.get("gain", 0.0)),
                        "time": r3(objectives.get("time", 0.0)),
                        "smoothness": r3(objectives.get("smoothness", 0.0)),
                        "fit": r3(objectives.get("fit", 0.0)),
                    },
                }
            )
        return rows

    def pareto(self, learner_id: str, preset: str = "balanced") -> dict[str, Any]:
        """帕累托前沿（时间—收益权衡散点）。"""
        self._require_learner(learner_id)
        preset = preset if preset in PLAN_PRESETS else "balanced"
        cache_key = (learner_id, preset)
        cached = self._pareto_cache.get(cache_key)
        if cached is not None:
            return cached

        model = self.learner_model(learner_id)
        front = list(self.planner.pareto_front(model) or [])

        preset_points: list[dict[str, Any]] = []
        for key in PLAN_PRESETS:
            plan = self._plan(learner_id, key)
            preset_points.append(self._pareto_point(plan, f"preset-{key}", key, key == preset))

        taken = {(round(p["hours"], 2), round(p["gain"], 3)) for p in preset_points}
        points: list[dict[str, Any]] = []
        for index, plan in enumerate(front, start=1):
            point = self._pareto_point(plan, f"p{index}", getattr(plan, "preset", "") or None, False)
            if (round(point["hours"], 2), round(point["gain"], 3)) in taken:
                continue
            points.append(point)
        points.extend(preset_points)

        payload = {
            "learner_id": learner_id,
            "x": "hours",
            "y": "gain",
            "points": points,
        }
        with self._lock:
            self._pareto_cache[cache_key] = payload
        return payload

    def _pareto_point(self, plan: Any, point_id: str, preset: str | None, selected: bool) -> dict[str, Any]:
        objectives = dict(getattr(plan, "objectives", None) or {})
        return {
            "id": point_id,
            "hours": r2(getattr(plan, "total_hours", 0.0)),
            "gain": r3(getattr(plan, "total_gain", 0.0)),
            "smoothness": r3(objectives.get("smoothness", 0.0)),
            "fit": r3(objectives.get("fit", 0.0)),
            "n_courses": len(getattr(plan, "steps", []) or []),
            "is_selected": bool(selected),
            "preset": preset or None,
            "weeks": int(getattr(plan, "weeks", 0)),
        }

    # ========================================================
    # 6. 自适应反馈
    # ========================================================
    def apply_feedback(
        self,
        learner_id: str,
        course_id: str,
        progress: float,
        quiz_score: float | None,
        rating: float | None,
        hours_spent: float | None,
        preset: str = "balanced",
    ) -> dict[str, Any]:
        """把一条学习反馈交给自适应引擎，并返回更新后的画像与重规划路径。"""
        learner = self._require_learner(learner_id)
        course = self._require_course(course_id)
        preset = preset if preset in PLAN_PRESETS else "balanced"

        engine = self.adaptive_engine  # 未就绪则抛 ModuleNotReady → 503
        outcome_cls = _import_attr("learnpath.adaptive.engine", "FeedbackOutcome")
        model = self.learner_model(learner_id)

        outcome = outcome_cls(
            learner_id=learner_id,
            course_id=course_id,
            progress=float(progress),
            quiz_score=None if quiz_score is None else float(quiz_score),
            rating=None if rating is None else float(rating),
            hours_spent=float(hours_spent) if hours_spent is not None else round(course.duration_hours * float(progress), 2),
        )

        with self._lock:
            result = engine.apply_feedback(model, outcome, preset=preset)
            # 掌握度已变 → 精确失效该学习者的派生缓存
            self._invalidate_learner(learner_id)

        new_path = getattr(result, "new_path", None)
        if new_path is not None:
            path_payload = self._path_payload(learner_id, new_path, preset)
            with self._lock:
                self._plan_object_cache[(learner_id, preset)] = new_path
                self._path_cache[(learner_id, preset)] = path_payload
        else:
            path_payload = self.path(learner_id, preset)

        return {
            "learner_id": learner_id,
            "course_id": course_id,
            "mastery_updates": {kp: r3(v) for kp, v in (getattr(result, "mastery_updates", None) or {}).items()},
            "delta": {kp: r3(v) for kp, v in (getattr(result, "delta", None) or {}).items()},
            "actions": list(getattr(result, "actions", None) or []),
            "replanned": bool(getattr(result, "replanned", False)),
            "reason": str(getattr(result, "reason", "")),
            "profile": self.profile_payload(learner, model),
            "path": path_payload,
        }

    def _invalidate_learner(self, learner_id: str) -> None:
        """清掉与某学习者相关的全部派生缓存。"""
        self._recommend_cache = {k: v for k, v in self._recommend_cache.items() if k[0] != learner_id}
        self._path_cache = {k: v for k, v in self._path_cache.items() if k[0] != learner_id}
        self._pareto_cache = {k: v for k, v in self._pareto_cache.items() if k[0] != learner_id}
        self._plan_object_cache = {k: v for k, v in self._plan_object_cache.items() if k[0] != learner_id}
        self._social_cache = {k: v for k, v in self._social_cache.items() if k[0] != learner_id}
        self._analytics_cache = {k: v for k, v in self._analytics_cache.items() if k[0] != learner_id}
        self._prediction_cache = {k: v for k, v in self._prediction_cache.items() if k[0] != learner_id}
        self._graph_cache = {k: v for k, v in self._graph_cache.items() if k[1] != learner_id}

    # ========================================================
    # 7. 社交学习
    # ========================================================
    def social(self, learner_id: str, k: int = 6) -> dict[str, Any]:
        """同伴匹配 + 学习小组 + 同期群摘要。"""
        self._require_learner(learner_id)
        cache_key = (learner_id, int(k))
        cached = self._social_cache.get(cache_key)
        if cached is not None:
            return cached

        matcher = self.matcher
        model = self.learner_model(learner_id)
        peers_raw = list(matcher.find_peers(model, k=int(k)))
        completed = self._completed_counts()
        dataset = self.dataset

        peers = []
        for peer in peers_raw:
            learner: Learner = peer.learner
            goal = dataset.goals.get(learner.goal_id)
            peers.append(
                {
                    "learner_id": learner.learner_id,
                    "nickname": learner.nickname,
                    "ability": r3(learner.ability),
                    "goal_name": goal.name if goal else "",
                    "similarity": r3(peer.similarity),
                    "shared_goal_kps": list(getattr(peer, "shared_goals", None) or []),
                    "shared_courses": list(getattr(peer, "shared_courses", None) or []),
                    "ahead_in": list(getattr(peer, "ahead_in", None) or []),
                    "completed_count": int(
                        getattr(peer, "completed_count", None)
                        if getattr(peer, "completed_count", None) is not None
                        else completed.get(learner.learner_id, 0)
                    ),
                }
            )

        # 以「知识缺口」为共识主题组队，缺口不足时退化为目标知识点
        snapshot = model.snapshot()
        topic_ids = list(snapshot.gaps)[: max(2, k)]
        if not topic_ids:
            topic_ids = list(snapshot.target_kps)[: max(2, k)]
        groups = []
        if topic_ids:
            for group in matcher.form_study_groups(topic_ids, size=4):
                groups.append(
                    {
                        "group_id": str(group.group_id),
                        "kp_names": self.graph.names_of(group.kp_ids),
                        "members": [
                            {
                                "learner_id": member.learner_id,
                                "nickname": member.nickname,
                                "ability": r3(member.ability),
                            }
                            for member in group.members
                        ],
                        "avg_completion": r3(group.avg_completion),
                        "synergy": r3(group.synergy),
                    }
                )

        peer_rates = [self._completion_rate_of(peer["learner_id"]) for peer in peers]
        payload = {
            "learner_id": learner_id,
            "peers": peers,
            "groups": groups,
            "cohort_summary": {
                "n_peers": len(peers),
                "avg_similarity": r3(_mean(peer["similarity"] for peer in peers)),
                "cohort_completion_rate": r3(_mean(peer_rates)),
                "your_completion_rate": self._completion_rate_of(learner_id),
            },
        }
        with self._lock:
            self._social_cache[cache_key] = payload
        return payload

    # ========================================================
    # 8. 学习分析仪表板
    # ========================================================
    def analytics(self, learner_id: str, preset: str = "balanced") -> dict[str, Any]:
        """仪表板聚合数据：交由 ``learnpath.analytics.dashboard.build_dashboard`` 计算。"""
        self._require_learner(learner_id)
        preset = preset if preset in PLAN_PRESETS else "balanced"
        cache_key = (learner_id, preset)
        cached = self._analytics_cache.get(cache_key)
        if cached is not None:
            return cached

        build_dashboard: Callable[..., dict[str, Any]] = _import_attr(
            "learnpath.analytics.dashboard", "build_dashboard"
        )
        model = self.learner_model(learner_id)
        plan = self._plan(learner_id, preset)

        payload = build_dashboard(self.dataset, self.graph, model, plan)
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.setdefault("learner_id", learner_id)
        else:  # 兼容返回非 dict 的实现
            raise ModuleNotReady("learnpath.analytics.dashboard", "仪表板模块返回了非字典结果")

        with self._lock:
            self._analytics_cache[cache_key] = payload
        return payload

    # ========================================================
    # 9. 知识图谱
    # ========================================================
    def knowledge_graph(self, domain: str | None = None, learner_id: str | None = None) -> dict[str, Any]:
        """ECharts graph 数据（可选领域过滤 + 学习者掌握度染色）。"""
        if domain and domain not in DOMAINS:
            raise NotFound(f"未知领域：{domain}")
        if learner_id is not None:
            self._require_learner(learner_id)

        cache_key = (domain, learner_id)
        cached = self._graph_cache.get(cache_key)
        if cached is not None:
            return cached

        graph = self.graph
        mastery: dict[str, float] | None = None
        target_kps: list[str] = []
        path_kps: list[str] = []

        if learner_id is not None:
            try:
                model = self.learner_model(learner_id)
                snapshot = model.snapshot()
                mastery = dict(snapshot.mastery)
                target_kps = list(snapshot.target_kps)
            except ModuleNotReady:
                mastery = None
            try:
                plan = self._plan(learner_id, "balanced")
                for step in getattr(plan, "steps", []) or []:
                    path_kps.extend(list(step.kp_ids or step.course.kp_ids))
            except ModuleNotReady:
                path_kps = []

        payload = graph.to_echarts(domain=domain, highlight=target_kps, mastery=mastery)
        payload["focus"] = {
            "target_kps": target_kps,
            "path_kps": sorted(set(path_kps)),
        }

        with self._lock:
            self._graph_cache[cache_key] = payload
        return payload

    # ========================================================
    # 10. 系统评测
    # ========================================================
    def evaluation(self) -> dict[str, Any]:
        """离线评测报告：直接返回 ``build_evaluation`` 的结果（不加工字段）。"""
        if self._evaluation_cache is not None:
            return self._evaluation_cache

        build_evaluation: Callable[..., dict[str, Any]] = _import_attr(
            "learnpath.analytics.evaluation", "build_evaluation"
        )
        payload = build_evaluation(self.dataset, self.graph)
        if not isinstance(payload, Mapping):
            raise ModuleNotReady("learnpath.analytics.evaluation", "评测模块返回了非字典结果")
        payload = dict(payload)
        with self._lock:
            self._evaluation_cache = payload
        return payload

    # ========================================================
    # 分页
    # ========================================================
    @staticmethod
    def _paginate(
        rows: list[Any],
        page: int,
        page_size: int,
        mapper: Callable[[Any], dict[str, Any]],
    ) -> dict[str, Any]:
        """通用分页：``{"items", "total", "page", "page_size"}``。"""
        page = max(1, int(page))
        page_size = max(1, min(200, int(page_size)))
        start = (page - 1) * page_size
        chunk = rows[start : start + page_size]
        return {
            "items": [mapper(row) for row in chunk],
            "total": len(rows),
            "page": page,
            "page_size": page_size,
        }


# ============================================================
# 进程级单例
# ============================================================

_container: ServiceContainer | None = None
_container_lock = threading.Lock()


def get_services() -> ServiceContainer:
    """获取进程级服务容器（首次调用时创建）。"""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = ServiceContainer()
    return _container


def reset_services() -> None:
    """丢弃当前容器（测试中用于强制重新加载数据 / 重新训练）。"""
    global _container
    with _container_lock:
        _container = None


__all__ = [
    "VERSION",
    "NotFound",
    "ModuleNotReady",
    "ServiceContainer",
    "get_services",
    "reset_services",
]
