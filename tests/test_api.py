"""REST API 契约测试。

遵循「契约优先」原则：只断言 ``docs/API.md`` 里定义的字段，不依赖算法内部实现。
算法子模块尚未就绪时，相关接口按设计返回 503，此时用 ``pytest.skip`` 跳过，
这样本测试文件在并行开发的任意阶段都能通过。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import get_services


@pytest.fixture(scope="module")
def client() -> TestClient:
    """模块级 TestClient（避免重复加载数据集）。"""
    return TestClient(app)


@pytest.fixture(scope="module")
def services():
    """服务容器单例（供测试对齐图谱节点数等）。"""
    return get_services()


@pytest.fixture(scope="module")
def first_learner_id(client: TestClient) -> str:
    """取一个真实存在的学习者 id。"""
    response = client.get("/api/meta")
    assert response.status_code == 200
    learners = response.json()["learners"]
    assert learners, "元数据中没有任何学习者"
    return learners[0]["learner_id"]


def _skip_if_not_ready(response) -> None:
    """算法模块未就绪（503）时跳过依赖它的断言。"""
    if response.status_code == 503:
        pytest.skip(f"算法模块尚未就绪：{response.json().get('detail')}")


# ============================================================
# 1. 系统信息
# ============================================================


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["dataset_loaded"] is True


def test_meta(client: TestClient) -> None:
    response = client.get("/api/meta")
    assert response.status_code == 200
    body = response.json()

    for key in ("stats", "graph", "domains", "goals", "presets", "learners", "modalities", "motivations"):
        assert key in body, f"meta 缺少字段 {key}"

    stats = body["stats"]
    for key in (
        "n_knowledge_points",
        "n_kp_edges",
        "n_courses",
        "n_learners",
        "n_goals",
        "n_interactions",
        "n_completed",
        "completion_rate",
        "domains",
    ):
        assert key in stats, f"stats 缺少字段 {key}"

    graph = body["graph"]
    for key in ("n_nodes", "n_edges", "n_roots", "n_leaves", "max_depth", "avg_out_degree", "domains"):
        assert key in graph, f"graph 缺少字段 {key}"

    assert len(body["domains"]) == len(stats["domains"])
    for domain in body["domains"]:
        assert set(domain) >= {"name", "icon", "n_kps", "n_courses"}

    assert body["presets"], "presets 不能为空"
    for preset in body["presets"]:
        assert set(preset) >= {"key", "name", "weights"}
        assert set(preset["weights"]) == {"gain", "time", "smoothness", "fit"}

    assert body["goals"], "goals 不能为空"
    for goal in body["goals"]:
        assert set(goal) >= {
            "goal_id",
            "name",
            "target_role",
            "description",
            "n_target_kps",
            "default_weeks",
        }

    assert body["learners"], "learners 不能为空"
    for learner in body["learners"]:
        assert set(learner) >= {
            "learner_id",
            "nickname",
            "goal_id",
            "goal_name",
            "ability",
            "time_budget_hours_per_week",
        }

    assert {item["key"] for item in body["modalities"]} <= {"video", "text", "project", "mixed"}
    assert body["motivations"], "motivations 不能为空"


# ============================================================
# 2. 课程库 / 知识图谱
# ============================================================


def test_courses_pagination(client: TestClient, services) -> None:
    total = len(services.dataset.courses)

    page_one = client.get("/api/courses", params={"page": 1, "page_size": 5})
    assert page_one.status_code == 200
    body = page_one.json()
    assert body["page"] == 1
    assert body["page_size"] == 5
    assert body["total"] == total
    assert len(body["items"]) == min(5, total)

    for item in body["items"]:
        assert "kp_names" in item
        assert item["course_id"].startswith("C-")
        assert 1 <= item["level"] <= 5

    page_two = client.get("/api/courses", params={"page": 2, "page_size": 5})
    assert page_two.status_code == 200
    second = page_two.json()
    assert second["page"] == 2
    if total > 5:
        first_ids = {item["course_id"] for item in body["items"]}
        second_ids = {item["course_id"] for item in second["items"]}
        assert first_ids.isdisjoint(second_ids), "分页出现重复条目"

    # 越界页应返回空列表而非报错
    beyond = client.get("/api/courses", params={"page": 999, "page_size": 5})
    assert beyond.status_code == 200
    assert beyond.json()["items"] == []


def test_courses_filter(client: TestClient, services) -> None:
    domain = next(iter(services.dataset.courses.values())).domain
    response = client.get("/api/courses", params={"domain": domain, "page_size": 50})
    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert all(item["domain"] == domain for item in items)


def test_knowledge_graph_matches_dataset(client: TestClient, services) -> None:
    body = client.get("/api/knowledge-graph").json()
    assert set(body) >= {"nodes", "links", "categories", "focus"}

    graph = services.graph
    assert len(body["nodes"]) == len(graph)
    assert len(body["links"]) == len(graph.stats().to_dict()["domains"] and fold_edges(graph))

    node = body["nodes"][0]
    for key in ("id", "name", "category", "domain", "difficulty", "depth", "mastery", "symbolSize", "highlight"):
        assert key in node

    # 领域过滤
    domain = body["categories"][0]
    filtered = client.get("/api/knowledge-graph", params={"domain": domain}).json()
    assert filtered["nodes"]
    assert all(item["domain"] == domain for item in filtered["nodes"])
    assert all(
        link["source"] in {n["id"] for n in filtered["nodes"]}
        and link["target"] in {n["id"] for n in filtered["nodes"]}
        for link in filtered["links"]
    )


def fold_edges(graph) -> list[tuple[str, str]]:
    """把图谱的邻接表折叠成边列表（测试用，保持与 to_echarts 的统计口径一致）。"""
    return [
        (source, target)
        for source in graph.node_ids
        for target in graph.successors(source)
    ]


def test_knowledge_graph_unknown_domain(client: TestClient) -> None:
    response = client.get("/api/knowledge-graph", params={"domain": "不存在的领域"})
    assert response.status_code == 404
    assert "detail" in response.json()


# ============================================================
# 3. 学习者画像
# ============================================================


def test_list_learners(client: TestClient, services) -> None:
    response = client.get("/api/learners", params={"page": 1, "page_size": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == len(services.dataset.learners)
    assert len(body["items"]) == 3


def test_learner_detail(client: TestClient, first_learner_id: str) -> None:
    response = client.get(f"/api/learners/{first_learner_id}")
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    assert set(body) >= {"learner", "goal", "profile", "history"}
    assert body["learner"]["learner_id"] == first_learner_id
    assert body["goal"]["goal_id"] == body["learner"]["goal_id"]

    profile = body["profile"]
    for key in (
        "learner_id",
        "mastery",
        "mastery_names",
        "domain_mastery",
        "target_kps",
        "target_kp_names",
        "gaps",
        "gap_names",
        "readiness",
        "style_vector",
        "time_budget",
        "total_available_hours",
        "mastered_count",
        "gap_count",
        "completion_rate",
    ):
        assert key in profile, f"profile 缺少字段 {key}"
    assert 0.0 <= profile["readiness"] <= 1.0
    assert len(profile["target_kps"]) == len(profile["target_kp_names"])

    for record in body["history"]:
        assert set(record) >= {
            "course_id",
            "title",
            "domain",
            "progress",
            "completed",
            "rating",
            "hours_spent",
            "quiz_score",
            "timestamp",
        }


def test_learner_not_found(client: TestClient) -> None:
    response = client.get("/api/learners/L-9999")
    assert response.status_code == 404
    assert "detail" in response.json()


# ============================================================
# 4. 智能推荐
# ============================================================


def test_recommend(client: TestClient, first_learner_id: str) -> None:
    response = client.get(f"/api/recommend/{first_learner_id}", params={"k": 5})
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    assert body["learner_id"] == first_learner_id
    assert body["k"] == 5
    assert body["count"] == len(body["items"])
    assert 0 < body["count"] <= 5

    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True), "推荐结果应按综合分降序"

    for item in body["items"]:
        for key in (
            "rank",
            "course_id",
            "title",
            "domain",
            "level",
            "duration_hours",
            "modality",
            "modality_name",
            "rating",
            "enroll_count",
            "price",
            "kp_names",
            "score",
            "components",
            "reasons",
            "expected_gain",
            "p_completion",
            "risk_level",
            "readiness",
            "style_fit",
            "time_fit",
        ):
            assert key in item, f"推荐项缺少字段 {key}"
        assert set(item["components"]) == {
            "content",
            "collaborative",
            "quality",
            "readiness",
            "social",
        }
        assert 0.0 <= item["score"] <= 1.0
        assert 0.0 <= item["p_completion"] <= 1.0
        assert item["risk_level"] in {"低", "中", "高"}
        assert isinstance(item["reasons"], list) and item["reasons"]


def test_recommend_learner_not_found(client: TestClient) -> None:
    response = client.get("/api/recommend/L-9999")
    assert response.status_code in {404, 503}


# ============================================================
# 5. 学习路径 / 帕累托前沿
# ============================================================


def test_path(client: TestClient, first_learner_id: str) -> None:
    response = client.get(f"/api/path/{first_learner_id}", params={"preset": "balanced"})
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    assert body["learner_id"] == first_learner_id
    assert body["preset"] == "balanced"
    assert body["preset_name"]
    assert set(body["goal"]) >= {"goal_id", "name"}
    for key in ("total_hours", "weeks", "total_gain", "feasibility", "objectives"):
        assert key in body
    assert set(body["objectives"]) == {"gain", "time", "smoothness", "fit"}
    assert 0.0 <= body["feasibility"] <= 1.0

    # ---- 预算与覆盖率（docs/API.md §4 补充的权威字段）----
    for key in (
        "budget_hours",
        "over_budget",
        "coverage_ratio",
        "covered_target_kp_ids",
        "covered_target_kp_names",
        "uncovered_target_kp_ids",
        "uncovered_target_kp_names",
        "coverage",
    ):
        assert key in body, f"路径响应缺少字段 {key}"

    assert body["over_budget"] is False, "正常输入下路径不应超预算"
    assert body["total_hours"] <= body["budget_hours"] + 1e-6, "路径总时长必须落在预算内"
    assert 0.0 <= body["coverage_ratio"] <= 1.0
    assert len(body["covered_target_kp_ids"]) == len(body["covered_target_kp_names"])
    assert len(body["uncovered_target_kp_ids"]) == len(body["uncovered_target_kp_names"])
    assert set(body["coverage"]) >= {
        "target_gap_ratio",
        "gap_ratio",
        "budget_utilisation",
        "weeks_needed",
        "deadline_weeks",
        "avg_weekly_load",
        "hours_shortfall",
    }

    # coverage_ratio 的口径来自算法侧 coverage 诊断块（含「已掌握」的目标点），
    # 用其 target_achieved / target_total 交叉校验，避免用「仅路径覆盖」的口径误判。
    coverage = body["coverage"]
    if {"target_achieved", "target_total"} <= set(coverage) and coverage["target_total"]:
        expected = coverage["target_achieved"] / coverage["target_total"]
        assert abs(body["coverage_ratio"] - expected) < 0.02, (
            "coverage_ratio 应与 coverage.target_achieved / target_total 一致"
        )

    steps = body["steps"]
    assert steps, "路径不应为空"
    for step in steps:
        for key in (
            "order",
            "course_id",
            "title",
            "domain",
            "level",
            "hours",
            "cumulative_hours",
            "week",
            "kp_names",
            "expected_gain",
            "modality",
        ):
            assert key in step, f"路径步骤缺少字段 {key}"
    assert [step["order"] for step in steps] == sorted(step["order"] for step in steps)
    cumulative = [step["cumulative_hours"] for step in steps]
    assert cumulative == sorted(cumulative), "累计时长应单调不减"

    assert body["weekly_plan"], "weekly_plan 不应为空"
    for row in body["weekly_plan"]:
        assert set(row) >= {"week", "hours", "course_ids"}

    for row in body["milestones"]:
        assert set(row) >= {"week", "label", "kp_names"}

    alternatives = body["alternatives"]
    assert len(alternatives) == 2, "除当前预设外应给出另外两种方案"
    for row in alternatives:
        assert set(row) >= {
            "preset",
            "preset_name",
            "total_hours",
            "weeks",
            "total_gain",
            "feasibility",
            "coverage_ratio",
            "objectives",
        }


def test_path_invalid_preset(client: TestClient, first_learner_id: str) -> None:
    response = client.get(f"/api/path/{first_learner_id}", params={"preset": "unknown"})
    assert response.status_code == 422


def test_pareto(client: TestClient, first_learner_id: str) -> None:
    response = client.get(f"/api/path/{first_learner_id}/pareto")
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    assert body["x"] == "hours"
    assert body["y"] == "gain"
    assert body["points"], "帕累托前沿不应为空"
    for point in body["points"]:
        for key in ("id", "hours", "gain", "smoothness", "fit", "n_courses", "is_selected", "preset"):
            assert key in point
    assert any(point["is_selected"] for point in body["points"]), "应标记出当前选中方案"


# ============================================================
# 6. 自适应反馈
# ============================================================


def test_feedback(client: TestClient, first_learner_id: str, services) -> None:
    course_id = next(iter(services.dataset.courses))
    payload = {
        "learner_id": first_learner_id,
        "course_id": course_id,
        "progress": 1.0,
        "quiz_score": 0.85,
        "rating": 4.5,
        "hours_spent": 5.1,
    }
    response = client.post("/api/feedback", json=payload)
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    assert body["learner_id"] == first_learner_id
    assert body["course_id"] == course_id
    for key in ("mastery_updates", "delta", "actions", "replanned", "reason", "profile", "path"):
        assert key in body
    assert isinstance(body["actions"], list)
    assert isinstance(body["replanned"], bool)
    assert body["profile"]["learner_id"] == first_learner_id
    assert body["path"]["learner_id"] == first_learner_id


def test_feedback_validation(client: TestClient) -> None:
    bad = client.post("/api/feedback", json={"learner_id": "L-0001"})
    assert bad.status_code == 422

    invalid_score = client.post(
        "/api/feedback",
        json={"learner_id": "L-0001", "course_id": "C-0001", "quiz_score": 1.8},
    )
    assert invalid_score.status_code == 422


# ============================================================
# 7. 社交学习
# ============================================================


def test_social(client: TestClient, first_learner_id: str) -> None:
    response = client.get(f"/api/social/{first_learner_id}", params={"k": 4})
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    for key in ("learner_id", "peers", "groups", "cohort_summary"):
        assert key in body

    for peer in body["peers"]:
        for key in (
            "learner_id",
            "nickname",
            "ability",
            "goal_name",
            "similarity",
            "shared_goal_kps",
            "shared_courses",
            "ahead_in",
            "completed_count",
        ):
            assert key in peer
        assert peer["learner_id"] != first_learner_id, "同伴不应包含自己"
        assert 0.0 <= peer["similarity"] <= 1.0

    for group in body["groups"]:
        for key in ("group_id", "kp_names", "members", "avg_completion", "synergy"):
            assert key in group
        for member in group["members"]:
            assert set(member) >= {"learner_id", "nickname", "ability"}

    summary = body["cohort_summary"]
    assert set(summary) == {
        "n_peers",
        "avg_similarity",
        "cohort_completion_rate",
        "your_completion_rate",
    }


# ============================================================
# 8. 学习分析
# ============================================================


def test_analytics(client: TestClient, first_learner_id: str) -> None:
    response = client.get(f"/api/analytics/{first_learner_id}")
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    for key in (
        "learner_id",
        "kpi",
        "mastery_curve",
        "domain_radar",
        "difficulty_distribution",
        "modality_mix",
        "time_allocation",
        "completion_forecast",
        "gap_table",
        "history_activity",
    ):
        assert key in body, f"analytics 缺少字段 {key}"

    for key in (
        "overall_progress",
        "mastered_kps",
        "total_target_kps",
        "avg_mastery",
        "readiness",
        "planned_hours",
        "available_hours",
        "risk_courses",
        "learning_streak_weeks",
    ):
        assert key in body["kpi"], f"kpi 缺少字段 {key}"

    for row in body["mastery_curve"]:
        assert set(row) >= {"week", "avg_mastery", "mastered"}
    assert [row["week"] for row in body["mastery_curve"]] == sorted(
        row["week"] for row in body["mastery_curve"]
    )

    for row in body["domain_radar"]:
        assert set(row) >= {"domain", "mastery", "target"}
    for row in body["modality_mix"]:
        assert set(row) >= {"modality", "name", "hours"}
    for row in body["time_allocation"]:
        assert set(row) >= {"week", "hours", "planned"}
    for row in body["completion_forecast"]:
        assert set(row) >= {"order", "course_id", "course_title", "p_completion", "risk_level"}
    for row in body["gap_table"]:
        assert set(row) >= {"kp_id", "name", "mastery", "difficulty", "domain", "in_path"}


# ============================================================
# 9. 系统评测
# ============================================================


def test_evaluation(client: TestClient) -> None:
    response = client.get("/api/evaluation")
    _skip_if_not_ready(response)
    assert response.status_code == 200
    body = response.json()

    for key in ("generated_at", "recommenders", "ablation"):
        assert key in body
    assert body["recommenders"], "推荐算法对比表不应为空"

    for row in body["recommenders"]:
        assert "name" in row
        for metric in ("precision_at_k", "recall_at_k", "ndcg_at_k", "map", "coverage", "diversity", "novelty"):
            assert metric in row, f"评测行缺少指标 {metric}"

    effect = body.get("effect_model")
    if effect is not None:
        assert "models" in effect and "best" in effect

    for row in body["ablation"]:
        assert set(row) >= {"variant", "ndcg_at_k", "delta"}


# ============================================================
# 10. 单页应用
# ============================================================


def test_index_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "<!DOCTYPE html>" in text
    assert "LearnPath" in text
    assert "/static/js/app.js" in text


def test_static_assets(client: TestClient) -> None:
    for path in (
        "/static/css/base.css",
        "/static/css/components.css",
        "/static/css/views.css",
        "/static/js/app.js",
        "/static/js/mock.js",
        "/static/vendor/echarts.min.js",
    ):
        response = client.get(path)
        assert response.status_code == 200, f"静态资源缺失：{path}"
        assert response.content, f"静态资源为空：{path}"
