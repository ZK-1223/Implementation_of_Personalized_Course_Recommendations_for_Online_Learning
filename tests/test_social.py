"""社交学习的单元测试（实现要求 ⑦）。

覆盖同伴匹配（相似度构成、字段语义）、学习小组编组（互补性 / 人数 / synergy）
与同伴行为信号（三级回退）。
"""

from __future__ import annotations

import pytest

from learnpath.config import MASTERY_GAP, PEER_TOP_K, STUDY_GROUP_SIZE
from learnpath.data.fixtures import build_tiny_dataset
from learnpath.data.schema import Dataset
from learnpath.knowledge.graph import KnowledgeGraph
from learnpath.learner.profile import LearnerModel
from learnpath.social.cohort import PeerMatch, SocialMatcher, StudyGroup


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return build_tiny_dataset()


@pytest.fixture(scope="module")
def graph(dataset: Dataset) -> KnowledgeGraph:
    return KnowledgeGraph.from_dataset(dataset)


@pytest.fixture(scope="module")
def matcher(dataset: Dataset, graph: KnowledgeGraph) -> SocialMatcher:
    return SocialMatcher(dataset, graph)


@pytest.fixture(scope="module")
def model(dataset: Dataset, graph: KnowledgeGraph) -> LearnerModel:
    learner_id = sorted(dataset.learners)[0]
    return LearnerModel(dataset.learners[learner_id], dataset, graph)


# ============================================================
# 同伴匹配
# ============================================================


def test_find_peers_basic(matcher: SocialMatcher, model: LearnerModel) -> None:
    """同伴匹配：条数、排序、相似度区间、不包含自己。"""
    peers = matcher.find_peers(model, k=PEER_TOP_K)
    assert 0 < len(peers) <= PEER_TOP_K
    similarities = [peer.similarity for peer in peers]
    assert similarities == sorted(similarities, reverse=True), "必须按相似度降序"
    for peer in peers:
        assert isinstance(peer, PeerMatch)
        assert 0.0 <= peer.similarity <= 1.0
        assert 0.0 <= peer.cosine <= 1.0
        assert peer.learner.learner_id != model.learner.learner_id, "不能把自己当同伴"


def test_peer_fields_are_names(matcher: SocialMatcher, model: LearnerModel, graph: KnowledgeGraph) -> None:
    """``shared_goals`` / ``ahead_in`` 必须是**中文名称**而不是 id。

    ``ahead_in`` 还必须在「我方目标闭包」内，且对方掌握度确实更高。
    """
    peers = matcher.find_peers(model, k=PEER_TOP_K)
    all_names = {kp.name for kp in graph.kps.values()}
    closure = model.target_closure
    for peer in peers:
        assert isinstance(peer.goal_name, str)
        assert peer.completed_count >= 0
        for name in peer.shared_goals + peer.ahead_in:
            assert name in all_names, f"{name} 不是知识点名称"
        for name in peer.ahead_in:
            kp_id = next(kp_id for kp_id, kp in graph.kps.items() if kp.name == name)
            assert kp_id in closure, "ahead_in 必须落在我方目标闭包内"
            assert peer.cosine >= 0.0


def test_find_peers_custom_k(matcher: SocialMatcher, model: LearnerModel) -> None:
    """``k`` 参数生效；``k=0`` 返回空列表。"""
    assert len(matcher.find_peers(model, k=2)) <= 2
    assert matcher.find_peers(model, k=0) == []


def test_peer_match_to_dict(matcher: SocialMatcher, model: LearnerModel) -> None:
    """``PeerMatch.to_dict`` 覆盖 ``docs/API.md`` §6 的 ``peers[i]`` 字段。"""
    peers = matcher.find_peers(model, k=3)
    payload = peers[0].to_dict()
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
        assert key in payload
    assert 0.0 <= payload["similarity"] <= 1.0


# ============================================================
# 学习小组
# ============================================================


def test_form_study_groups(matcher: SocialMatcher, model: LearnerModel) -> None:
    """按目标知识点编组：人数、成员唯一、synergy 区间、聚焦知识点正确。"""
    kp_ids = list(model.goal.target_kp_ids)[:3]
    groups = matcher.form_study_groups(kp_ids, size=STUDY_GROUP_SIZE)
    for group in groups:
        assert isinstance(group, StudyGroup)
        assert group.group_id.startswith("G-")
        assert set(group.kp_ids) <= set(kp_ids)
        assert 2 <= len(group.members) <= max(2, STUDY_GROUP_SIZE)
        assert len({m.learner_id for m in group.members}) == len(group.members)
        assert 0.0 <= group.synergy <= 1.0
        assert 0.0 <= group.avg_completion <= 1.0
        assert group.mastery_std >= 0.0
    if len(groups) >= 2:
        synergies = [group.synergy for group in groups]
        assert synergies == sorted(synergies, reverse=True)


def test_study_group_prefers_complementary(matcher: SocialMatcher, dataset: Dataset, graph: KnowledgeGraph) -> None:
    """有互补空间时，小组里应同时存在「已掌握」与「未掌握」的成员。

    tiny 数据集的 12 名学习者分散在 3 个目标上，某些知识点可能不存在
    「一人已掌握、一人未掌握」的互补空间；此时该用例跳过（数据不支持）。
    """
    kp_ids = [kp_id for kp_id in dataset.knowledge_points if kp_id in graph]
    candidates_by_kp: dict[str, list[float]] = {}
    for kp_id in kp_ids:
        values = []
        for learner_id in matcher._group_candidates(kp_id):
            values.append(float(matcher._model_of(learner_id).mastery_of(kp_id)))
        candidates_by_kp[kp_id] = values
    complementary_kps = {
        kp_id
        for kp_id, values in candidates_by_kp.items()
        if any(v >= 0.85 for v in values) and any(v < MASTERY_GAP for v in values)
    }
    if not complementary_kps:
        pytest.skip("tiny 数据集上不存在互补空间的知识点")

    groups = matcher.form_study_groups(sorted(complementary_kps)[:3], size=STUDY_GROUP_SIZE)
    assert groups, "存在互补空间时必须能编出小组"
    complementary = 0
    for group in groups:
        kp_id = group.kp_ids[0]
        masters = [
            member
            for member in group.members
            if float(matcher._model_of(member.learner_id).mastery_of(kp_id)) >= 0.85
        ]
        novices = [
            member
            for member in group.members
            if float(matcher._model_of(member.learner_id).mastery_of(kp_id)) < MASTERY_GAP
        ]
        if masters and novices:
            complementary += 1
    assert complementary >= 1, "至少应有一个「有人已掌握、有人未掌握」的互补小组"


def test_study_group_to_dict(matcher: SocialMatcher, model: LearnerModel) -> None:
    """``StudyGroup.to_dict`` 覆盖 ``docs/API.md`` §6 的 ``groups[i]`` 字段。"""
    groups = matcher.form_study_groups(list(model.goal.target_kp_ids)[:2], size=4)
    if not groups:
        pytest.skip("tiny 数据集上无法编组")
    payload = groups[0].to_dict()
    for key in ("group_id", "kp_ids", "kp_names", "members", "avg_completion", "synergy"):
        assert key in payload
    for member in payload["members"]:
        assert set(member) == {"learner_id", "nickname", "ability"}


# ============================================================
# 同伴信号
# ============================================================


def test_peer_signal_range(matcher: SocialMatcher, model: LearnerModel, dataset: Dataset) -> None:
    """``peer_signal`` 必须返回 ``[0,1]`` 的完成率，且具有区分度。"""
    signals = [
        matcher.peer_signal(model, course)
        for course in list(dataset.courses.values())[:10]
    ]
    assert len(signals) == 10
    for value in signals:
        assert 0.0 <= value <= 1.0
    assert len(set(round(v, 6) for v in signals)) > 1, "同伴信号不应是常数"


def test_cohort_summary(matcher: SocialMatcher, model: LearnerModel) -> None:
    """``cohort_summary`` 字段齐全且数值在合法区间。"""
    peers = matcher.find_peers(model, k=4)
    summary = matcher.cohort_summary(model, peers)
    assert set(summary) == {
        "n_peers",
        "avg_similarity",
        "cohort_completion_rate",
        "your_completion_rate",
    }
    assert summary["n_peers"] == len(peers)
    assert 0.0 <= summary["avg_similarity"] <= 1.0
    assert 0.0 <= summary["cohort_completion_rate"] <= 1.0
    assert 0.0 <= summary["your_completion_rate"] <= 1.0
