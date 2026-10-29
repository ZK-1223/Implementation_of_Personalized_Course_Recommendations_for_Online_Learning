"""社交学习（实现要求第 7 条）。

对外契约：

    # learnpath/social/cohort.py
    @dataclass
    class PeerMatch:
        learner: Learner
        similarity: float            # 0~1
        shared_goals: list[str]      # 共同目标知识点名称
        shared_courses: list[str]    # 共同学过/在学的课程标题
        ahead_in: list[str]          # 对方更擅长的知识点名称

    @dataclass
    class StudyGroup:
        group_id: str
        kp_ids: list[str]
        members: list[Learner]
        avg_completion: float
        synergy: float               # 互补性得分

    class SocialMatcher:
        def __init__(self, dataset: Dataset, graph: KnowledgeGraph) -> None
        def find_peers(self, model: LearnerModel,
                       k: int = PEER_TOP_K) -> list[PeerMatch]
        def form_study_groups(self, kp_ids: Iterable[str],
                              size: int = STUDY_GROUP_SIZE) -> list[StudyGroup]
        def peer_signal(self, model: LearnerModel, course: Course) -> float
"""
