"""学习效果预测（实现要求第 5 条）。

对外契约：

    # learnpath/prediction/features.py
    def build_features(dataset: Dataset, graph: KnowledgeGraph,
                       model: LearnerModel, course: Course) -> dict[str, float]

    # learnpath/prediction/effect.py
    @dataclass
    class CompletionPrediction:
        p_completion: float
        p_drop: float
        risk_level: str            # 低 / 中 / 高
        drivers: list[tuple[str, float]]   # 影响力最大的特征（可解释）

    @dataclass
    class GainPrediction:
        expected_gain: float                     # 课程结束后掌握度平均提升
        expected_kp_mastery: dict[str, float]    # 每个知识点预测掌握度

    class EffectPredictor:
        def __init__(self, model_path: Path | None = None) -> None
        @property
        def metrics(self) -> dict[str, float]    # auc / accuracy / f1 / mae / r2
        def fit(self, dataset: Dataset, graph: KnowledgeGraph) -> dict[str, float]
        def load(self) -> None
        def save(self) -> None
        def predict_completion(self, dataset, graph, model: LearnerModel,
                               course: Course) -> CompletionPrediction
        def predict_gain(self, dataset, graph, model: LearnerModel,
                         course: Course) -> GainPrediction

要求：完成概率用分类模型（逻辑回归 + 梯度提升树对比，5 折交叉验证），
掌握度增益用回归模型；指标写入 ``models/model_metrics.json``（入库），
二进制约文件写入 ``models/*.joblib``（不入库，可复现）。
"""
