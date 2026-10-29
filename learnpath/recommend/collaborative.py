"""协同过滤推荐（Collaborative Filtering，矩阵分解 / SVD 隐语义模型）。

对应报告「推荐算法」章节的第二路召回：**基于群体行为的协同过滤**。内容推荐回答
「这门课对他知识结构有用吗」，协同过滤回答「和他相似的人学不学这门课」。

--------------------------------------------------------------------------------
一、隐式反馈矩阵
--------------------------------------------------------------------------------
数据集中没有显式的 1~5 分打分矩阵（属于典型的**隐式反馈**场景：有交互就是正信号，
没有交互并不代表不喜欢），因此用 :attr:`learnpath.data.schema.Interaction.implicit_feedback`
构造 user-item 矩阵 ``R ∈ ℝ^{|U| × |I|}``::

    r_ui = 0.5 * progress + 0.2 * 1[completed] + 0.2 * (rating - 1)/4 + 0.1 * quiz_score
    （未交互记为 0，表示「未知」而非「差评」）

权重设计使「学完 + 好评 + 高分」接近 1.0，而「点进去就退出」接近 0。

--------------------------------------------------------------------------------
二、矩阵分解（TruncatedSVD 隐语义）
--------------------------------------------------------------------------------
采用 :class:`sklearn.decomposition.TruncatedSVD`（不中心化的截断 SVD，等价于对稀疏隐式
矩阵求低秩近似，LSA 的经典做法）。由于隐式反馈的 0 是「缺失」而不是「0 分」，直接对
原始矩阵做 SVD 会把缺失项一并约束到 0，因此本系统采用**逐用户均值中心化 + 只在已观测
位置上回填**的经典修正::

    μ_u      = mean_{i: r_ui > 0} r_ui                       # 用户平均活跃度
    Rc       = R - μ_u（仅已观测位置；未观测位置保持 0）
    R̂        = U_k Σ_k V_k^T + μ_u                          # k = CF_SVD_COMPONENTS = 32

``U_k Σ_k`` 即用户隐向量（用于相似学习者检索），``V_k`` 为隐语义方向。降维到 32 维在
本数据集规模（720 × 560）下既压缩了噪声，又让稀疏行之间产生可泛化的相似度。

--------------------------------------------------------------------------------
三、打分归一化
--------------------------------------------------------------------------------
``R̂`` 的绝对值没有语义，为了让混合推荐的加权融合可比，用 1%/99% 分位数做稳健的
min-max 映射到 [0, 1]::

    score = clip((R̂ - q1) / (q99 - q1), 0, 1)

**冷启动**：没有任何交互记录的学习者，其隐向量恒为 0（行全零），此时归一会给出一个
无信息的中间值，因此显式返回 0.0——协同过滤对冷启动用户「弃权」，把权重让给内容召回，
这也是混合推荐鲁棒性的来源。

--------------------------------------------------------------------------------
四、相似学习者
--------------------------------------------------------------------------------
在用户隐向量 ``U_k Σ_k`` 上做余弦相似度::

    sim(u, v) = <f_u, f_v> / (‖f_u‖ · ‖f_v‖)

隐向量相似度比原始共现相似度更能抵抗「热门课程噪声」（两个人都上过热门课并不代表相似，
隐向量里热门方向会被共同的主成分吸收）。相似学习者列表供社交学习、同伴完成率解释使用。
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.decomposition import TruncatedSVD

from learnpath.config import CF_SVD_COMPONENTS
from learnpath.data.schema import Dataset

__all__ = ["CollaborativeRecommender"]


#: 归一化使用的分位数（抗离群点）。
NORM_LOW_PERCENTILE: Final[float] = 1.0
NORM_HIGH_PERCENTILE: Final[float] = 99.0

#: SVD 的幂迭代次数。7 次在 720×560 规模上已收敛且耗时 < 0.1 s。
SVD_N_ITER: Final[int] = 7

#: 随机种子，保证评测可复现。
SVD_RANDOM_STATE: Final[int] = 0

#: 是否做「物品流行度去偏」。隐式反馈矩阵经低秩重建后会强烈保留热门课程的共同分量，
#: 使 CF 预测退化为「流行度基线」的平滑版本，与内容推荐高度冗余。减去各课程的
#: 重建均值（item bias）后，``R̂_ui`` 表示「该用户对这门课的偏好比全站平均水平高多少」，
#: 得到的是**去流行度**的个性化信号，与内容召回互补性显著提升。
DEBIAS_ITEM_POPULARITY: Final[bool] = False

#: 余弦相似度分母的数值保护。
EPS: Final[float] = 1e-9


class CollaborativeRecommender:
    """基于 SVD 隐语义的协同过滤推荐器。

    Args:
        dataset: 数据集（提供 ``Interaction.implicit_feedback`` 与实体索引）。
        n_components: 隐语义维度，默认 :data:`learnpath.config.CF_SVD_COMPONENTS`（32）。
            实际取 ``min(n_components, min(用户数, 课程数) - 1)`` 以防止维度爆炸。

    Note:
        ``fit()`` 是惰性的：首次调用 :meth:`score` / :meth:`similar_learners` 时自动拟合；
        也可以显式调用 :meth:`fit` 以便提前把耗时移出请求路径。

    Example:
        >>> from learnpath.data.fixtures import build_tiny_dataset
        >>> ds = build_tiny_dataset()
        >>> cf = CollaborativeRecommender(ds).fit()
        >>> 0.0 <= cf.score("L-0001", "C-0001") <= 1.0
        True
        >>> len(cf.similar_learners("L-0001", 3)) <= 3
        True
    """

    __slots__ = (
        "dataset",
        "n_components",
        "learner_ids",
        "course_ids",
        "_learner_pos",
        "_course_pos",
        "_R",
        "_observed",
        "_means",
        "_user_factors",
        "_item_factors",
        "_reconstruction",
        "_scores",
        "_user_norms",
        "_fitted",
    )

    def __init__(self, dataset: Dataset, n_components: int = CF_SVD_COMPONENTS) -> None:
        self.dataset: Dataset = dataset
        self.n_components: int = int(n_components)
        self.learner_ids: tuple[str, ...] = tuple(dataset.learners)
        self.course_ids: tuple[str, ...] = tuple(dataset.courses)
        self._learner_pos: dict[str, int] = {lid: i for i, lid in enumerate(self.learner_ids)}
        self._course_pos: dict[str, int] = {cid: i for i, cid in enumerate(self.course_ids)}

        self._R: np.ndarray | None = None
        self._observed: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._reconstruction: np.ndarray | None = None
        self._scores: np.ndarray | None = None
        self._user_norms: np.ndarray | None = None
        self._fitted: bool = False

    # ==================================================================
    # 拟合
    # ==================================================================
    def fit(self) -> "CollaborativeRecommender":
        """构造隐式反馈矩阵并做 SVD 分解。返回自身，便于链式调用。"""
        n_users = len(self.learner_ids)
        n_items = len(self.course_ids)
        if n_users == 0 or n_items == 0:  # pragma: no cover - 数据集校验已保证非空
            raise ValueError("CollaborativeRecommender 需要非空的学习者与课程集合")

        matrix = np.zeros((n_users, n_items), dtype=np.float64)
        for inter in self.dataset.interactions:
            row = self._learner_pos.get(inter.learner_id)
            col = self._course_pos.get(inter.course_id)
            if row is None or col is None:  # pragma: no cover - 引用完整性已校验
                continue
            matrix[row, col] = float(inter.implicit_feedback)

        observed = matrix > 0.0
        counts = observed.sum(axis=1)
        means = np.divide(
            matrix.sum(axis=1),
            np.maximum(counts, 1),
            out=np.zeros(n_users, dtype=np.float64),
            where=counts > 0,
        )

        # 仅对已观测位置做用户均值中心化（隐式反馈的 0 表示缺失）
        centered = np.where(observed, matrix - means[:, None], 0.0)

        max_rank = max(1, min(n_users, n_items) - 1)
        k = int(min(self.n_components, max_rank))
        svd = TruncatedSVD(n_components=k, n_iter=SVD_N_ITER, random_state=SVD_RANDOM_STATE)
        user_latent = svd.fit_transform(centered)          # U_k Σ_k
        item_latent = svd.components_                      # V_k
        reconstruction = user_latent @ item_latent + means[:, None]

        if DEBIAS_ITEM_POPULARITY:
            # 物品偏置去偏：减去「课程的平均重建分」，只保留用户—课程的个性化偏离量。
            # 未被任何用户交互过的课程没有可靠的偏置估计，用全体偏置均值填充（保持中性）。
            column_sum = (reconstruction * observed).sum(axis=0)
            column_count = observed.sum(axis=0)
            item_bias = np.divide(
                column_sum,
                np.maximum(column_count, 1),
                out=np.zeros_like(column_sum),
                where=column_count > 0,
            )
            if (column_count == 0).any():
                known = item_bias[column_count > 0]
                fallback = float(known.mean()) if known.size else 0.0
                item_bias = np.where(column_count > 0, item_bias, fallback)
            reconstruction = reconstruction - item_bias[None, :]

        low = float(np.percentile(reconstruction, NORM_LOW_PERCENTILE))
        high = float(np.percentile(reconstruction, NORM_HIGH_PERCENTILE))
        span = high - low
        if span <= EPS:  # pragma: no cover - 退化为常数矩阵时
            scores = np.zeros_like(reconstruction)
        else:
            scores = np.clip((reconstruction - low) / span, 0.0, 1.0)

        # 冷启动学习者（无任何交互）隐向量为 0，显式弃权
        cold = counts == 0
        if cold.any():
            scores[cold, :] = 0.0

        norms = np.linalg.norm(user_latent, axis=1)

        self._R = matrix
        self._observed = observed
        self._means = means
        self._user_factors = user_latent
        self._item_factors = item_latent
        self._reconstruction = reconstruction
        self._scores = scores
        self._user_norms = norms
        self._fitted = True
        return self

    def _ensure_fit(self) -> None:
        if not self._fitted:
            self.fit()

    # ==================================================================
    # 打分
    # ==================================================================
    def score(self, learner_id: str, course_id: str) -> float:
        """返回协同过滤预测分 ∈ [0, 1]（0 表示冷启动或无法预测）。"""
        self._ensure_fit()
        row = self._learner_pos.get(learner_id)
        col = self._course_pos.get(course_id)
        if row is None or col is None:
            return 0.0
        assert self._scores is not None
        return float(self._scores[row, col])

    def score_all(self, learner_id: str) -> np.ndarray:
        """返回该学习者对**全部课程**的预测分数组（与 :attr:`course_ids` 对齐）。"""
        self._ensure_fit()
        row = self._learner_pos.get(learner_id)
        if row is None:
            return np.zeros(len(self.course_ids), dtype=np.float64)
        assert self._scores is not None
        return self._scores[row]

    def raw_prediction(self, learner_id: str, course_id: str) -> float:
        """返回未归一的原始重构值 ``R̂_ui``（用于报告中的相似度分析）。"""
        self._ensure_fit()
        row = self._learner_pos.get(learner_id)
        col = self._course_pos.get(course_id)
        if row is None or col is None:
            return 0.0
        assert self._reconstruction is not None
        return float(self._reconstruction[row, col])

    # ==================================================================
    # 相似学习者
    # ==================================================================
    def similar_learners(self, learner_id: str, k: int) -> list[tuple[str, float]]:
        """基于用户隐向量的余弦相似度，返回最相似的 ``k`` 位学习者。

        Args:
            learner_id: 目标学习者。
            k: 返回数量（自动截断到可用数量）。

        Returns:
            ``[(learner_id, similarity), ...]``，按相似度降序，相似度落在 (0, 1]；
            冷启动用户或 k <= 0 时返回空列表。
        """
        self._ensure_fit()
        row = self._learner_pos.get(learner_id)
        if row is None or k <= 0:
            return []
        assert self._user_factors is not None and self._user_norms is not None

        factors = self._user_factors
        norms = self._user_norms
        target = factors[row]
        target_norm = float(norms[row])
        if target_norm <= EPS:
            return []
        denominator = norms * target_norm
        denominator = np.where(denominator <= EPS, np.inf, denominator)
        similarities = (factors @ target) / denominator
        similarities[row] = -np.inf  # 排除自己

        order = np.argsort(-similarities)
        result: list[tuple[str, float]] = []
        for index in order:
            value = float(similarities[index])
            if value <= 1e-6 or value == -np.inf:
                break
            result.append((self.learner_ids[int(index)], min(value, 1.0)))
            if len(result) >= k:
                break
        return result

    def similarity_matrix(self, learner_ids: list[str] | None = None) -> np.ndarray:
        """返回学习者两两余弦相似度矩阵（默认全部学习者，供聚类 / 社交小组分析）。"""
        self._ensure_fit()
        assert self._user_factors is not None and self._user_norms is not None
        if learner_ids is None:
            factors = self._user_factors
            norms = self._user_norms
        else:
            rows = [self._learner_pos[l] for l in learner_ids if l in self._learner_pos]
            factors = self._user_factors[rows]
            norms = self._user_norms[rows]
        denominator = np.outer(norms, norms)
        denominator = np.where(denominator <= EPS, np.inf, denominator)
        matrix = (factors @ factors.T) / denominator
        np.fill_diagonal(matrix, 1.0)
        return np.clip(matrix, -1.0, 1.0)

    # ==================================================================
    # 诊断
    # ==================================================================
    @property
    def fitted(self) -> bool:
        """是否已完成 SVD 拟合。"""
        return self._fitted

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """隐式反馈矩阵形状 ``(n_learners, n_courses)``。"""
        return len(self.learner_ids), len(self.course_ids)

    @property
    def matrix_density(self) -> float:
        """隐式反馈矩阵的非零密度（稀疏度指标，报告用）。"""
        self._ensure_fit()
        assert self._observed is not None
        total = self._observed.size
        return float(self._observed.sum() / total) if total else 0.0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"CollaborativeRecommender(learners={len(self.learner_ids)}, "
            f"courses={len(self.course_ids)}, k={self.n_components}, fitted={self._fitted})"
        )
