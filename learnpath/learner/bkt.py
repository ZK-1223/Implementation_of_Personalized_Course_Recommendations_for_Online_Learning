"""贝叶斯知识追踪（Bayesian Knowledge Tracing, BKT）—— 掌握度推断内核。

本模块实现课程设计「实现要求 ①：学习者建模（知识水平）」中的**知识水平动态推断**部分，
是整个个性化推荐系统的「学习者状态方程」：所有推荐、路径规划、自适应重规划的收益
估计都建立在它输出的掌握度之上。

--------------------------------------------------------------------------------
一、模型定义
--------------------------------------------------------------------------------
BKT 是隐马尔可夫模型的一个特例：每个知识点 ``k`` 拥有一个二值隐状态
``L_t ∈ {已掌握, 未掌握}``，只能用「学习表现」间接观测。四参数
（见 :class:`learnpath.config.BKTParams`）为：

===============  ==========================================================
``p_init``       学习前已掌握该知识点的先验概率
``p_learn``      一次学习后从未掌握跃迁到已掌握的概率（学习转移率）
``p_slip``       已掌握却表现不达标的概率（失误率，slip）
``p_guess``      未掌握却表现达标的概率（蒙对率，guess）
===============  ==========================================================

转移方程（学习发生）::

    P(L_{t+1} = 1 | L_t = 0) = p_learn
    P(L_{t+1} = 1 | L_t = 1) = 1

观测方程（给定表现 ``obs``）::

    P(obs | L = 1) = (1 - p_slip)
    P(obs | L = 0) = p_guess

--------------------------------------------------------------------------------
二、本项目对经典 BKT 的两点工程化推广
--------------------------------------------------------------------------------
1. **软观测（soft evidence）**：经典 BKT 的观测是「答对 / 答错」二值，而本系统的学习表现
   是 ``score ∈ [0, 1]`` 的连续值（结课测验得分 / 完成进度折算）。我们把观测似然推广为
   对 ``score`` 的线性插值（等价于把 score 视为「答对比例」的软标签）::

       P(obs | L = 1) = score * (1 - p_slip) + (1 - score) * p_slip
       P(obs | L = 0) = score * p_guess     + (1 - score) * (1 - p_guess)

   当 ``score ∈ {0, 1}`` 时退化为经典二值 BKT。

2. **能力调制的学习率**：教学实践表明能力强的学习者单次学习的吸收率更高，因此实际学习率
   由学习者能力 ``ability ∈ [0, 1]`` 与知识点覆盖强度 ``kp_weight ∈ (0, 1]`` 共同调制::

       p_learn_eff = clip(p_learn * (0.55 + 0.90 * ability) * (0.50 + 0.50 * kp_weight),
                          1e-3, 0.95)

   其中 ``0.55 + 0.90 * ability`` 在 ``ability = 0.5`` 时恰为 1.0（以「中等能力」为基准，
   能力满分放大 1.45 倍，能力为零压缩到 0.55 倍）；``0.50 + 0.50 * kp_weight`` 表示
   「课程对该知识点的覆盖只有一半时，单次学习的有效摄入也打折」。

--------------------------------------------------------------------------------
三、一次完整更新（:meth:`MasteryTracker.update`）
--------------------------------------------------------------------------------
设先验 ``p = P(L_t = 1)``，观测 ``obs``：

1. **观测更新（贝叶斯后验）**::

       p_post = p * P(obs|L=1) / (p * P(obs|L=1) + (1-p) * P(obs|L=0))

2. **学习转移（`p_learn_eff`）**::

       p_next = p_post + (1 - p_post) * p_learn_eff

返回值即 ``p_next``，也就是**下一时刻的先验**——所以调用方不要对返回值再做一次衰减
或再套一层转移，直接把它当作新的掌握度即可。

--------------------------------------------------------------------------------
四、遗忘衰减（:meth:`MasteryTracker.decay`）
--------------------------------------------------------------------------------
``p_new = p * MASTERY_DECAY_PER_WEEK ** idle_weeks``（见
:data:`learnpath.config.MASTERY_DECAY_PER_WEEK` = 0.985）。取指数衰减而非线性衰减的理由：
遗忘曲线（Ebbinghaus）在半对数坐标上近似线性，指数衰减最贴合；0.985/周意味着 12 周不复习
保留 0.985^12 ≈ 83.4%，一年保留约 45.7%，量级与教育测量文献中的「技能折旧率」一致。
"""

from __future__ import annotations

from typing import Final

from learnpath.config import BKT, MASTERY_DECAY_PER_WEEK, BKTParams

__all__ = ["MasteryTracker", "MASTERY_CEIL", "MASTERY_FLOOR"]


# ---------------------------------------------------------------------------
# 数值边界与调制常数（全部在此集中定义，便于报告中的参数敏感性分析）
# ---------------------------------------------------------------------------

#: 掌握度下界。掌握度是概率，理论上可以无限接近 0，这里不做「下限托底」，
#: 避免出现「永远有 10% 掌握」的失真。
MASTERY_FLOOR: Final[float] = 0.0

#: 掌握度上界。留出 1e-3 的余量而不是取 1.0，避免后续 ``log`` / 除法出现数值退化。
MASTERY_CEIL: Final[float] = 0.999

#: 能力调制：``p_learn`` 的乘性因子 = ``ABILITY_BASE + ABILITY_GAIN * ability``。
#: 取 BASE = 0.55、GAIN = 0.90，使「中等能力 0.5」的乘子恰好为 1.0。
ABILITY_BASE: Final[float] = 0.55
ABILITY_GAIN: Final[float] = 0.90

#: 覆盖强度调制：``p_learn`` 的乘性因子 = ``WEIGHT_BASE + WEIGHT_GAIN * kp_weight``。
#: 取 BASE = WEIGHT_BASE = 0.5，使 kp_weight = 1（全覆盖）时乘子为 1.0。
WEIGHT_BASE: Final[float] = 0.50
WEIGHT_GAIN: Final[float] = 0.50

#: 调制后的有效学习率上下限，防止极端参数把单次更新变成「一步到位」。
P_LEARN_MIN: Final[float] = 1e-3
P_LEARN_MAX: Final[float] = 0.95


def _clip(value: float, low: float, high: float) -> float:
    """把 ``value`` 截断到 ``[low, high]``。"""
    if value < low:
        return low
    if value > high:
        return high
    return value


class MasteryTracker:
    """贝叶斯知识追踪（BKT-lite）四参数模型的掌握度推断。

    该类是**无状态**的纯函数集合（只持有超参数），因此可以被同一学习者模型反复复用、
    也可以在多线程读场景下共享；学习者状态由 :class:`learnpath.learner.profile.LearnerModel`
    持有。

    Args:
        params: BKT 四参数，默认取 :data:`learnpath.config.BKT`
            （p_init = 0.10, p_learn = 0.40, p_slip = 0.10, p_guess = 0.20）。
            参数含义见 :class:`learnpath.config.BKTParams`。

    Example:
        >>> tracker = MasteryTracker()
        >>> after = tracker.update(prior=0.30, observed_score=0.85,
        ...                        kp_weight=1.0, ability=0.6)
        >>> 0.0 <= after <= 1.0
        True
        >>> tracker.decay(0.60, 12.0) < 0.60   # 12 周不复习，掌握度下降
        True
    """

    __slots__ = ("_params",)

    def __init__(self, params: BKTParams = BKT) -> None:
        self._params = params

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def params(self) -> BKTParams:
        """当前使用的 BKT 四参数（供报告 / 模型卡引用）。"""
        return self._params

    @property
    def p_init(self) -> float:
        """学习前的先验掌握概率。"""
        return float(self._params.p_init)

    # ------------------------------------------------------------------
    # 核心：一次观测 + 学习转移
    # ------------------------------------------------------------------
    def observation_likelihood(self, observed_score: float) -> tuple[float, float]:
        """软观测似然 ``(P(obs|已掌握), P(obs|未掌握))``。

        Args:
            observed_score: 归一化学习表现 ∈ [0, 1]（0 = 完全失败，1 = 完全达标）。

        Returns:
            ``(like_mastered, like_unmastered)``，对应观测方程的两个概率。
        """
        score = _clip(float(observed_score), 0.0, 1.0)
        p_slip = float(self._params.p_slip)
        p_guess = float(self._params.p_guess)
        # 已掌握：大概率达标，小概率失误
        like_mastered = score * (1.0 - p_slip) + (1.0 - score) * p_slip
        # 未掌握：小概率蒙对，大概率不达标
        like_unmastered = score * p_guess + (1.0 - score) * (1.0 - p_guess)
        return like_mastered, like_unmastered

    def posterior(self, prior: float, observed_score: float) -> float:
        """仅做**观测更新**（贝叶斯后验），不含学习转移。

        公式::

            p_post = p * P(obs|L=1) / (p * P(obs|L=1) + (1-p) * P(obs|L=0))

        该方法主要用于报告里的「单步贝叶斯修正」解释与消融实验；
        日常掌握度更新请用 :meth:`update`（它还会追加学习转移）。

        Args:
            prior: 观测前的掌握概率 ∈ [0, 1]。
            observed_score: 归一化学习表现 ∈ [0, 1]。

        Returns:
            观测后的掌握概率 ∈ [0, 1]。
        """
        p = _clip(float(prior), MASTERY_FLOOR, MASTERY_CEIL)
        like_mastered, like_unmastered = self.observation_likelihood(observed_score)
        denominator = p * like_mastered + (1.0 - p) * like_unmastered
        if denominator <= 0.0:  # 数值保护：似然全为 0 时退化为先验
            return p
        return _clip(p * like_mastered / denominator, MASTERY_FLOOR, MASTERY_CEIL)

    def effective_learning_rate(self, kp_weight: float, ability: float) -> float:
        """能力与覆盖强度调制后的有效学习率 ``p_learn_eff``。

        公式::

            p_learn_eff = clip(p_learn * (0.55 + 0.90 * ability) * (0.50 + 0.50 * kp_weight),
                               1e-3, 0.95)

        Args:
            kp_weight: 课程对该知识点的覆盖强度 ∈ (0, 1]。
            ability: 学习者能力 ∈ [0, 1]。

        Returns:
            单次学习后「未掌握 → 已掌握」的跃迁概率。
        """
        weight = _clip(float(kp_weight), 1e-3, 1.0)
        ability_ = _clip(float(ability), 0.0, 1.0)
        rate = float(self._params.p_learn)
        rate *= ABILITY_BASE + ABILITY_GAIN * ability_
        rate *= WEIGHT_BASE + WEIGHT_GAIN * weight
        return _clip(rate, P_LEARN_MIN, P_LEARN_MAX)

    def update(
        self,
        prior: float,
        observed_score: float,
        kp_weight: float,
        ability: float,
    ) -> float:
        """BKT 一次完整更新：观测更新 + 学习转移。

        这是本系统的掌握度状态方程，被历史回放、自适应反馈、路径收益估计共同调用。

        Args:
            prior: 本次学习前的掌握概率 ``P(L_t = 1)`` ∈ [0, 1]。
            observed_score: 归一化学习表现 ∈ [0, 1]（结课测验 / 完成度折算）。
            kp_weight: 该课程对该知识点的覆盖强度 ∈ (0, 1]。
            ability: 学习者能力 ∈ [0, 1]（来自 ``Learner.ability``）。

        Returns:
            ``P(L_{t+1} = 1)``，即**学完本节课之后的掌握度**（同时也就是下节课的先验），
            落在 ``[0, 0.999]``。
        """
        p = _clip(float(prior), MASTERY_FLOOR, MASTERY_CEIL)
        post = self.posterior(p, observed_score)
        p_learn_eff = self.effective_learning_rate(kp_weight, ability)
        # 学习转移：未掌握的部分按 p_learn_eff 转化为掌握
        next_mastery = post + (1.0 - post) * p_learn_eff
        return _clip(next_mastery, MASTERY_FLOOR, MASTERY_CEIL)

    # ------------------------------------------------------------------
    # 遗忘
    # ------------------------------------------------------------------
    def decay(self, mastery: float, idle_weeks: float) -> float:
        """按空闲周数做指数遗忘衰减（以先验基线为锚点）。

        公式::

            floor  = p_init
            decayed = floor + max(0, mastery - floor) * MASTERY_DECAY_PER_WEEK ** idle_weeks

        指数衰减把「每周保留固定比例」应用于任意长度的时间间隔，等价于遗忘曲线在半对数
        坐标下线性（Ebbinghaus）。``idle_weeks <= 0`` 时原值返回。

        为什么以 ``p_init`` 为锚点而不是衰减到 0：
            遗忘抹去的是**已学到**的部分，而不是「学习前的先验基线」。若直接乘到 0，
            一个 30 周未复习的知识点会被压到接近 0，与「基础常识不会完全忘光」的事实不符，
            也会让全库掌握度均值塌陷、知识缺口被人为放大。因此这里只让超出先验的那部分
            按遗忘曲线回落。

        Args:
            mastery: 衰减前掌握度 ∈ [0, 1]。
            idle_weeks: 距上次学习 / 复习的空闲周数（可为小数）。

        Returns:
            衰减后的掌握度 ∈ [0, 1]。
        """
        value = _clip(float(mastery), MASTERY_FLOOR, MASTERY_CEIL)
        weeks = float(idle_weeks)
        if weeks <= 0.0:
            return value
        floor = float(self._params.p_init)
        if value <= floor:
            return value
        factor = float(MASTERY_DECAY_PER_WEEK) ** weeks
        return _clip(floor + (value - floor) * factor, MASTERY_FLOOR, MASTERY_CEIL)

    def decay_factor(self, idle_weeks: float) -> float:
        """纯衰减系数 ``MASTERY_DECAY_PER_WEEK ** idle_weeks``（向量化衰减时复用）。"""
        weeks = float(idle_weeks)
        if weeks <= 0.0:
            return 1.0
        return float(MASTERY_DECAY_PER_WEEK) ** weeks

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 调试用
        p = self._params
        return (
            f"MasteryTracker(p_init={p.p_init}, p_learn={p.p_learn}, "
            f"p_slip={p.p_slip}, p_guess={p.p_guess})"
        )
