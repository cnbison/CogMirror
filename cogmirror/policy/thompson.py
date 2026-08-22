"""Thompson Sampling - 贝叶斯 Bandit.

迁移自 ECOS `ecos/lca/l4_optimization/thompson.py`（算法逻辑未改）。

算法 (Beta-Bernoulli conjugate)：
    - 每 arm 维护 (α, β) 标量
    - select_arm: sample θ_a ~ Beta(α_a, β_a), return argmax
    - update(arm, reward): α += reward, β += (1 - reward)

接口与 LinUCB 同构（context 参数忽略，non-contextual Beta）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

_log = logging.getLogger(__name__)


class ThompsonConfig:
    """Thompson Sampling 配置.

    Attributes:
        n_arms:     arm 数量（与 LinUCB 一致）
        alpha_prior: Beta prior α 初值（默认 1.0 = uniform prior）
        beta_prior:  Beta prior β 初值（默认 1.0 = uniform prior）
        seed:       PRNG seed（None = 系统 entropy，测试用固定 seed）
    """

    def __init__(self, n_arms: int = 10, alpha_prior: float = 1.0,
                 beta_prior: float = 1.0, seed: Optional[int] = None) -> None:
        self.n_arms = n_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.seed = seed


class ThompsonSampling:
    """Beta-Bernoulli Thompson Sampling.

    用法：
        bandit = ThompsonSampling(n_arms=10, seed=42)
        arm_idx = bandit.select_arm(context=None)
        bandit.update(arm_idx, context=None, reward=0.7)

    适用条件：
        - reward ∈ [0, 1]（Beta prior 假设）
        - non-contextual
        - n_arms 固定
    """

    def __init__(
        self,
        n_arms: int = 10,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_arms = int(n_arms)
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self._rng = np.random.default_rng(seed)
        # per-arm Beta 参数
        self.alpha: np.ndarray = np.full(self.n_arms, self.alpha_prior, dtype=float)
        self.beta: np.ndarray = np.full(self.n_arms, self.beta_prior, dtype=float)
        self.arm_pull_counts: np.ndarray = np.zeros(self.n_arms, dtype=int)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """Beta(α, β) 采样选 argmax."""
        if self.n_arms <= 0:
            _log.warning("ThompsonSampling.select_arm: n_arms=%s, 返 0 (degenerate)", self.n_arms)
            return 0
        samples = self._rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, context: Optional[np.ndarray] = None, reward: float = 0.0) -> None:
        """Beta conjugate update: α += reward, β += (1 - reward).

        防御性：arm 越界 warning + 跳过；reward 截断到 [0, 1]。
        """
        if arm < 0 or arm >= self.n_arms:
            _log.warning(
                "ThompsonSampling.update: arm 越界 (arm=%s, n_arms=%s), 跳过",
                arm, self.n_arms,
            )
            return
        clamped = max(0.0, min(1.0, float(reward)))
        self.alpha[arm] += clamped
        self.beta[arm] += (1.0 - clamped)
        self.arm_pull_counts[arm] += 1

    def get_arm_stats(self) -> Dict[str, Any]:
        """获取每个 arm 的统计信息（与 LinUCB 接口同构）."""
        expected_reward = (self.alpha / (self.alpha + self.beta)).tolist()
        return {
            "n_arms": self.n_arms,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "arm_pull_counts": self.arm_pull_counts.tolist(),
            "total_pulls": int(self.arm_pull_counts.sum()),
            "expected_reward": expected_reward,
        }

    def dump_state(self) -> Dict[str, Any]:
        """导出状态."""
        return {
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "arm_pull_counts": self.arm_pull_counts.tolist(),
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "n_arms": self.n_arms,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """加载状态（含维度校验）."""
        n_arms = state.get("n_arms", self.n_arms)
        if int(n_arms) != self.n_arms:
            raise ValueError(
                f"ThompsonSampling state n_arms 不匹配 (expected={self.n_arms}, got={n_arms})"
            )

        alpha = state.get("alpha") or []
        beta = state.get("beta") or []
        arm_pull_counts = state.get("arm_pull_counts") or []

        if len(alpha) != self.n_arms or len(beta) != self.n_arms:
            raise ValueError(
                f"ThompsonSampling state 长度不匹配 (alpha={len(alpha)}, beta={len(beta)}, "
                f"expected={self.n_arms})"
            )

        self.alpha = np.array(alpha, dtype=float)
        self.beta = np.array(beta, dtype=float)
        self.arm_pull_counts = (
            np.array(arm_pull_counts, dtype=int)
            if len(arm_pull_counts) else np.zeros(self.n_arms, dtype=int)
        )


__all__ = ["ThompsonSampling", "ThompsonConfig"]
