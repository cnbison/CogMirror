"""L4 Contextual Bandits (LinUCB 算法).

迁移自 ECOS `ecos/lca/l4_optimization/linucb.py`（算法逻辑未改，
去掉对 ECOS BeliefState 的无用 import）。

LinUCB 算法（Li et al., 2010）：
  - 每个 arm a 维护参数 θ_a
  - 选择 arm：argmax_a (θ_a^T x + α √(x^T A_a^{-1} x))
  - 探索-利用平衡：α 控制（exploration bonus）
  - 更新：A_a += x x^T, b_a += r x
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

_log = logging.getLogger(__name__)


class BanditConfig:
    """Bandit 配置."""

    def __init__(
        self,
        n_arms: int = 10,
        context_dim: int = 16,
        alpha: float = 1.0,
        decay_factor: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        # Discounted LinUCB decay factor (Russac et al. 2019)
        #   1.0（默认）= 无衰减；<1.0 让历史 reward 衰减，鼓励探索被忽略 arm
        self.decay_factor = float(decay_factor)


class LinUCB:
    """LinUCB 算法实现--Li et al., 2010.

    用法：
        bandit = LinUCB(n_arms=10, context_dim=16, alpha=1.0)
        arm_idx = bandit.select_arm(context_vector)
        bandit.update(arm_idx, context_vector, reward=0.3)
    """

    def __init__(self, n_arms: int = 10, context_dim: int = 16, alpha: float = 1.0,
                 decay_factor: float = 1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.decay_factor = float(decay_factor)
        # 每个 arm 的协方差矩阵 + 线性参数
        self.A: List[np.ndarray] = [np.eye(context_dim) for _ in range(n_arms)]
        self.b: List[np.ndarray] = [np.zeros(context_dim) for _ in range(n_arms)]
        # 统计信息
        self.arm_pull_counts: np.ndarray = np.zeros(n_arms, dtype=int)

    def select_arm(self, context: np.ndarray) -> int:
        """根据 UCB 选择 arm.

        Args:
            context: 上下文向量（dim = context_dim）

        Returns:
            arm 索引 [0, n_arms)
        """
        x = np.asarray(context, dtype=float).flatten()
        assert x.shape[0] == self.context_dim, (
            f"context dim mismatch: expected {self.context_dim}, got {x.shape[0]}"
        )

        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            ucb_values[arm] = self.score_arm(arm, x)

        return int(np.argmax(ucb_values))

    def score_arm(self, arm: int, context: np.ndarray) -> float:
        """计算指定 arm 在给定 context 下的 UCB 分数.

        Returns:
            UCB 分数 (expected_reward + alpha * confidence_bound)
            异常时返回 0.0, 不污染 bandit 状态
        """
        x = np.asarray(context, dtype=float).flatten()
        if arm < 0 or arm >= self.n_arms:
            _log.warning("score_arm: arm 越界 (arm=%s, n_arms=%s), 返回 0.0", arm, self.n_arms)
            return 0.0
        if x.shape[0] != self.context_dim:
            _log.warning(
                "score_arm: context dim 错误 (expected=%s, got=%s), 返回 0.0",
                self.context_dim, x.shape[0],
            )
            return 0.0
        try:
            A_inv = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.context_dim)
        theta = A_inv @ self.b[arm]
        expected_reward = float(theta @ x)
        confidence_bound = self.alpha * float(np.sqrt(x @ A_inv @ x))
        return expected_reward + confidence_bound

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """更新选中的 arm（在线岭回归 + Discounted LinUCB）.

            A_a ← decay_factor * A_a + x x^T
            b_a ← decay_factor * b_a + r x

        Args:
            arm: 选中的 arm 索引
            context: 上下文向量
            reward: 观测到的奖励（归一化到 [0, 1]）
        """
        x = np.asarray(context, dtype=float).flatten()
        self.A[arm] = self.decay_factor * self.A[arm] + np.outer(x, x)
        self.b[arm] = self.decay_factor * self.b[arm] + reward * x
        self.arm_pull_counts[arm] += 1

    def get_arm_stats(self) -> dict:
        """获取每个 arm 的统计信息（调试接口）."""
        return {
            "n_arms": self.n_arms,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "decay_factor": self.decay_factor,
            "arm_pull_counts": self.arm_pull_counts.tolist(),
            "total_pulls": int(self.arm_pull_counts.sum()),
        }


__all__ = ["LinUCB", "BanditConfig"]
