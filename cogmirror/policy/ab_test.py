"""Policy A/B 对比框架（P1 预留，MVP 不接入主流程）.

迁移自 ECOS `ecos/evaluation/policy_ab_test.py`，按 MIGRATION.md 第2节改造：
- 只保留 replay 路径（同一 event 序列 replay 到两个 fresh bandit）
- 去掉 POMDP 支持（不迁移）与 LCAEngine 耦合路径
- student_id -> user_id

P1 阶段做策略对比（LinUCB vs Thompson）时启用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .linucb import LinUCB
from .thompson import ThompsonSampling

_log = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Policy AB test 结果.

    Attributes:
        user_id:       用户 ID
        policy_a:      Policy A 标识 (e.g. "linucb")
        policy_b:      Policy B 标识 (e.g. "thompson")
        mean_reward_a: Policy A 平均 reward
        mean_reward_b: Policy B 平均 reward
        n_a:           Policy A 样本数
        n_b:           Policy B 样本数
        winner:        "a" / "b" / None（不显著）
    """

    user_id: str
    policy_a: str
    policy_b: str
    mean_reward_a: float
    mean_reward_b: float
    n_a: int
    n_b: int
    winner: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "mean_reward_a": self.mean_reward_a,
            "mean_reward_b": self.mean_reward_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "winner": self.winner,
        }


class PolicyABTest:
    """Policy 对比框架：replay 同一 event 序列到两个 fresh bandit.

    用法：
        ab = PolicyABTest()
        result = ab.compare("user_001", "linucb", "thompson", events=[...])

    winner 判定：5% 阈值 + 至少 5 样本（继承 ECOS 标准）。
    """

    SUPPORTED_POLICIES: tuple = ("linucb", "linucb_baseline", "thompson")

    def compare(
        self,
        user_id: str,
        policy_a: str,
        policy_b: str,
        events: List[Any],
    ) -> ABTestResult:
        if policy_a == policy_b:
            return self._degenerate(user_id, policy_a, policy_b)
        if policy_a not in self.SUPPORTED_POLICIES or policy_b not in self.SUPPORTED_POLICIES:
            _log.warning("PolicyABTest: 不支持的 policy (a=%s, b=%s)", policy_a, policy_b)
            return self._degenerate(user_id, policy_a, policy_b)
        if not events:
            return self._degenerate(user_id, policy_a, policy_b)

        try:
            bandit_a = self._create_fresh_bandit(policy_a)
            bandit_b = self._create_fresh_bandit(policy_b)
            # 16 维 zero context（LinUCB 需要固定维度，Thompson 忽略）。
            # Replay 对比的是 update mechanism，context 用 placeholder。
            context = np.zeros(16, dtype=float)

            total_a = 0.0
            total_b = 0.0
            count = 0

            for event in events:
                reward = self._extract_reward_from_event(event)
                if reward is None:
                    continue
                reward = float(reward)

                try:
                    arm_a = bandit_a.select_arm(context=context)
                    bandit_a.update(arm_a, context=context, reward=reward)
                    total_a += reward
                except Exception:  # noqa: BLE001
                    _log.warning("PolicyABTest: bandit_a 操作失败 (count=%d), 跳过", count, exc_info=True)

                try:
                    arm_b = bandit_b.select_arm(context=context)
                    bandit_b.update(arm_b, context=context, reward=reward)
                    total_b += reward
                except Exception:  # noqa: BLE001
                    _log.warning("PolicyABTest: bandit_b 操作失败 (count=%d), 跳过", count, exc_info=True)

                count += 1

            mean_a = total_a / count if count > 0 else 0.0
            mean_b = total_b / count if count > 0 else 0.0

            winner = None
            if count >= 5:
                if mean_a > mean_b * 1.05:
                    winner = "a"
                elif mean_b > mean_a * 1.05:
                    winner = "b"

            return ABTestResult(
                user_id=user_id,
                policy_a=policy_a,
                policy_b=policy_b,
                mean_reward_a=mean_a,
                mean_reward_b=mean_b,
                n_a=count,
                n_b=count,
                winner=winner,
            )
        except Exception:  # noqa: BLE001
            _log.warning(
                "PolicyABTest.compare 失败 (user=%s, a=%s, b=%s)",
                user_id, policy_a, policy_b, exc_info=True,
            )
            return self._degenerate(user_id, policy_a, policy_b)

    @staticmethod
    def _degenerate(user_id: str, policy_a: str, policy_b: str) -> ABTestResult:
        return ABTestResult(
            user_id=user_id, policy_a=policy_a, policy_b=policy_b,
            mean_reward_a=0.0, mean_reward_b=0.0, n_a=0, n_b=0, winner=None,
        )

    @staticmethod
    def _create_fresh_bandit(policy_id: str) -> Any:
        """根据 policy_id 创建 fresh bandit."""
        if policy_id in ("linucb", "linucb_baseline"):
            return LinUCB(n_arms=10, context_dim=16, alpha=1.0, decay_factor=1.0)
        if policy_id == "thompson":
            return ThompsonSampling(n_arms=10, seed=42)
        raise ValueError(f"PolicyABTest: 未知 policy_id={policy_id!r}")

    @staticmethod
    def _extract_reward_from_event(event: Any) -> Optional[float]:
        """从 event 提取 reward.

        优先顺序：payload["score"] > payload["reward"] > payload["correct"]。
        解析失败返 None（跳过），不 raise。
        """
        try:
            payload = getattr(event, "payload", None)
            if payload is None and isinstance(event, dict):
                payload = event.get("payload", event)
            if not isinstance(payload, dict):
                return None
            if payload.get("score") is not None:
                return float(payload["score"])
            if payload.get("reward") is not None:
                return float(payload["reward"])
            if payload.get("correct") is not None:
                return 1.0 if bool(payload["correct"]) else 0.0
            return None
        except Exception:  # noqa: BLE001
            _log.warning("PolicyABTest._extract_reward_from_event 失败, 跳过", exc_info=True)
            return None


__all__ = ["PolicyABTest", "ABTestResult"]
