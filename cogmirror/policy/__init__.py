"""策略学习（P1 预留，MVP 不接入主流程）.

来自 ECOS 的两种轻量 bandit 策略（MIGRATION.md 第2节）：
- LinUCB (Li et al., 2010)
- Thompson Sampling (Beta-Bernoulli)

POMDP+PBVI 不迁移（数据规模不支撑）。本包在 Phase 0 仅保证可用性，
A/B 对比在 P1 阶段做策略对比时才启用。
"""

from .linucb import LinUCB, BanditConfig
from .thompson import ThompsonSampling, ThompsonConfig
from .ab_test import PolicyABTest, ABTestResult

__all__ = [
    "LinUCB", "BanditConfig",
    "ThompsonSampling", "ThompsonConfig",
    "PolicyABTest", "ABTestResult",
]
