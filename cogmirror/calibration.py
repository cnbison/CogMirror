"""自评置信度校准曲线（P2，A1 迁移）.

移植 PersonalAGI `src/genesis/calibration/` 的纯算法核心（零 LLM）：
按自评置信度分桶，学习每桶的真实答对率，用曲线回答"自评 X% 的题，
实际答对率是多少"。伪自信折扣幅度由曲线驱动（belief_engine Step 7），
替代固定 ILLUSORY_MASTERY_DISCOUNT。

与源模式的必要本地化：Laplace 平滑 actual_rate = (correct+1)/(n+2)。
PersonalAGI 无平滑是因其样本量大；CogMirror 单用户、桶稀疏，
无平滑会出现 actual_rate=0/1 的极端桶，平滑是兜底而非风格选择。

判对语义与引擎一致：score >= 0.6 记 correct（ECOS v0.54.0 派生）。
"""

from __future__ import annotations

from dataclasses import dataclass

# 桶宽（PersonalAGI 同款 0.1）
BUCKET_WIDTH = 0.1
# 置信度 clamp 上界：1.0 会落出最后一个 0.1 桶，源模式 clamp 到 0.999
CONF_MAX = 0.999
# expected_accuracy 的最小桶样本：低于此返回 None（数据不足诚实回退）
MIN_BUCKET_N = 5


def bucket_confidence(conf: float) -> str:
    """置信度 -> 0.1 宽桶标签（"0.3" 表示 [0.3, 0.4)）.

    clamp 到 [0, 0.999]：负值入 0.0 桶，1.0 入 0.9 桶（0.999 的 int 截断）。
    """
    c = min(max(float(conf), 0.0), CONF_MAX)
    return f"{int(c / BUCKET_WIDTH) * BUCKET_WIDTH:.1f}"


@dataclass(frozen=True)
class CalibrationCurve:
    """单个置信度桶的校准数据.

    Attributes:
        bucket: 桶标签（"0.3" = 自评 [0.3, 0.4)）
        n: 该桶样本数（原始计数，未经平滑）
        correct: 该桶判对次数（score >= 0.6）
        predicted: 桶中点（自评的标称置信度）
        actual_rate: Laplace 平滑后的真实答对率 (correct+1)/(n+2)
        correction_factor: actual_rate / predicted（自评高于实绩 -> <1）
    """

    bucket: str
    n: int
    correct: int
    predicted: float
    actual_rate: float
    correction_factor: float


class CalibrationCurveComputer:
    """从作答记录计算校准曲线（records = [{self_confidence, score}]）."""

    def compute(self, records: list[dict]) -> list[CalibrationCurve]:
        """分桶聚合，返回按桶排序的曲线（空记录 -> 空列表）.

        records 中 self_confidence 为 None 的行跳过（未自评的作答无校准信息）。
        """
        by_bucket: dict[str, tuple[int, int]] = {}
        for r in records:
            conf = r.get("self_confidence")
            if conf is None:
                continue
            correct = 1 if (r.get("score") or 0.0) >= 0.6 else 0
            b = bucket_confidence(conf)
            n, c = by_bucket.get(b, (0, 0))
            by_bucket[b] = (n + 1, c + correct)

        curves = []
        for b in sorted(by_bucket):
            n, correct = by_bucket[b]
            predicted = float(b) + BUCKET_WIDTH / 2.0
            actual_rate = (correct + 1) / (n + 2)
            factor = actual_rate / predicted if predicted > 0.0 else 1.0
            curves.append(CalibrationCurve(
                bucket=b, n=n, correct=correct,
                predicted=predicted, actual_rate=actual_rate,
                correction_factor=factor,
            ))
        return curves

    @staticmethod
    def expected_accuracy(curves: list[CalibrationCurve], claimed_conf: float,
                          min_n: int = MIN_BUCKET_N) -> float | None:
        """查询"自评 claimed_conf 的题，实际答对率是多少".

        桶样本 < min_n 返回 None（数据不足，调用方回退固定折扣）；
        无该桶（从未在该区间自评过）同样返回 None。
        """
        b = bucket_confidence(claimed_conf)
        for c in curves:
            if c.bucket == b:
                return c.actual_rate if c.n >= min_n else None
        return None


def compute_ece(curves: list[CalibrationCurve]) -> float:
    """Expected Calibration Error：加权 |actual_rate - predicted|.

    权重 = 桶样本数占比（PersonalAGI metrics.py 同款聚合）。
    actual 用 Laplace 平滑值（与曲线口径一致，小样本下不出现 0/1 极端）。
    无样本返回 0.0（无数据 = 无可测失准，由调用方负责标注数据不足）。
    """
    total = sum(c.n for c in curves)
    if total == 0:
        return 0.0
    return sum(c.n * abs(c.actual_rate - c.predicted) for c in curves) / total
