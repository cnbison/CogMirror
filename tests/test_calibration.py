"""校准曲线测试（P2，方案 3.6）.

覆盖：桶划分 / Laplace 数学 / correction_factor 语义 / min_n 回退 /
曲线驱动折扣比固定折扣更敏感（验收核心）/ ECE 单调性 / 无校准回退。
"""

import pytest

from cogmirror.belief_engine import (
    BeliefEngine,
    CALIBRATED_DISCOUNT_MAX,
    ILLUSORY_MASTERY_DISCOUNT,
)
from cogmirror.calibration import (
    CalibrationCurve,
    CalibrationCurveComputer,
    bucket_confidence,
    compute_ece,
)
from cogmirror.belief_state import BloomLevel
from cogmirror.questions import QuestionBank


def recs(n: int, conf: float, correct: bool) -> list[dict]:
    score = 0.9 if correct else 0.0
    return [{"self_confidence": conf, "score": score}] * n


class TestBucketConfidence:
    @pytest.mark.parametrize("conf,want", [
        (0.0, "0.0"), (0.05, "0.0"), (0.099, "0.0"),
        (0.1, "0.1"), (0.34, "0.3"), (0.99, "0.9"),
        (1.0, "0.9"),  # clamp 0.999 -> 0.9 桶
        (-0.2, "0.0"),
    ])
    def test_bucket(self, conf, want):
        assert bucket_confidence(conf) == want


class TestCompute:
    def test_laplace_math(self):
        # n=1 全错: (0+1)/(1+2) = 1/3；n=1 全对: (1+1)/3 = 2/3
        curves = CalibrationCurveComputer().compute(recs(1, 0.35, correct=False))
        assert len(curves) == 1
        c = curves[0]
        assert c.bucket == "0.3"
        assert c.n == 1 and c.correct == 0
        assert c.actual_rate == pytest.approx(1 / 3)
        assert c.predicted == pytest.approx(0.35)

        curves = CalibrationCurveComputer().compute(recs(1, 0.35, correct=True))
        assert curves[0].actual_rate == pytest.approx(2 / 3)
        assert curves[0].correct == 1

    def test_correct_derivation_uses_06_line(self):
        # score 0.6 -> correct（与引擎 partial credit 派生一致）；0.5 -> 错
        records = [{"self_confidence": 0.55, "score": 0.6},
                   {"self_confidence": 0.55, "score": 0.5}]
        curves = CalibrationCurveComputer().compute(records)
        assert curves[0].n == 2
        # Laplace: (1+1)/(2+2) = 0.5
        assert curves[0].actual_rate == pytest.approx(0.5)

    def test_correction_factor_semantics(self):
        # 过度自信（自评 0.9 桶全错）-> factor < 1
        curves = CalibrationCurveComputer().compute(recs(10, 0.9, correct=False))
        assert curves[0].correction_factor < 1.0
        # 欠自信（自评 0.2 桶全对）-> factor > 1
        curves = CalibrationCurveComputer().compute(recs(10, 0.2, correct=True))
        assert curves[0].correction_factor > 1.0

    def test_none_self_confidence_skipped_and_sorted(self):
        records = [{"self_confidence": None, "score": 1.0}] + \
            recs(2, 0.8, correct=True) + recs(1, 0.1, correct=False)
        curves = CalibrationCurveComputer().compute(records)
        assert [c.bucket for c in curves] == ["0.1", "0.8"]
        assert curves[1].n == 2

    def test_empty_records(self):
        assert CalibrationCurveComputer().compute([]) == []


class TestExpectedAccuracy:
    def test_min_n_fallback(self):
        curves = CalibrationCurveComputer().compute(recs(4, 0.9, correct=False))
        assert CalibrationCurveComputer.expected_accuracy(curves, 0.9) is None
        curves = CalibrationCurveComputer().compute(recs(5, 0.9, correct=False))
        assert CalibrationCurveComputer.expected_accuracy(curves, 0.92) is not None

    def test_missing_bucket_returns_none(self):
        curves = CalibrationCurveComputer().compute(recs(5, 0.9, correct=True))
        assert CalibrationCurveComputer.expected_accuracy(curves, 0.3) is None


class TestEngineDiscount:
    def _engine_with_curves(self, records) -> BeliefEngine:
        bank = QuestionBank()
        engine = BeliefEngine()
        engine.l2.register_items_bulk(bank.mirt_items())
        engine.set_calibration(CalibrationCurveComputer().compute(records))
        return engine

    def _hit(self, engine) -> float:
        from cogmirror.belief_engine import Observation
        state = engine.create_initial_state("u-cal")
        obs = Observation(skill_id="python.loops", problem_id="pl-l3-01",
                          score=0.1, bloom_level=BloomLevel.APPLY,
                          self_confidence=0.9)
        state = engine.update(state, obs)
        return state.C.discount_factor

    def test_overconfident_learner_deeper_discount_than_fixed(self):
        """验收核心（方案 3.7 DISPROVEN 反向）：曲线驱动必须比固定 0.15 更敏感."""
        # 自评 0.9 桶 6 次全错 -> expected = 1/8 -> discount = 0.875 -> 夹到 0.5
        engine = self._engine_with_curves(recs(6, 0.9, correct=False))
        calibrated = self._hit(engine)
        fixed = self._hit(BeliefEngine())
        assert calibrated == pytest.approx(1.0 - CALIBRATED_DISCOUNT_MAX)
        assert fixed == pytest.approx(1.0 - ILLUSORY_MASTERY_DISCOUNT)
        assert calibrated < fixed

    def test_moderate_bucket_discount_unclamped(self):
        # 自评 0.9 桶 6 对 5 错? 用 5 对 1 错: (5+1)/8 = 0.75 -> discount 0.25（不触夹取）
        records = recs(5, 0.9, correct=True) + recs(1, 0.9, correct=False)
        engine = self._engine_with_curves(records)
        assert self._hit(engine) == pytest.approx(0.75)

    def test_insufficient_bucket_falls_back_to_fixed(self):
        # 桶样本 4 < min_n=5 -> None -> 固定 0.15（与迁移前行为一致）
        engine = self._engine_with_curves(recs(4, 0.9, correct=False))
        assert self._hit(engine) == pytest.approx(1.0 - ILLUSORY_MASTERY_DISCOUNT)

    def test_no_calibration_unchanged(self):
        # 不注入曲线 = 迁移前行为（黄金回归基线依赖这一点）
        assert self._hit(BeliefEngine()) == pytest.approx(0.85)


class TestECE:
    def test_monotonic_worse_calibration_higher_ece(self):
        # 校准良好（自评与实绩一致）-> 低 ECE；严重过度自信 -> 高 ECE
        good = CalibrationCurveComputer().compute(
            recs(10, 0.9, correct=True) + recs(10, 0.2, correct=False))
        bad = CalibrationCurveComputer().compute(
            recs(10, 0.9, correct=False) + recs(10, 0.2, correct=True))
        assert compute_ece(bad) > compute_ece(good)

    def test_empty_curves_zero(self):
        assert compute_ece([]) == 0.0
