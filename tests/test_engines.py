"""BKT / MIRT / TC 检测器迁移正确性测试."""

import numpy as np
import pytest

from cogmirror.bkt import BKTEvolutionLayer, BKTParams
from cogmirror.belief_state import BloomLevel, TCState
from cogmirror.mirt import BiFactorMIRT5D, MIRTItemParams
from cogmirror.tc import TCStateDetector


class TestBKT:
    def test_monotonic_increase_on_correct(self):
        layer = BKTEvolutionLayer()
        p = layer.get_mastery("python.variables")
        assert p == pytest.approx(0.1)  # p_init 默认
        for _ in range(5):
            p_new = layer.update("python.variables", correct=True)
            assert p_new > p
            p = p_new

    def test_decrease_on_wrong(self):
        layer = BKTEvolutionLayer()
        layer.update("python.variables", correct=True)
        p1 = layer.get_mastery("python.variables")
        p2 = layer.update("python.variables", correct=False)
        assert p2 < p1

    def test_mastery_bounded(self):
        layer = BKTEvolutionLayer()
        for _ in range(100):
            p = layer.update("python.variables", correct=True)
            assert 0.0 <= p <= 1.0

    def test_decay(self):
        layer = BKTEvolutionLayer()
        layer.update("python.variables", correct=True)
        p = layer.get_mastery("python.variables")
        p_decayed = layer.apply_decay("python.variables", days_since_last=30)
        assert p_decayed < p

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            BKTParams(p_init=1.5)


class TestMIRT:
    def test_all_correct_positive_theta(self):
        mirt = BiFactorMIRT5D()
        responses = np.ones(10)
        problem_ids = [f"p{i}" for i in range(10)]
        theta_hat, theta_cov = mirt.estimate_theta(responses, problem_ids)
        assert theta_hat.shape == (5,)
        assert theta_cov.shape == (5, 5)
        # 全对 -> θ 整体应为正
        assert float(np.mean(theta_hat)) > 0

    def test_all_wrong_negative_theta(self):
        mirt = BiFactorMIRT5D()
        responses = np.zeros(10)
        problem_ids = [f"p{i}" for i in range(10)]
        theta_hat, _ = mirt.estimate_theta(responses, problem_ids)
        assert float(np.mean(theta_hat)) < 0

    def test_empty_returns_prior(self):
        mirt = BiFactorMIRT5D()
        theta_hat, theta_cov = mirt.estimate_theta(np.array([]), [])
        assert np.allclose(theta_hat, np.zeros(5))
        assert np.allclose(theta_cov, np.eye(5))

    def test_length_mismatch_raises(self):
        mirt = BiFactorMIRT5D()
        with pytest.raises(ValueError):
            mirt.estimate_theta(np.ones(3), ["a", "b"])

    def test_partial_credit_scores_accepted(self):
        # 新场景：score 是 0-1 连续值（非 0/1），MAP 估计应仍产出有限值
        mirt = BiFactorMIRT5D()
        responses = np.array([1.0, 0.7, 0.0, 0.5, 1.0, 0.3])
        theta_hat, _ = mirt.estimate_theta(responses, [f"q{i}" for i in range(6)])
        assert np.all(np.isfinite(theta_hat))

    def test_predict_probability_range(self):
        mirt = BiFactorMIRT5D()
        item = mirt.default_item_params("q1")
        for theta in (np.zeros(5), np.ones(5) * 3, np.ones(5) * -3):
            p = mirt.predict_probability(theta, item)
            assert 0.0 < p < 1.0


class TestTCDetector:
    def test_pre_to_liminal(self):
        det = TCStateDetector()
        tc = None
        # L3 正确 +0.3/次，3 次到 0.9 >= 0.7 -> liminal
        for _ in range(3):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY,
                            current_tc_state=tc, has_active_misc=False)
        assert tc.status == "liminal"

    def test_misconception_blocks_liminal(self):
        det = TCStateDetector()
        tc = None
        for _ in range(3):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY,
                            current_tc_state=tc, has_active_misc=True)
        assert tc.status == "pre_liminal"

    def test_liminal_to_post_liminal(self):
        det = TCStateDetector()
        tc = None
        for _ in range(3):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        # liminal 后继续 L3+ 正确，+0.25/次，3 次到 1.0 -> post_liminal
        for _ in range(3):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        assert tc.status == "post_liminal"
        assert tc.irreversible

    def test_post_liminal_irreversible(self):
        det = TCStateDetector()
        done = TCState(tc_id="python.loops", status="post_liminal",
                       progress=1.0, irreversible=True)
        out = det.detect("python.loops", correct=False,
                         bloom_level=BloomLevel.REMEMBER,
                         current_tc_state=done, has_active_misc=False)
        assert out is done  # 不可退回

    def test_wrong_answer_in_liminal_does_not_regress_status(self):
        det = TCStateDetector()
        tc = None
        for _ in range(3):
            tc = det.detect("python.loops", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        assert tc.status == "liminal"
        tc = det.detect("python.loops", correct=False,
                        bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                        has_active_misc=False)
        assert tc.status == "liminal"
        assert tc.progress < 1.0
