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

    def test_liminal_to_post_liminal_requires_streak(self):
        # docstring 规则：进入 liminal 后需连续 3 次 L3+ 正确才 post_liminal
        det = TCStateDetector()
        tc = None
        for _ in range(3):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        assert tc.status == "liminal"
        # 仅 2 次连续 L3+ 正确：progress 到 1.0 但仍不算跨过
        for _ in range(2):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        assert tc.status == "liminal"
        assert tc.progress == pytest.approx(1.0)
        # 第 3 次连续 L3+ 正确 -> post_liminal
        tc = det.detect("python.variables", correct=True,
                        bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                        has_active_misc=False)
        assert tc.status == "post_liminal"
        assert tc.irreversible

    def test_wrong_answer_in_liminal_resets_streak(self):
        # 答错清零连续 L3+ 计数，需重新累积 3 次
        det = TCStateDetector()
        tc = None
        for _ in range(3):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        assert tc.status == "liminal"
        for _ in range(2):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        tc = det.detect("python.variables", correct=False,
                        bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                        has_active_misc=False)
        assert tc.status == "liminal"
        for _ in range(3):
            tc = det.detect("python.variables", correct=True,
                            bloom_level=BloomLevel.APPLY, current_tc_state=tc,
                            has_active_misc=False)
        assert tc.status == "post_liminal"

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


class TestPeakAndDecayView:
    """P3 间隔衰减：历史重放峰值 + 无状态衰减视图（BeliefEngine 层）."""

    @staticmethod
    def _rows(n_correct: int, n_wrong: int, skill: str = "python.loops",
              days_ago: int = 0) -> list[dict]:
        from datetime import datetime, timedelta
        ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
        return [{"skill_id": skill, "correct": 1, "created_at": ts}] * n_correct + \
               [{"skill_id": skill, "correct": 0, "created_at": ts}] * n_wrong

    def test_peak_replay_is_readonly_and_differs_from_fresh_l1(self):
        # 缺口 1：BKT 不持久化，历史峰值靠重放推导；重放不得创建 l1 模型
        from cogmirror.belief_engine import BeliefEngine
        engine = BeliefEngine()
        peaks = engine.peak_mastery_from_history(self._rows(6, 2, days_ago=42))
        # 6 连对后 P(L) 峰值明显高（~0.97），随后的错把"当前"拉低但峰值保留
        assert peaks["python.loops"] >= 0.7
        # 只读：l1 不被污染，当前掌握仍是先验
        assert "python.loops" not in engine.l1.all_skills()
        assert engine.get_bkt_mastery("python.loops") == pytest.approx(0.1)

    def test_peak_ignores_rows_without_skill(self):
        from cogmirror.belief_engine import BeliefEngine
        engine = BeliefEngine()
        rows = self._rows(3, 0) + [{"correct": 1, "created_at": "2026-01-01T00:00:00"}]
        assert set(engine.peak_mastery_from_history(rows)) == {"python.loops"}

    def test_decayed_math_30days_is_e_minus_1(self):
        from cogmirror.belief_engine import BeliefEngine
        from datetime import datetime
        engine = BeliefEngine()
        rows = self._rows(6, 0, days_ago=30)
        view = engine.decayed_mastery_view(rows, now=datetime.now())
        peak, decayed, days = view["python.loops"]
        assert days == 30
        assert decayed == pytest.approx(peak * np.exp(-1.0))

    def test_decayed_view_idempotent(self):
        # 缺口 2：连续调用两次结果相同（无状态，不经 l1 原地乘法）
        from cogmirror.belief_engine import BeliefEngine
        engine = BeliefEngine()
        rows = self._rows(6, 1, days_ago=42)
        assert engine.decayed_mastery_view(rows) == engine.decayed_mastery_view(rows)

    def test_recent_practice_no_decay(self):
        # 连续练习（days=0）不衰减：DISPROVEN 点的引擎侧（无间隔不误报）
        from cogmirror.belief_engine import BeliefEngine
        engine = BeliefEngine()
        peak, decayed, days = engine.decayed_mastery_view(self._rows(6, 0))[ "python.loops"]
        assert days == 0
        assert decayed == peak

    def test_days_from_last_response_only(self):
        # 间隔按该 skill 最近一条算：近期练过则旧记录不参与衰减
        from cogmirror.belief_engine import BeliefEngine
        from datetime import datetime, timedelta
        engine = BeliefEngine()
        old = [{"skill_id": "python.loops", "correct": 1,
                "created_at": (datetime.now() - timedelta(days=90)).isoformat()}] * 6
        recent = [{"skill_id": "python.loops", "correct": 1, "created_at": datetime.now().isoformat()}]
        peak, decayed, days = engine.decayed_mastery_view(old + recent)["python.loops"]
        assert days == 0
        assert decayed == peak
