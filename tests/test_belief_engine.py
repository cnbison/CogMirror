"""简化版 BeliefEngine / BeliefState 测试.

Phase 0 关卡问题的自动化部分：5D 状态更新数值是否合理
（不是全零、不是全部相同）。
"""

import numpy as np
import pytest

from cogmirror.belief_engine import BeliefEngine, Observation
from cogmirror.belief_state import BeliefState, BloomLevel


def make_obs(problem_id: str, skill_id: str, score: float,
             bloom: BloomLevel = BloomLevel.APPLY,
             self_confidence=None) -> Observation:
    return Observation(
        skill_id=skill_id, problem_id=problem_id, score=score,
        bloom_level=bloom, self_confidence=self_confidence,
    )


class TestBeliefState:
    def test_initial_validate(self):
        state = BeliefState(user_id="u1")
        ok, issues = state.validate()
        assert ok, issues

    def test_invalid_flagged(self):
        state = BeliefState(user_id="u1")
        state.K.mastery_prob = 1.5
        ok, issues = state.validate()
        assert not ok
        assert any("K.mastery_prob" in s for s in issues)

    def test_serialization_roundtrip(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(6):
            state = engine.update(state, make_obs(f"q{i}", "python.loops", 0.8))
        d = state.to_dict()
        restored = BeliefState.from_dict(d)
        assert restored.user_id == "u1"
        assert np.allclose(restored.theta_mean, state.theta_mean)
        assert restored.bloom_profile.apply == pytest.approx(state.bloom_profile.apply)
        assert restored.C.tc_states["python.loops"].status == state.C.tc_states["python.loops"].status
        ok, issues = restored.validate()
        assert ok, issues

    def test_bloom_dominant_tie_takes_highest(self):
        state = BeliefState(user_id="u1")
        state.bloom_profile.remember = 0.9
        state.bloom_profile.apply = 0.9
        state.bloom_profile.update_dominant()
        assert state.bloom_profile.dominant_layer == BloomLevel.APPLY

    def test_bloom_dominant_all_neutral_takes_lowest(self):
        state = BeliefState(user_id="u1")
        state.bloom_profile.update_dominant()
        assert state.bloom_profile.dominant_layer == BloomLevel.REMEMBER


class TestBeliefEngineUpdate:
    def test_partial_credit_correct_derivation(self):
        obs = make_obs("q1", "python.variables", 0.7)
        assert obs.correct is True
        obs = make_obs("q2", "python.variables", 0.5)
        assert obs.correct is False
        obs = make_obs("q3", "python.variables", 0.6)
        assert obs.correct is True

    def test_5d_not_degenerate_after_answers(self):
        """Phase 0 关卡：混合作答后 5D 数值合理（非全零、非全同）.

        前提：题库 MIRT 载荷已注册。所有题共用默认参数时后验在维度间
        对称，五维必然相同（实测发现，见 questions.py 模块注释）。
        """
        from cogmirror.questions import QuestionBank

        bank = QuestionBank()
        questions = bank.all_questions()
        engine = BeliefEngine()
        engine.l2.register_items_bulk(bank.mirt_items())
        state = engine.create_initial_state("u1")
        # 模拟一个"概念记得住、代码写不出"的用户：概念题对、代码题错
        for i, q in enumerate(questions):
            score = 0.9 if q.qtype in ("choice", "fill") else 0.1
            state = engine.update(state, make_obs(q.problem_id, q.skill_id, score, q.bloom_level))
        theta = state.theta_vector()
        assert not np.allclose(theta, np.zeros(5)), "theta 全零"
        mastery = state.mastery_vector()
        assert len(set(np.round(mastery, 6))) > 1, f"mastery_prob 全部相同: {mastery}"
        # 概念题对、代码题错 -> K 应高于 P
        assert state.K.theta > state.P.theta, f"K={state.K.theta} 应大于 P={state.P.theta}"
        ok, issues = state.validate()
        assert ok, issues

    def test_all_correct_raises_theta(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(8):
            state = engine.update(state, make_obs(f"q{i}", "python.variables", 1.0))
        assert float(np.mean(state.theta_vector())) > 0

    def test_bloom_update_responds_to_score(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        before = state.bloom_profile.apply
        state = engine.update(state, make_obs("q1", "python.loops", 1.0, bloom=BloomLevel.APPLY))
        assert state.bloom_profile.apply > before
        before = state.bloom_profile.apply
        state = engine.update(state, make_obs("q2", "python.loops", 0.0, bloom=BloomLevel.APPLY))
        assert state.bloom_profile.apply < before

    def test_bloom_probability_clamped(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(50):
            state = engine.update(state, make_obs(f"q{i}", "python.loops", 1.0, bloom=BloomLevel.APPLY))
        assert 0.0 <= state.bloom_profile.apply <= 1.0

    def test_illusory_confidence_detected(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        state = engine.update(state, make_obs("q1", "python.variables", 0.1, self_confidence=0.95))
        assert state.C.illusory_confidence_flag
        assert len(state.C.illusory_confidence_hits) == 1
        hit = state.C.illusory_confidence_hits[0]
        assert hit.problem_id == "q1"
        assert hit.gap == pytest.approx(0.85)

    def test_calibrated_confidence_not_flagged(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        state = engine.update(state, make_obs("q1", "python.variables", 0.9, self_confidence=0.9))
        assert not state.C.illusory_confidence_flag
        # 自评低实际差也不算伪自信（是校准良好或低自信，不是虚高）
        state = engine.update(state, make_obs("q2", "python.variables", 0.1, self_confidence=0.3))
        assert not state.C.illusory_confidence_flag

    def test_misconception_keyword_discount(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        obs = Observation(
            skill_id="python.variables", problem_id="q1", score=1.0,
            explanation_text="我不明白，x = x + 1 这个等式两边不相等，这不是无解吗？矛盾啊",
        )
        state = engine.update(state, obs)
        assert len(state.C.misconception_hits) == 1
        assert state.C.misconception_hits[0].misc_id == "M1"
        assert state.C.discount_factor < 1.0

    def test_overall_confidence_grows_with_data(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        assert state.overall_confidence == 0.0
        for i in range(5):
            state = engine.update(state, make_obs(f"q{i}", "python.loops", 0.7))
        assert state.overall_confidence > 0.0

    def test_trajectory_appends(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(3):
            state = engine.update(state, make_obs(f"q{i}", "python.loops", 0.8))
        assert len(state.trajectory.snapshots) == 3

    def test_history_and_bkt(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(4):
            state = engine.update(state, make_obs(f"q{i}", "python.loops", 1.0))
        assert len(engine.get_history("u1")) == 4
        assert engine.get_bkt_mastery("python.loops") > 0.1
