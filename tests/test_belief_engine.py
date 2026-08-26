"""简化版 BeliefEngine / BeliefState 测试.

Phase 0 关卡问题的自动化部分：5D 状态更新数值是否合理
（不是全零、不是全部相同）。
"""

import numpy as np
import pytest

from cogmirror.belief_engine import BeliefEngine, Observation
from cogmirror.belief_state import BeliefState, BloomLevel
from cogmirror.questions import QuestionBank


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
        # covered_layers 必须随序列化往返，否则恢复后主导层级会被重置
        assert restored.bloom_profile.covered_layers == state.bloom_profile.covered_layers
        assert restored.bloom_profile.dominant_layer == state.bloom_profile.dominant_layer
        ok, issues = restored.validate()
        assert ok, issues

    def test_bloom_dominant_tie_takes_highest(self):
        state = BeliefState(user_id="u1")
        state.bloom_profile.remember = 0.9
        state.bloom_profile.apply = 0.9
        state.bloom_profile.covered_layers = {BloomLevel.REMEMBER, BloomLevel.APPLY}
        state.bloom_profile.update_dominant()
        assert state.bloom_profile.dominant_layer == BloomLevel.APPLY

    def test_bloom_dominant_all_neutral_takes_lowest(self):
        state = BeliefState(user_id="u1")
        state.bloom_profile.update_dominant()
        assert state.bloom_profile.dominant_layer == BloomLevel.REMEMBER

    def test_bloom_dominant_ignores_uncovered_high_prior(self):
        # 回归：弱学习者全挂时 L5/L6 停先验 0.5，不能被判成"主导 EVALUATE"
        state = BeliefState(user_id="u1")
        state.bloom_profile.apply = 0.38
        state.bloom_profile.analyze = 0.38
        state.bloom_profile.remember = 0.38
        state.bloom_profile.understand = 0.38
        # evaluate/create 未练，停在先验 0.5——不应参与竞争
        state.bloom_profile.covered_layers = {
            BloomLevel.REMEMBER, BloomLevel.UNDERSTAND, BloomLevel.APPLY, BloomLevel.ANALYZE,
        }
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

    def test_bloom_layers_differentiate(self):
        # 步长加大后，练过的层明显高于先验、未练的层保持 0.5，六层拉开可见差距
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(4):
            state = engine.update(state, make_obs(f"q{i}", "python.loops", 1.0, bloom=BloomLevel.APPLY))
        assert state.bloom_profile.apply > 0.8
        assert state.bloom_profile.remember == pytest.approx(0.5)

    def test_bloom_probability_clamped(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(50):
            state = engine.update(state, make_obs(f"q{i}", "python.loops", 1.0, bloom=BloomLevel.APPLY))
        assert 0.0 <= state.bloom_profile.apply <= 1.0

    def test_tc_lifecycle_via_engine_update(self):
        # 双指标交叉印证（规则2）：TC 三态机在真实引擎路径下按连续 3 次 L3+
        # 正确推进，答错重置连续计数。detector 层单测之外的第二条证据线。
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for i in range(3):  # 3 次 L3+ 正确 -> liminal
            state = engine.update(
                state, make_obs(f"q{i}", "python.loops", 1.0, bloom=BloomLevel.APPLY))
        tc = state.C.tc_states["python.loops"]
        assert tc.status == "liminal"
        for i in range(2):  # 2 次连续 L3+ 正确，未满 3 次不算跨过
            state = engine.update(
                state, make_obs(f"q{i+10}", "python.loops", 1.0, bloom=BloomLevel.APPLY))
        tc = state.C.tc_states["python.loops"]
        assert tc.status == "liminal"
        state = engine.update(  # 答错：重置连续计数，不退回 pre_liminal
            state, make_obs("q-x", "python.loops", 0.0, bloom=BloomLevel.APPLY))
        tc = state.C.tc_states["python.loops"]
        assert tc.status == "liminal"
        assert "post_liminal_candidate" not in tc.liminal_signals
        for i in range(3):  # 重新累计 3 次连续 L3+ 正确 -> post_liminal
            state = engine.update(
                state, make_obs(f"q{i+20}", "python.loops", 1.0, bloom=BloomLevel.APPLY))
        tc = state.C.tc_states["python.loops"]
        assert tc.status == "post_liminal"
        assert tc.irreversible

    # F10 收口：临界概念三态（pre->liminal->post_liminal）在产品内端到端可达。
    # 用真实题库 + 真实判分逐题答对全部 L3+，验证每 topic 都能走完整条链。
    CODE_SOLUTIONS = {
        "pv-l3-01": "def swap_values(a, b):\n    return (b, a)",
        "pv-l3-03": "def repeat_word(s):\n    return s * 2",
        "pl-l3-01": "def sum_to(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total",
        "pl-l3-02": "def max_of(nums):\n    m = nums[0]\n    for x in nums:\n        if x > m:\n            m = x\n    return m",
        "pl-l3-03": "def count_even(nums):\n    c = 0\n    for n in nums:\n        if n % 2 == 0:\n            c += 1\n    return c",
        "pl-l3-04": "def sum_range(a, b):\n    total = 0\n    for i in range(a, b + 1):\n        total += i\n    return total",
        "pf-l3-01": "def is_even(n):\n    return n % 2 == 0",
        "pf-l3-02": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "pf-l3-03": "def first_last(nums):\n    return (nums[0], nums[-1])",
        "pf-l3-04": "def sum_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total",
        "pr-l3-01": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
        "pr-l3-02": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)",
        "ps-l3-01": ("def make_counter():\n    count = 0\n    def counter():\n        nonlocal count\n        count += 1\n        return count\n    return counter"),
        "ps-l3-02": "count = 0\ndef step():\n    global count\n    count += 1\n    return count",
    }

    @pytest.mark.parametrize("topic", [
        "python.variables", "python.loops", "python.functions",
        "python.recursion", "python.scope",
    ])
    def test_tc_post_liminal_reachable_all_topics_real_bank(self, topic):
        bank = QuestionBank()
        engine = BeliefEngine()
        engine.l2.register_items_bulk(bank.mirt_items())
        state = engine.create_initial_state("e2e")

        l3_plus = [q for q in bank.by_topic(topic)
                   if q.bloom_level.value >= BloomLevel.APPLY.value]
        assert len(l3_plus) >= 6, f"{topic} L3+ 不足 6 道"
        for q in l3_plus:
            if q.qtype == "choice":
                answer = str(q.answer)
            elif q.qtype == "fill":
                answer = q.accepted[0]
            else:
                answer = self.CODE_SOLUTIONS[q.problem_id]
            score, details = bank.grade_answer(q, answer)
            assert score == 1.0, f"{topic} {q.problem_id} 正确解判分错误: {details}"
            state = engine.update(
                state, Observation(
                    skill_id=q.skill_id, problem_id=q.problem_id, score=score,
                    bloom_level=q.bloom_level, self_confidence=None,
                    explanation_text="",  # 避免 misconception 关键词误拦 liminal
                ))
        tc = state.C.tc_states[topic]
        assert tc.status == "post_liminal", f"{topic} 未到 post_liminal: {tc.status}"
        assert tc.irreversible

    def test_weak_learner_dominant_not_unpracticed_layer(self):
        # 回归（自测弱学习者场景）：只练了 L1-L4 且全错，
        # 主导层级不能跳到未练的 L5/L6（它们停先验 0.5 会反超）
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        for level in (BloomLevel.REMEMBER, BloomLevel.UNDERSTAND,
                      BloomLevel.APPLY, BloomLevel.ANALYZE):
            for i in range(3):
                state = engine.update(
                    state, make_obs(f"q-{level.name}-{i}", "python.loops", 0.0, bloom=level))
        assert state.bloom_profile.dominant_layer.value <= BloomLevel.ANALYZE.value
        assert state.bloom_profile.dominant_layer != BloomLevel.EVALUATE

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

    def test_illusory_hit_discounts_c_mastery(self):
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        before = state.C.mastery_prob
        state = engine.update(state, make_obs("q1", "python.variables", 0.1, self_confidence=0.95))
        assert state.C.illusory_confidence_flag
        assert state.C.mastery_prob < before
        ok, issues = state.validate()
        assert ok, issues

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
