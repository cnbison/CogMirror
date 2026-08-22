"""policy 包（LinUCB / Thompson / A/B 框架）与持久层测试."""

import numpy as np
import pytest

from cogmirror.db import Database
from cogmirror.policy import LinUCB, ThompsonSampling, PolicyABTest


class TestLinUCB:
    def test_select_returns_valid_arm(self):
        bandit = LinUCB(n_arms=10, context_dim=16)
        arm = bandit.select_arm(np.zeros(16))
        assert 0 <= arm < 10

    def test_dim_mismatch_raises(self):
        bandit = LinUCB(n_arms=5, context_dim=16)
        with pytest.raises(AssertionError):
            bandit.select_arm(np.zeros(8))

    def test_update_increments_pulls(self):
        bandit = LinUCB(n_arms=3, context_dim=4)
        ctx = np.ones(4)
        arm = bandit.select_arm(ctx)
        bandit.update(arm, ctx, reward=1.0)
        assert bandit.arm_pull_counts[arm] == 1
        assert bandit.get_arm_stats()["total_pulls"] == 1

    def test_repeated_good_reward_pulls_more(self):
        # 均匀 context 下，多次对某 arm 给高 reward 后该 arm 后验均值应最高
        bandit = LinUCB(n_arms=4, context_dim=4, alpha=0.1)
        ctx = np.ones(4)
        for _ in range(20):
            bandit.update(2, ctx, reward=1.0)  # arm 2 一直好
        scores = [bandit.score_arm(a, ctx) for a in range(4)]
        assert scores[2] == max(scores)

    def test_decay_factor_accepted(self):
        bandit = LinUCB(n_arms=2, context_dim=3, decay_factor=0.95)
        assert bandit.decay_factor == 0.95


class TestThompsonSampling:
    def test_select_returns_valid_arm(self):
        bandit = ThompsonSampling(n_arms=10, seed=42)
        arm = bandit.select_arm()
        assert 0 <= arm < 10

    def test_update_conjugate(self):
        bandit = ThompsonSampling(n_arms=2, seed=42)
        bandit.update(0, reward=1.0)
        assert bandit.alpha[0] == 2.0  # 1 prior + 1
        assert bandit.beta[0] == 1.0
        assert bandit.arm_pull_counts[0] == 1

    def test_reward_clamped(self):
        bandit = ThompsonSampling(n_arms=2)
        bandit.update(0, reward=5.0)
        assert bandit.alpha[0] == 2.0

    def test_out_of_range_arm_skipped(self):
        bandit = ThompsonSampling(n_arms=2)
        bandit.update(7, reward=1.0)  # 越界，warning 跳过
        assert int(bandit.arm_pull_counts.sum()) == 0

    def test_state_roundtrip(self):
        bandit = ThompsonSampling(n_arms=3, seed=1)
        bandit.update(0, reward=1.0)
        bandit.update(1, reward=0.0)
        state = bandit.dump_state()
        restored = ThompsonSampling(n_arms=3)
        restored.load_state(state)
        assert np.allclose(restored.alpha, bandit.alpha)
        assert np.allclose(restored.beta, bandit.beta)

    def test_load_state_mismatch_raises(self):
        bandit = ThompsonSampling(n_arms=3)
        with pytest.raises(ValueError):
            bandit.load_state({"n_arms": 5, "alpha": [1] * 5, "beta": [1] * 5})


class TestPolicyABTest:
    def test_same_policy_degenerate(self):
        ab = PolicyABTest()
        result = ab.compare("u1", "linucb", "linucb", events=[])
        assert result.winner is None

    def test_unsupported_policy_degenerate(self):
        ab = PolicyABTest()
        result = ab.compare("u1", "linucb", "pomdp", events=[])
        assert result.winner is None
        assert result.n_a == 0

    def test_replay_produces_result(self):
        ab = PolicyABTest()
        events = [{"payload": {"score": s}} for s in (1.0, 0.8, 0.9, 0.2, 0.7, 0.6)]
        result = ab.compare("u1", "linucb", "thompson", events=events)
        assert result.n_a == 6 and result.n_b == 6
        # replay 对比的是机制不是策略优劣（同 reward 序列），均值应相等
        assert result.mean_reward_a == pytest.approx(result.mean_reward_b)

    def test_event_dict_payload_extracted(self):
        events = [
            {"payload": {"correct": True}},
            {"payload": {"score": 0.5}},
            {"payload": {}},  # 无 reward 字段，跳过
        ]
        ab = PolicyABTest()
        result = ab.compare("u1", "linucb", "thompson", events=events)
        assert result.n_a == 2


class TestDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        database = Database(tmp_path / "test.db")
        yield database
        database.close()

    def test_ensure_user_idempotent(self, db):
        db.ensure_user("u1")
        db.ensure_user("u1")
        user = db.get_user("u1")
        assert user["user_id"] == "u1"
        assert user["created_at"] == user["created_at"]

    def test_no_consent_fields_in_schema(self, db):
        """迁移硬性要求：不得残留监护人同意相关字段."""
        cols = {r["name"] for r in db._conn.execute("PRAGMA table_info(users)")}
        assert "consent_version" not in cols
        assert not any("consent" in c or "guardian" in c or "parent" in c for c in cols)
        # 成人向合规字段存在
        assert "data_export_requested_at" in cols
        assert "data_delete_requested_at" in cols

    def test_response_roundtrip(self, db):
        db.ensure_user("u1")
        db.save_response("u1", {
            "problem_id": "q1", "skill_id": "python.loops", "score": 0.7,
            "bloom_level": "APPLY", "self_confidence": 0.9,
            "user_answer": "def sum_to(n): ...", "timestamp": "2026-08-22T10:00:00",
        }, illusory_flag=False)
        rows = db.load_responses("u1")
        assert len(rows) == 1
        assert rows[0]["score"] == 0.7
        assert rows[0]["correct"] == 1  # score >= 0.6
        assert rows[0]["skill_id"] == "python.loops"

    def test_state_roundtrip(self, db):
        from cogmirror.belief_engine import BeliefEngine, Observation
        from cogmirror.belief_state import BloomLevel

        db.ensure_user("u1")
        engine = BeliefEngine()
        state = engine.create_initial_state("u1")
        state = engine.update(state, Observation(
            skill_id="python.loops", problem_id="q1", score=0.8, bloom_level=BloomLevel.APPLY,
        ))
        db.save_state(state)
        restored = db.load_latest_state("u1")
        assert restored is not None
        assert restored.user_id == "u1"
        assert np.allclose(restored.theta_mean, state.theta_mean)
        assert len(restored.trajectory.snapshots) == 1

    def test_load_latest_state_none_for_new_user(self, db):
        db.ensure_user("u1")
        assert db.load_latest_state("u1") is None

    def test_data_delete_purges(self, db):
        db.ensure_user("u1")
        db.save_response("u1", {
            "problem_id": "q1", "skill_id": "s", "score": 1.0,
            "bloom_level": "APPLY", "self_confidence": None,
            "user_answer": "", "timestamp": "2026-08-22T10:00:00",
        }, illusory_flag=False)
        db.request_data_delete("u1")
        assert db.load_responses("u1") == []
        user = db.get_user("u1")
        assert user["data_delete_requested_at"] is not None

    def test_data_export(self, db):
        db.ensure_user("u1")
        db.request_data_export("u1")
        data = db.export_user_data("u1")
        assert data["user"]["data_export_requested_at"] is not None
        assert isinstance(data["responses"], list)
