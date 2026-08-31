"""纵向档案 + 会话反思段测试（方案 P5，测试计划 6.5）."""

import copy
import io
import sys
from datetime import datetime, timedelta

import pytest

from cogmirror import cli
from cogmirror.belief_state import BeliefState
from cogmirror.db import Database
from cogmirror.session import (
    last_session_struggles,
    multi_session_trend,
    trend_line,
)

BASE = datetime(2026, 8, 31, 10, 0, 0)


def _state(user_id: str, k: float, p: float = 0.5, s: float = 0.5) -> BeliefState:
    state = BeliefState(user_id=user_id)
    state.K.mastery_prob = k
    state.P.mastery_prob = p
    state.S.mastery_prob = s
    return state


def _seed_sessions(tmp_path, n_sessions: int, wrong_skills: list[tuple[str, str]] | None = None):
    """构造 n_sessions 个会话（间隔 40 分钟 > 聚类阈值 30 分钟）.

    每会话 1 条快照（K mastery 按 0.4 + 0.1*会话序递增）；wrong_skills =
    [(skill_id, 会话序)] 在指定会话窗口内放一条答错的 response。
    """
    db = Database(tmp_path / "sess.db")
    db.ensure_user("t1")
    for i in range(n_sessions):
        t = BASE + timedelta(minutes=40 * i)
        db.save_state(_state("t1", k=0.4 + 0.1 * i), created_at=t)
        for skill, session_i in (wrong_skills or []):
            if session_i != i:
                continue
            db.save_response("t1", {
                "problem_id": f"q-{skill}-{i}", "skill_id": skill, "score": 0.0,
                "bloom_level": "APPLY", "self_confidence": None,
                "user_answer": "", "timestamp": (t - timedelta(minutes=1)).isoformat(),
            }, illusory_flag=False)
    return db


# ── B1: last_session_struggles ────────────────────────────────────


def test_last_session_struggles_aggregation(tmp_path):
    # 会话 0 全对（variables 错题挂在会话 1）、会话 1 有 variables+loops 两道错题
    db = _seed_sessions(tmp_path, 2, wrong_skills=[
        ("python.variables", 1), ("python.loops", 1), ("python.variables", 1),
    ])
    try:
        assert last_session_struggles(db, "t1") == ["python.variables", "python.loops"]
    finally:
        db.close()


def test_last_session_struggles_excludes_older_session(tmp_path):
    # 错题挂在会话 0 -> 不属于"上次"（会话 1）
    db = _seed_sessions(tmp_path, 2, wrong_skills=[("python.loops", 0)])
    try:
        assert last_session_struggles(db, "t1") == []
    finally:
        db.close()


def test_last_session_struggles_no_snapshots(tmp_path):
    # 首次运行（无快照）-> 无卡点可召回
    db = Database(tmp_path / "sess.db")
    db.ensure_user("t1")
    try:
        assert last_session_struggles(db, "t1") == []
    finally:
        db.close()


def test_same_run_multiple_snapshots_one_session(tmp_path):
    # 一次 CLI 运行内多轮练习各存快照（间隔 2 分钟 < 30）-> 聚成同一会话，
    # 上次会话 = 全部快照的窗口
    db = Database(tmp_path / "sess.db")
    db.ensure_user("t1")
    try:
        for i in range(3):
            db.save_state(_state("t1", k=0.5 + 0.1 * i),
                          created_at=BASE + timedelta(minutes=2 * i))
        t = BASE + timedelta(minutes=1)
        db.save_response("t1", {
            "problem_id": "q1", "skill_id": "python.scope", "score": 0.0,
            "bloom_level": "APPLY", "self_confidence": None,
            "user_answer": "", "timestamp": t.isoformat(),
        }, illusory_flag=False)
        assert last_session_struggles(db, "t1") == ["python.scope"]
    finally:
        db.close()


# ── B1: multi_session_trend ───────────────────────────────────────


def test_multi_session_trend_first_last(tmp_path):
    # 3 会话 K 0.4 -> 0.5 -> 0.6：first=0.4 last=0.6 n=3
    db = _seed_sessions(tmp_path, 3)
    try:
        trend = multi_session_trend(db, "t1")
        assert trend["K"] == (pytest.approx(0.4), pytest.approx(0.6), 3)
        assert trend["P"] == (pytest.approx(0.5), pytest.approx(0.5), 3)
    finally:
        db.close()


def test_multi_session_trend_capped_to_n(tmp_path):
    # 5 个会话、n=3 -> 只取最近 3 个（0.6 -> 0.7 -> 0.8）
    db = _seed_sessions(tmp_path, 5)
    try:
        trend = multi_session_trend(db, "t1", n=3)
        assert trend["K"] == (pytest.approx(0.6), pytest.approx(0.8), 3)
    finally:
        db.close()


def test_multi_session_trend_needs_two_sessions(tmp_path):
    # 单会话无法谈趋势 -> 空 dict
    db = _seed_sessions(tmp_path, 1)
    try:
        assert multi_session_trend(db, "t1") == {}
    finally:
        db.close()


# ── B1: trend_line 文案 ──────────────────────────────────────────


def test_trend_line_rising_and_flat():
    line = trend_line({"K": (0.4, 0.6, 3), "P": (0.5, 0.51, 3)})
    assert "知识 +20%（3 次会话 40% -> 60%）" in line
    assert "程序技能" not in line  # 变化 < 2% 不提


def test_trend_line_all_flat():
    # 各维度都稳定 -> 一句"基本稳定"，不臆造趋势
    line = trend_line({"K": (0.5, 0.51, 2), "P": (0.6, 0.6, 2), "S": (0.4, 0.41, 2)})
    assert "基本稳定" in line


def test_trend_line_empty():
    assert trend_line({}) == ""


# ── B2: 反思句三态 ──────────────────────────────────────────────


def _engine_state():
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    # 解读段在无历史时提前返回"没有作答记录"，反思句测试需要非空历史
    engine.set_history("t1", [{"score": 1.0}, {"score": 0.0}])
    return engine, state


def test_reflection_with_delta():
    # 有 delta + 有作答 -> 反思句报变化 + 归因，下一步仍由收尾句引用
    engine, state = _engine_state()
    prev = copy.deepcopy(state)
    state.K.mastery_prob = 0.62
    state.P.mastery_prob = 0.45
    rows = [
        {"bloom_level": "REMEMBER", "score": 1.0},
        {"bloom_level": "UNDERSTAND", "score": 1.0},
        {"bloom_level": "APPLY", "score": 0.0},
    ]
    s = cli.map_interpretation(engine, state, prev_state=prev, session_rows=rows)
    assert "本次K 知识: +12%" in s
    assert "2 道记忆/理解层题答对" in s
    assert "本次P 程序技能: -5%" in s
    assert "1 道应用层题答错" in s
    assert "「一句话建议」" in s


def test_reflection_no_delta_no_claim():
    # DISPROVEN 点（方案 6.6）：无 delta -> 不声称"本次有变化"
    engine, state = _engine_state()
    prev = copy.deepcopy(state)
    s = cli.map_interpretation(engine, state, prev_state=prev,
                               session_rows=[{"bloom_level": "APPLY", "score": 1.0}])
    assert "本次" not in s
    assert "一句话建议" in s


def test_reflection_no_prev_state():
    # map-only / 黄金回归 runner 路径：不传 prev_state -> 无反思句
    engine, state = _engine_state()
    s = cli.map_interpretation(engine, state)
    assert "本次K" not in s and "本次P" not in s


def test_reflection_delta_without_matching_rows():
    # 有 delta 但无对应层作答（MIRT 先验微调）-> 只报变化不臆造归因
    engine, state = _engine_state()
    prev = copy.deepcopy(state)
    state.K.mastery_prob = 0.62
    s = cli.map_interpretation(engine, state, prev_state=prev,
                               session_rows=[{"bloom_level": "APPLY", "score": 1.0}])
    assert "本次K 知识: +12%" in s
    assert "题答对" not in s
    assert "题答错" not in s


# ── CLI 端到端：跨会话欢迎卡点 + 地图趋势段 ──────────────────────


def test_cli_welcome_struggles_and_trend(monkeypatch, tmp_path):
    # 会话 1 答错 1 题 -> 快照时间改到 40 分钟前；会话 2 再答 1 题：
    # 欢迎行浮现"上次卡住"（会话 1 的错题 topic），地图出现 [近几次趋势]
    # （2 个会话末对比）
    db_path = str(tmp_path / "cli.db")
    run_cli(monkeypatch, tmp_path, answers=["80\n", "0\n", "\n"],
            args=["--questions", "1"])

    db = Database(db_path)
    try:
        old = (datetime.now() - timedelta(minutes=40)).isoformat()
        # 快照与该会话 responses 一并改到过去（真实时序：作答在快照前）
        db._conn.execute("UPDATE belief_snapshots SET created_at = ?", (old,))
        db._conn.execute("UPDATE responses SET created_at = ?", (old,))
        db._conn.commit()
    finally:
        db.close()

    _, out = run_cli(monkeypatch, tmp_path, answers=["90\n", "1\n"],
                     args=["--questions", "1"])
    assert "上次卡住：变量赋值" in out
    assert "[近几次趋势]（2 次会话末对比）" in out
    # 无第三会话数据时不臆造更多趋势
    assert "3 次会话" not in out


def test_cli_no_struggles_line_when_all_correct(monkeypatch, tmp_path):
    # 会话 1 全对 -> 会话 2 欢迎行无"上次卡住"（无可召回卡点，不显示空段）
    db_path = str(tmp_path / "cli.db")
    run_cli(monkeypatch, tmp_path, answers=["80\n", "1\n"],
            args=["--questions", "1"])

    db = Database(db_path)
    try:
        old = (datetime.now() - timedelta(minutes=40)).isoformat()
        # 快照与该会话 responses 一并改到过去（真实时序：作答在快照前）
        db._conn.execute("UPDATE belief_snapshots SET created_at = ?", (old,))
        db._conn.execute("UPDATE responses SET created_at = ?", (old,))
        db._conn.commit()
    finally:
        db.close()

    _, out = run_cli(monkeypatch, tmp_path, answers=["90\n", "1\n"],
                     args=["--questions", "1"])
    assert "上次卡住" not in out
    # 无卡点不显示该行；趋势照常（2 个会话末对比，全对时数值照样可复核）
    assert "[近几次趋势]（2 次会话末对比）" in out


def run_cli(monkeypatch, tmp_path, answers, args):
    """驱动 cli.main 的输入流（与 test_cli.run_cli 同款，本文件自带避免跨文件依赖）."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(answers)))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    db_path = str(tmp_path / "cli.db")
    code = cli.main(["--user", "t1", "--db", db_path, *args])
    return code, sys.stdout.getvalue()
