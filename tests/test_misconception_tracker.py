"""misconception 证据闭环测试（方案 P4，tests 计划 5.5）."""

from datetime import datetime

import pytest

from cogmirror.db import Database
from cogmirror.misconception_tracker import MisconceptionTracker


def _row(skill_id: str, misc_id: str | None, score: float) -> dict:
    """构造 reconcile 输入行（DB load_responses 格式的最小子集）."""
    return {"skill_id": skill_id, "misc_id": misc_id, "score": score}


# ── Laplace 置信度数学 ────────────────────────────────────────────


def test_confidence_laplace_math():
    t = MisconceptionTracker()
    assert t.confidence("M1") == 0.5  # 无证据 -> 先验
    t.record_success("M1")
    assert t.confidence("M1") == pytest.approx(2 / 3)  # (1+1)/(1+0+2)
    t.record_failure("M1")
    assert t.confidence("M1") == 0.5  # (1+1)/(1+1+2)


def test_confidence_direction():
    # 方案 5.6 验收：反复持续的误解权重 > 0.6；被克服的回落
    t = MisconceptionTracker()
    for _ in range(3):
        t.record_success("M8")  # 检测被证实 3 次
    assert t.confidence("M8") > 0.6
    assert t.confidence("M8") == pytest.approx(4 / 5)
    for _ in range(5):
        t.record_failure("M4")  # 检测被证伪 5 次
    assert t.confidence("M4") < 0.6


# ── reconcile 对账分支 ────────────────────────────────────────────


def test_reconcile_overcome_records_failure():
    # 命中后同 skill 下一条答对且未重触发 -> 检测被证伪（已克服/误报）
    t = MisconceptionTracker()
    t.reconcile([
        _row("python.scope", "M8", 0.0),
        _row("python.scope", None, 1.0),
    ])
    e = t.evidence("M8")
    assert e["failure"] == 1 and e["success"] == 0


def test_reconcile_persistent_records_success():
    # 命中后同 skill 下一条仍错 -> 检测被证实
    t = MisconceptionTracker()
    t.reconcile([
        _row("python.scope", "M8", 0.0),
        _row("python.scope", None, 0.0),
    ])
    e = t.evidence("M8")
    assert e["success"] == 1 and e["failure"] == 0


def test_reconcile_retrigger_records_success():
    # 命中后同 skill 下一条重触发同一误解 -> 检测被证实（即使那题答对）
    t = MisconceptionTracker()
    t.reconcile([
        _row("python.scope", "M8", 0.0),
        _row("python.scope", "M8", 1.0),
    ])
    e = t.evidence("M8")
    assert e["success"] == 1 and e["failure"] == 0


def test_reconcile_skips_other_skills_and_no_outcome():
    # 后续是别的 skill -> 不算对账结果；后面没有同 skill 响应 -> 无证据
    t = MisconceptionTracker()
    t.reconcile([
        _row("python.scope", "M8", 0.0),
        _row("python.loops", None, 1.0),
    ])
    assert t.evidence("M8") is None

    t2 = MisconceptionTracker()
    t2.reconcile([_row("python.scope", "M8", 0.0)])
    assert t2.evidence("M8") is None


def test_reconcile_multiple_hits_each_joined_to_next():
    # 两次 M8 命中，各自 join 到其后最近的同 skill 响应
    t = MisconceptionTracker()
    t.reconcile([
        _row("python.scope", "M8", 0.0),
        _row("python.scope", "M8", 0.0),  # 第 1 次命中的 outcome = 重触发 -> success
        _row("python.scope", None, 1.0),  # 第 2 次命中的 outcome = 克服 -> failure
    ])
    e = t.evidence("M8")
    assert e["success"] == 1 and e["failure"] == 1


# ── quarantine ────────────────────────────────────────────────────


def test_quarantine_thresholds():
    t = MisconceptionTracker()
    assert not t.quarantined("M1")  # 无证据不隔离
    for _ in range(3):
        t.record_failure("M1")  # s=0 f=3 -> conf 0.25 < 0.3 且 s+f >= 3
    assert t.quarantined("M1")
    # 证据不足（s+f < 3）时即使 conf 低也不隔离
    t2 = MisconceptionTracker()
    t2.record_failure("M2")
    t2.record_failure("M2")
    assert not t2.quarantined("M2")


# ── DB 往返 ──────────────────────────────────────────────────────


def test_evidence_db_roundtrip(tmp_path):
    db = Database(tmp_path / "ev.db")
    try:
        t = MisconceptionTracker()
        t.record_success("M8")
        t.record_failure("M8")
        t.record_success("M3")
        db.save_misconception_evidence(t.dump())

        restored = MisconceptionTracker()
        restored.load(db.load_misconception_evidence())
        assert restored.confidence("M8") == pytest.approx(t.confidence("M8"))
        assert restored.confidence("M3") == pytest.approx(t.confidence("M3"))
        assert restored.evidence("M8")["last_updated"] == t.evidence("M8")["last_updated"]
    finally:
        db.close()


def test_save_evidence_upsert_idempotent(tmp_path):
    db = Database(tmp_path / "ev.db")
    try:
        rows = [{"misc_id": "M8", "success_count": 1, "failure_count": 0,
                 "last_updated": datetime.now().isoformat()}]
        db.save_misconception_evidence(rows)
        db.save_misconception_evidence(rows)  # 幂等覆盖，不重复插入
        loaded = db.load_misconception_evidence()
        assert len(loaded) == 1
        assert loaded[0]["success_count"] == 1
    finally:
        db.close()


def test_delete_purges_evidence(tmp_path):
    # 合规：删除用户数据时证据一并清除（表无 user_id，单用户本地库全清）
    db = Database(tmp_path / "ev.db")
    try:
        db.ensure_user("u1")
        db.save_misconception_evidence(
            [{"misc_id": "M8", "success_count": 1, "failure_count": 0,
              "last_updated": datetime.now().isoformat()}])
        db.request_data_delete("u1")
        assert db.load_misconception_evidence() == []
    finally:
        db.close()
