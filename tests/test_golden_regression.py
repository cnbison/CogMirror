"""黄金回归套件（P1，docs/implementation-plan.md 第 2 节）.

固定黄金集（tests/golden/sequences.py），每次引擎变更跑全量：
- expect 窗口 = 客观 scorer；
- baseline.json 数值摘要对比（带容差）= 行为回归检测；
- 回归判定：本次 FAIL 且 baseline 中该 case 曾 PASS（冷启动不报）。
失败即 fail 整个 run（首版决定：工程纪律 > 平滑）。

跑法：pytest -m regression（已注册 marker，见 pyproject.toml）。
基线更新：python -m tests.golden.update_baseline（必须带文档化 diff）。
"""

from __future__ import annotations

import pytest

from cogmirror import belief_engine
from tests.golden.runner import evaluate, load_baseline
from tests.golden.sequences import SEQUENCES, sequence_by_name

pytestmark = pytest.mark.regression


@pytest.mark.parametrize("sequence", SEQUENCES, ids=lambda s: s["name"])
def test_golden_sequence(sequence):
    failures, entry = evaluate(sequence)
    if not failures:
        return
    lines = list(failures)
    if entry is not None and entry.get("passed"):
        # 本次 FAIL 且基线曾 PASS = 回归（引擎行为变化，不是序列/窗口问题）
        lines.insert(0, "REGRESSION: 该 case 在 baseline.json 中为 PASS，本次 FAIL")
    elif entry is None:
        lines.insert(0, "冷启动（baseline 未收录该 case），不标回归，仅报 expect 失败")
    pytest.fail("\n".join(lines))


def test_baseline_file_exists_and_covers_all_sequences():
    """基线文件存在且收录全部黄金序列（防基线丢失后回归静默失效）."""
    baseline = load_baseline()
    assert baseline is not None, "tests/golden/baseline.json 缺失，回归只剩冷启动"
    cases = baseline.get("cases", {})
    missing = [s["name"] for s in SEQUENCES if s["name"] not in cases]
    assert not missing, f"基线未收录: {missing}（先跑 python -m tests.golden.update_baseline）"


def test_seeded_engine_change_is_caught(monkeypatch):
    """证伪测试（验收核心，方案 2.5）：故意改引擎阈值，回归必须报 FAIL.

    ILLUSORY_MASTERY_DISCOUNT 0.15 -> 0.16：discount 0.85^5 与 0.84^5 的差
    远超容差 1e-6，baseline 数值对比必须抓到。
    """
    monkeypatch.setattr(belief_engine, "ILLUSORY_MASTERY_DISCOUNT", 0.16)
    failures, entry = evaluate(sequence_by_name("overconfident_loop_learner"))
    assert entry is not None and entry.get("passed"), "该 case 基线应为 PASS"
    assert failures, "seeded 引擎改动未被回归抓到（回归基建失效）"
    assert any("discount_factor" in f or "mastery" in f for f in failures), (
        f"seeded 改动应体现为 discount/mastery 漂移，实际: {failures}")


def test_seeded_gap_threshold_change_is_caught(monkeypatch):
    """ILLUSORY_GAP_THRESHOLD 0.5 -> 0.4：边界序列 gap 0.45 的题由不命中变命中.

    曾实测漏报（黄金集 gap 全 >=0.5 时该改动不可观测），补
    calibration_boundary_learner 序列后必须能抓到。
    """
    monkeypatch.setattr(belief_engine, "ILLUSORY_GAP_THRESHOLD", 0.4)
    failures, entry = evaluate(sequence_by_name("calibration_boundary_learner"))
    assert entry is not None and entry.get("passed"), "该 case 基线应为 PASS"
    assert failures, "GAP 阈值 seeded 改动未被回归抓到（边界覆盖缺口）"
    assert any("illusory_hits" in f or "discount_factor" in f for f in failures), (
        f"seeded 改动应体现为命中数/折扣漂移，实际: {failures}")


def test_seeded_self_conf_min_change_is_caught(monkeypatch):
    """ILLUSORY_SELF_CONF_MIN 0.7 -> 0.6：自评 0.65 的题由不命中变命中."""
    monkeypatch.setattr(belief_engine, "ILLUSORY_SELF_CONF_MIN", 0.6)
    failures, entry = evaluate(sequence_by_name("calibration_boundary_learner"))
    assert entry is not None and entry.get("passed"), "该 case 基线应为 PASS"
    assert failures, "自评下限 seeded 改动未被回归抓到（边界覆盖缺口）"
