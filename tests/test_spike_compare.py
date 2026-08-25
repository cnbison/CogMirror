"""比对分析测试：Spearman 已知数据验算、n=1 比对、未覆盖不比对、维度相关."""

import numpy as np
import pytest

from cogmirror.belief_state import BloomLevel

from spike.compare import (
    compare_n1,
    five_d_corr_matrix,
    render_comparison,
    spearman_corr,
)
from spike.dialogue import AnchorTurn, ExecResult
from spike.graph import DimensionId
from spike.protocol import GroundTruthAnchors, SessionRecord
from spike.scorer import ScorerOutput


def _record(transcript_anchors=("loops-L1-S1-K", "loops-L3-S3-P"),
            solo={"loops": 3.0},
            five_d=None, overall=0.55,
            per_topic=None, per_bloom=None) -> SessionRecord:
    gt = GroundTruthAnchors(
        source="bank_deterministic",
        per_topic_bank=per_topic or {"python.loops": 0.75, "python.variables": 0.5},
        per_bloom_bank=per_bloom or {"REMEMBER": 1.0, "UNDERSTAND": 0.5,
                                     "APPLY": 0.5, "ANALYZE": 0.5},
        per_topic_correct={"python.loops": {"answered": 4, "correct": 2,
                                            "total_score": 3.0}},
    )
    transcript = []
    for i, a in enumerate(transcript_anchors):
        transcript.append(AnchorTurn(role="assistant", anchor=a, text=f"q{i}"))
        transcript.append(AnchorTurn(role="user", anchor=None, text=f"a{i}"))
    est = ScorerOutput(
        five_d=five_d or {DimensionId.K: 0.65, DimensionId.P: 0.40,
                          DimensionId.S: 0.55, DimensionId.C: 0.55,
                          DimensionId.X: 0.45},
        bloom={BloomLevel.REMEMBER: 0.70, BloomLevel.UNDERSTAND: 0.65,
               BloomLevel.APPLY: 0.40, BloomLevel.ANALYZE: 0.55},
        solo=solo,
        overall=overall,
        evidence_notes={},
        insufficient=[],
    )
    return SessionRecord(user_id="u1", date="2026-08-25", graph_version="0.1.0",
                         model="fake", ground_truth=gt,
                         transcript=transcript, exec_results=[], estimate=est)


class TestSpearman:
    def test_perfect_positive(self):
        assert spearman_corr([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0

    def test_perfect_negative(self):
        assert spearman_corr([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0

    def test_monotone_partial(self):
        r = spearman_corr([1, 2, 3, 4, 5], [2, 4, 3, 5, 6])
        assert 0.7 <= r <= 1.0

    def test_insufficient_samples(self):
        assert np.isnan(spearman_corr([1], [2]))

    def test_constant_series_nan(self):
        assert np.isnan(spearman_corr([1, 1, 1], [1, 2, 3]))


class TestFiveDCorrMatrix:
    def test_known_correlation(self):
        r1 = _record(per_topic={"python.loops": 0.5},
                     five_d={DimensionId.K: 0.2, DimensionId.P: 0.2,
                             DimensionId.S: 0.5, DimensionId.C: 0.5,
                             DimensionId.X: 0.5})
        r2 = _record(per_topic={"python.loops": 0.5},
                     five_d={DimensionId.K: 0.9, DimensionId.P: 0.9,
                             DimensionId.S: 0.5, DimensionId.C: 0.5,
                             DimensionId.X: 0.5})
        corr = five_d_corr_matrix([r1, r2])
        assert corr is not None and corr.shape == (5, 5)
        # K 与 P 完全同向 -> 1.0；K 与常数 S -> NaN
        assert corr[0, 1] == pytest.approx(1.0)
        assert np.isnan(corr[0, 2])

    def test_insufficient_records(self):
        assert five_d_corr_matrix([_record()]) is None
        assert five_d_corr_matrix([]) is None


class TestCompareN1:
    def test_delta_math(self):
        rec = _record(solo={"loops": 3.0})
        report = compare_n1(rec)
        loops_row = [r for r in report.per_topic
                     if r["topic"] == "python.loops"][0]
        # solo 3 -> (3-1)/4 = 0.5；锚点 0.75 -> delta = -0.25
        assert loops_row["dialogue_est"] == pytest.approx(0.5)
        assert loops_row["delta"] == pytest.approx(-0.25)
        assert loops_row["covered"] is True

    def test_uncovered_topic_not_compared(self):
        """对话没覆盖的 topic 必须显式标注，不能拿空数据当 0 分."""
        rec = _record(transcript_anchors=("loops-L1-S1-K",))  # 只覆盖 loops
        report = compare_n1(rec)
        var_row = [r for r in report.per_topic
                   if r["topic"] == "python.variables"][0]
        assert var_row["covered"] is False
        assert var_row["dialogue_est"] is None
        assert var_row["delta"] is None
        assert var_row["note"] == "对话未覆盖，不比对"

    def test_uncovered_bloom_not_compared(self):
        rec = _record(transcript_anchors=("loops-L1-S1-K",))
        report = compare_n1(rec)
        bloom_rows = {r["level"]: r for r in report.per_bloom}
        assert bloom_rows["APPLY"]["covered"] is False
        assert bloom_rows["APPLY"]["delta"] is None

    def test_overall_delta(self):
        rec = _record(overall=0.55, per_topic={"python.loops": 0.75})
        report = compare_n1(rec)
        assert report.overall_bank == pytest.approx(0.75)
        assert report.overall_delta == pytest.approx(-0.20)

    def test_five_d_summary_includes_insufficient(self):
        rec = _record()
        rec.estimate.insufficient = ["S"]
        report = compare_n1(rec)
        assert report.five_d_summary["insufficient"] == ["S"]
        assert report.five_d_summary["K"] == pytest.approx(0.65)

    def test_notes_mention_proxy_and_no_gt_for_cxs(self):
        report = compare_n1(_record())
        joined = " ".join(report.agreement_notes)
        assert "代理指标" in joined
        assert "C/X/S" in joined

    def test_render_marks_uncovered(self):
        rec = _record(transcript_anchors=("loops-L1-S1-K",))
        text = render_comparison(compare_n1(rec))
        assert "对话未覆盖，不比对" in text
        assert "[逐 topic]" in text
        assert "[逐 Bloom 层]" in text
        assert "[五维估计摘要" in text
