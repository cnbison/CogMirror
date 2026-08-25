"""冒烟测试：FakeLLM 全链路端到端（无需 API key），输出文件与回读."""

import io
import sys

import pytest

from spike import __main__
from spike.protocol import load_sessions


def _run_smoke(monkeypatch, tmp_path, topics=("loops", "variables")):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    code = __main__.main(["smoke", "--jsonl", str(tmp_path),
                          "--topics", *topics])
    out = sys.stdout.getvalue()
    return code, out, tmp_path


class TestSmokeEndToEnd:
    def test_smoke_succeeds(self, monkeypatch, tmp_path):
        code, out, _ = _run_smoke(monkeypatch, tmp_path)
        assert code == 0
        assert "冒烟测试" in out
        assert "会话比对报告" in out
        assert "非法 anchor 重试" in out

    def test_report_file_written_and_roundtrip(self, monkeypatch, tmp_path):
        code, out, path = _run_smoke(monkeypatch, tmp_path)
        report_file = path / "smoke_report.jsonl"
        assert report_file.exists()
        records, errors = load_sessions(path)
        assert errors == []
        assert len(records) == 1
        rec = records[0]
        assert rec.user_id == "smoke-user"
        assert rec.estimate is not None
        assert rec.ground_truth is not None
        # 关键数据回读一致
        assert rec.estimate.overall == 0.55
        assert rec.ground_truth.per_topic_bank["python.loops"] == pytest.approx(0.75)

    def test_smoke_has_two_code_anchors(self, monkeypatch, tmp_path):
        """P 维度硬锚点路径被真实走过（确定性执行）."""
        code, out, path = _run_smoke(monkeypatch, tmp_path)
        records, _ = load_sessions(path)
        assert len(records[0].exec_results) == 2
        # loops-L3 sum_to 用错代码 -> 0 分；variables-L3 swap 用对 -> 1 分
        scores = {e.node_id: e.score for e in records[0].exec_results}
        assert scores["loops-L3-S3-P"] == 0.0
        assert scores["variables-L3-S3-P"] == 1.0

    def test_smoke_covers_expected_nodes(self, monkeypatch, tmp_path):
        code, out, path = _run_smoke(monkeypatch, tmp_path)
        records, _ = load_sessions(path)
        anchored = {t.anchor for t in records[0].transcript if t.anchor}
        assert anchored == {
            "loops-L1-S1-K", "loops-L2-S2-C", "loops-L3-S3-P", "loops-L4-S4-S",
            "variables-L1-S1-K", "variables-L2-S3-C", "variables-L3-S3-P",
            "variables-L4-S4-X",
        }

    def test_smoke_without_env_vars(self, monkeypatch, tmp_path):
        """smoke 不依赖任何环境变量（无 MINIMAX_* 也能跑）."""
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
        monkeypatch.delenv("MINIMAX_MODEL", raising=False)
        code, out, _ = _run_smoke(monkeypatch, tmp_path)
        assert code == 0
