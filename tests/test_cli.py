"""CLI 端到端集成测试（Phase 0 链路：做题 -> 5D 更新 -> 地图展示）."""

import io
import sys

import pytest

from cogmirror import cli


def run_cli(monkeypatch, tmp_path, answers, args):
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(answers)))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    db_path = str(tmp_path / "cli.db")
    code = cli.main(["--user", "t1", "--db", db_path, *args])
    out = sys.stdout.getvalue()
    return code, out


def test_session_then_map(monkeypatch, tmp_path):
    # 前 2 题：pv-l1-01（选 1，对）、pv-l2-01（选 2，对）
    code, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert code == 0
    assert "你的认知地图" in out
    assert "[5 维状态]" in out
    assert "[Bloom 六层分布]" in out
    assert "得分: 1.00" in out
    assert "[一句话建议]" in out


def test_map_only_and_restore(monkeypatch, tmp_path):
    run_cli(
        monkeypatch, tmp_path,
        answers=["50\n", "0\n"],
        args=["--questions", "1"],
    )
    # 第二次以 map-only 进入：应恢复历史并只展示地图
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    db_path = str(tmp_path / "cli.db")
    code = cli.main(["--user", "t1", "--db", db_path, "--map-only"])
    out = sys.stdout.getvalue()
    assert code == 0
    assert "已完成 1 次作答" in out
    assert "你的认知地图" in out


def test_illusory_confidence_shown_in_map(monkeypatch, tmp_path):
    # 自评 100 但答错（选项 0 错误）-> 伪自信标注
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["100\n", "0\n"],
        args=["--questions", "1"],
    )
    assert "伪自信点" in out
    assert "pv-l1-01" in out
