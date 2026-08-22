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
    assert "发现 1 处失准" in out


def test_c_dimension_calibrated_when_no_illusory(monkeypatch, tmp_path):
    # 自评与表现一致（无失准）-> C 显示"未发现失准"，不再给误导性数值
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["90\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "未发现失准" in out
    assert "置信度" in out


def test_c_dimension_no_selfconf_data(monkeypatch, tmp_path):
    # 从未填自评 -> C 显示"暂无自评数据"
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["\n", "1\n", "\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "暂无自评数据" in out


def test_x_dimension_annotated_unmeasured(monkeypatch, tmp_path):
    # X 维度 MVP 无支架/提示机制，应诚实标注而非显示先验假数值
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "外部支架" in out
    assert "暂未测量" in out


def test_bloom_l56_annotated(monkeypatch, tmp_path):
    # 题库最高只到 L4，L5/L6 无对应题目 -> 标注而非显示永远不变的 0.50
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "L5 评价" in out
    assert "L6 创造" in out
    assert "暂无对应层级题目" in out


def test_tc_display_name_uses_state_machine_library():
    # TC 显示名唯一来源 = 状态机库，避免 content 库文案漂移
    engine = cli.BeliefEngine()
    assert cli._tc_display_name(engine, "python.loops") == "循环是受控的重复"
    assert cli._tc_display_name(engine, "python.scope") == "作用域是名字的查找规则"
    assert cli._tc_display_name(engine, "不存在的topic") == "不存在的topic"
