"""CLI 端到端集成测试（Phase 0 链路：做题 -> 5D 更新 -> 地图展示）."""

import argparse
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


def test_all_illusory_hits_listed_and_count_matches(monkeypatch, tmp_path):
    # 回归（自测弱学习者）：C 维度说"发现 N 处失准"但列表只显示 5 处（[-5:] 截断）。
    # 6 题全部自评 100 + 答错 -> 6 处命中应全部列出，首尾都在。
    answers = []
    # 每题先自评 100，再给错误答案
    answers += ["100\n", "0\n"]   # pv-l1-01 choice 错
    answers += ["100\n", "0\n"]   # pv-l2-01 choice 错
    answers += ["100\n", "wrong\n"]  # pv-l2-02 fill 错
    answers += ["100\n", "END\n"]    # pv-l3-01 code 空
    answers += ["100\n", "0\n"]   # pv-l4-01 choice 错
    answers += ["100\n", "wrong\n"]  # pl-l1-01 fill 错
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=answers,
        args=["--questions", "6"],
    )
    assert "发现 6 处失准" in out
    assert "题 pv-l1-01" in out
    assert "题 pv-l4-01" in out
    assert "题 pl-l1-01" in out


def test_topic_label_chinese_in_suggestion(monkeypatch, tmp_path):
    # 回归：一句话建议露出原始英文 id（如 python.recursion）新手看不懂
    # 前 5 题全属 python.variables 且全错 -> BKT 最弱是 variables -> 建议该 topic
    # （Q4 是代码题，需要 END 结束符）
    answers = [
        "80\n", "0\n",     # pv-l1-01 choice 错
        "80\n", "0\n",     # pv-l2-01 choice 错
        "80\n", "wrong\n",  # pv-l2-02 fill 错
        "80\n", "END\n",    # pv-l3-01 code 空
        "80\n", "0\n",     # pv-l4-01 choice 错
    ]
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=answers,
        args=["--questions", "5"],
    )
    suggest_lines = [l for l in out.splitlines() if l.strip().startswith("「")]
    assert suggest_lines, "未找到建议行"
    assert "变量赋值" in suggest_lines[0], f"建议应为中文 topic 名: {suggest_lines[0]}"
    assert "python." not in suggest_lines[0], "建议行不应出现原始 skill id"


def test_topic_label_unit():
    assert cli._topic_label("python.recursion") == "递归"
    assert cli._topic_label("python.variables") == "变量赋值"
    assert cli._topic_label("unknown.topic") == "unknown.topic"


def test_parse_level_aliases():
    assert cli._parse_level("L3") == cli.BloomLevel.APPLY
    assert cli._parse_level("apply") == cli.BloomLevel.APPLY
    assert cli._parse_level("APPLY") == cli.BloomLevel.APPLY
    assert cli._parse_level("L1") == cli.BloomLevel.REMEMBER
    with pytest.raises(argparse.ArgumentTypeError):
        cli._parse_level("L9")


@pytest.mark.parametrize("level", ["L3", "APPLY", "apply"])
def test_topic_level_filter(monkeypatch, tmp_path, level):
    # F10 收口：--topic/--level 让「建议做 3 道某 topic 的 L3 题」真正可执行。
    # 前两道 loops-L3 均为代码题（pl-l3-01 sum_to、pl-l3-02 max_of），用 END 结束输入。
    answers = [
        "80\n",
        "def sum_to(n):\n", "    total = 0\n", "    for i in range(1, n + 1):\n", "        total += i\n", "    return total\n",
        "END\n",
        "80\n",
        "def max_of(nums):\n", "    m = nums[0]\n", "    for x in nums:\n", "        if x > m:\n", "            m = x\n", "    return m\n",
        "END\n",
    ]
    _, out = run_cli(
        monkeypatch, tmp_path, answers=answers,
        args=["--topic", "python.loops", "--level", level, "--questions", "2"],
    )
    assert "题 1/2" in out and "题 2/2" in out
    assert "pl-l3-01" in out and "pl-l3-02" in out
    assert "pl-l1-01" not in out, "不应出现 loops L1 题"
    assert "pv-l1-01" not in out, "不应出现其他 topic 的题"


def test_post_liminal_shown_in_map(monkeypatch):
    # F10 收口：临界概念「已跨越」态要在认知地图里给确认文案。
    # 引擎连答 6 次 L3+ 正确（真实 TC 路径）构造 post_liminal 状态后渲染地图。
    engine = cli.BeliefEngine()
    bank = cli.QuestionBank()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("t1")
    for i in range(6):
        state = engine.update(
            state, cli.Observation(
                skill_id="python.loops", problem_id=f"tcq{i}", score=1.0,
                bloom_level=cli.BloomLevel.APPLY, self_confidence=None,
                explanation_text="",
            ))
    assert state.C.tc_states["python.loops"].status == "post_liminal"

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    cli.print_map(engine, state)
    out = sys.stdout.getvalue()
    assert "已跨越" in out
    assert "循环是受控的重复" in out


# 交互流测试：主会话 3 道 loops-L3 全对 -> liminal；练习轮再 3 道全对 -> 已跨越。
# loops-L3 前 3 道按题库顺序：pl-l3-01(sum_to)/pl-l3-02(max_of)/pl-l3-03(count_even)，均代码题。
_LOOP_L3_ANSWERS = [
    "80\n",
    "def sum_to(n):\n", "    total = 0\n", "    for i in range(1, n + 1):\n", "        total += i\n", "    return total\n",
    "END\n",
    "80\n",
    "def max_of(nums):\n", "    m = nums[0]\n", "    for x in nums:\n", "        if x > m:\n", "            m = x\n", "    return m\n",
    "END\n",
    "80\n",
    "def count_even(nums):\n", "    c = 0\n", "    for n in nums:\n", "        if n % 2 == 0:\n", "            c += 1\n", "    return c\n",
    "END\n",
]


def test_practice_accept_flow(monkeypatch, tmp_path):
    # 地图末尾接受建议（y）-> 进入练习轮，liminal 概念跨过后重渲染「已跨越」地图
    answers = _LOOP_L3_ANSWERS + ["y\n"] + _LOOP_L3_ANSWERS + ["\n"]
    _, out = run_cli(
        monkeypatch, tmp_path, answers=answers,
        args=["--topic", "python.loops", "--level", "L3", "--questions", "3"],
    )
    assert "正在跨越中" in out, "主会话后应见 liminal"
    assert "已跨越" in out, "练习轮后应见 post_liminal"
    assert out.count("你的认知地图") >= 2, "应渲染两次地图（主会话 + 练习轮）"
    assert "循环是受控的重复" in out


def test_practice_decline_flow(monkeypatch, tmp_path):
    # 地图末尾拒绝（n）-> 不再出题，只渲染一次地图
    answers = _LOOP_L3_ANSWERS + ["n\n"]
    _, out = run_cli(
        monkeypatch, tmp_path, answers=answers,
        args=["--topic", "python.loops", "--level", "L3", "--questions", "3"],
    )
    assert "正在跨越中" in out
    assert out.count("你的认知地图") == 1, "拒绝后不应再渲染地图/出题"
    assert "已跨越" not in out


def test_suggestion_liminal_includes_remaining():
    # 建议把"跨越进度"翻译成可行动步骤：已连续答对 N 次，再对 1 次即跨越
    engine = cli.BeliefEngine()
    bank = cli.QuestionBank()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("t1")
    for i in range(5):  # 前 3 次进 liminal，后 2 次 streak=2 -> 再对 1 次即跨越
        state = engine.update(
            state, cli.Observation(
                skill_id="python.loops", problem_id=f"tcq{i}", score=1.0,
                bloom_level=cli.BloomLevel.APPLY, self_confidence=None,
                explanation_text="",
            ))
    s = cli.next_suggestion(engine, state)
    assert "循环" in s
    assert "正在跨越中" in s
    assert "再答对 1 次 L3+ 题即跨越" in s


def test_suggestion_weakest_includes_rationale():
    # 最弱分支带"为什么"：说明是这个 topic 是当前最弱的一项
    engine = cli.BeliefEngine()
    bank = cli.QuestionBank()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("t1")
    state = engine.update(
        state, cli.Observation(
            skill_id="python.variables", problem_id="tcq1", score=0.0,
            bloom_level=cli.BloomLevel.REMEMBER, self_confidence=None,
            explanation_text="",
        ))
    s = cli.next_suggestion(engine, state)
    assert "变量赋值" in s
    assert "当前最弱" in s


def test_practice_prompt_includes_remaining(monkeypatch, tmp_path):
    # 练习交互提示带剩余次数：主轮 3 对进 liminal（streak=0）-> 再对 3 次即跨越
    _, out = run_cli(
        monkeypatch, tmp_path, answers=_LOOP_L3_ANSWERS + ["n\n"],
        args=["--topic", "python.loops", "--level", "L3", "--questions", "3"],
    )
    assert "再答对 3 次 L3+ 题即跨越" in out
