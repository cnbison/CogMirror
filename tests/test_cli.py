"""CLI 端到端集成测试（Phase 0 链路：做题 -> 5D 更新 -> 地图展示）."""

import argparse
import io
import sys

import pytest

from cogmirror import cli
from cogmirror.belief_state import IllusoryConfidenceHit, TCState


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


def test_map_includes_interpretation_section(monkeypatch, tmp_path):
    # 整体解读段渲染在地图标题下方，先结论后证据
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    idx = out.index("你的认知地图")
    assert "[整体解读]" in out[idx:]
    assert "整体掌握良好" in out
    assert "作答样本还少" in out
    assert "一句话建议" in out


def test_map_interpretation_no_history():
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    s = cli.map_interpretation(engine, state)
    assert "没有作答记录" in s


def test_map_interpretation_kp_gap_clause():
    # 知识记忆明显强于程序技能（K 0.9 / P 0.5）-> 解读段点出「概念懂、转代码还差一步」
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    state.K.mastery_prob = 0.9
    state.P.mastery_prob = 0.5
    engine.set_history("t1", [{"score": 1.0}, {"score": 1.0}, {"score": 0.4}])
    s = cli.map_interpretation(engine, state)
    assert "知识记忆" in s and "程序技能" in s
    assert "部分正确" in s
    assert "还在建立中" in s


def test_map_interpretation_p_over_k_clause():
    # 反向：程序技能强于知识记忆 -> 解读段如实指出
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    state.K.mastery_prob = 0.4
    state.P.mastery_prob = 0.8
    engine.set_history("t1", [{"score": 1.0}, {"score": 1.0}])
    s = cli.map_interpretation(engine, state)
    assert "程序技能" in s
    assert "超过背概念" in s


def test_map_interpretation_no_dim_claim_when_both_low():
    # 回归（真机发现）：全错学习者 K 0.24 / P 0.40 相对差距不小但都未掌握，
    # 不应被读成「程序技能更扎实/超过背概念」——维度结论只在"确实掌握"水平才下
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    state.K.mastery_prob = 0.24
    state.P.mastery_prob = 0.40
    engine.set_history("t1", [{"score": 0.0}, {"score": 0.0}])
    s = cli.map_interpretation(engine, state)
    assert "整体还比较薄弱" in s
    assert "超过背概念" not in s
    assert "还在建立中" not in s


def test_map_interpretation_illusory_clause():
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    state.C.illusory_confidence_hits.append(IllusoryConfidenceHit(
        problem_id="q1", self_confidence=0.9, score=0.0, gap=0.9))
    engine.set_history("t1", [{"score": 1.0}, {"score": 0.0}])
    s = cli.map_interpretation(engine, state)
    assert "1 处伪自信" in s
    assert "以为会了其实没会" in s


def test_map_interpretation_liminal_clause():
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    state.C.tc_states["python.loops"] = TCState(
        tc_id="TC_python_loops", status="liminal", progress=0.9)
    engine.set_history("t1", [{"score": 1.0}, {"score": 1.0}, {"score": 1.0}])
    s = cli.map_interpretation(engine, state)
    assert "循环" in s
    assert "正在跨越中" in s
    assert "中间态" in s


def _answer_loops(engine, state, score, level=cli.BloomLevel.APPLY):
    """模拟 run_session 单题流程：update + 捕获前置 TC 状态 + 返回逐题反馈行."""
    tc_before = state.C.tc_states.get("python.loops")
    prev = tc_before.status if tc_before else None
    n = len(engine.get_history("t1"))
    state = engine.update(state, cli.Observation(
        skill_id="python.loops", problem_id=f"tcq{n}", score=score,
        bloom_level=level, self_confidence=None, explanation_text=""))
    line = cli._liminal_live_feedback(engine, state, "python.loops", score, level, prev)
    return state, line


def test_liminal_live_feedback_enters_advances_crosses():
    # 3 次 L3+ 正确进入 liminal（前两次静默）-> 逐题推进剩余次数递减 -> 跨过报已跨越 -> 之后静默
    engine = cli.BeliefEngine()
    bank = cli.QuestionBank()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("t1")
    for i in range(2):
        state, line = _answer_loops(engine, state, 1.0)
        assert line == "", f"第 {i+1} 题未达 liminal 应静默: {line!r}"
    state, line = _answer_loops(engine, state, 1.0)
    assert "跨越进度 90%" in line and "再答对 3 次 L3+ 题即跨越" in line
    state, line = _answer_loops(engine, state, 1.0)
    assert "再答对 2 次 L3+ 题即跨越" in line
    state, line = _answer_loops(engine, state, 1.0)
    assert "再答对 1 次 L3+ 题即跨越" in line
    state, line = _answer_loops(engine, state, 1.0)
    assert "已跨越！恭喜" in line
    state, line = _answer_loops(engine, state, 1.0)
    assert line == "", "已跨越后再答不应重复庆祝"


def test_liminal_live_feedback_wrong_resets():
    engine = cli.BeliefEngine()
    bank = cli.QuestionBank()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("t1")
    for _ in range(3):
        state, _ = _answer_loops(engine, state, 1.0)  # 进 liminal（进度 90%）
    state, line = _answer_loops(engine, state, 0.0)
    assert "这次答错" in line
    assert "回落到" in line
    assert "这不是退步" in line


def test_liminal_live_feedback_low_level_silent():
    # liminal 中 L1 答对不推进临界概念 -> 不应误报"跨越进度"
    engine = cli.BeliefEngine()
    bank = cli.QuestionBank()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("t1")
    for _ in range(3):
        state, _ = _answer_loops(engine, state, 1.0)
    state, line = _answer_loops(engine, state, 1.0, level=cli.BloomLevel.REMEMBER)
    assert line == "", f"L1 答对不推进 liminal 应静默: {line!r}"


def test_liminal_live_feedback_unknown_skill_silent():
    engine = cli.BeliefEngine()
    state = engine.create_initial_state("t1")
    assert cli._liminal_live_feedback(
        engine, state, "python.nonexistent", 1.0, cli.BloomLevel.APPLY, None) == ""


def test_illusory_live_feedback_hit_and_silent():
    engine = cli.BeliefEngine()
    bank = cli.QuestionBank()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("t1")
    # 自评高 + 答错 -> 命中，当题点出
    before = len(state.C.illusory_confidence_hits)
    state = engine.update(state, cli.Observation(
        skill_id="python.variables", problem_id="pv-l1-01", score=0.0,
        bloom_level=cli.BloomLevel.REMEMBER, self_confidence=0.9, explanation_text=""))
    line = cli._illusory_live_feedback(state, before)
    assert "伪自信提示" in line
    assert "自评 90%" in line and "实际得分 0%" in line
    # 自评高 + 答对 -> 未命中，静默
    before = len(state.C.illusory_confidence_hits)
    state = engine.update(state, cli.Observation(
        skill_id="python.variables", problem_id="pv-l2-01", score=1.0,
        bloom_level=cli.BloomLevel.UNDERSTAND, self_confidence=0.9, explanation_text=""))
    line = cli._illusory_live_feedback(state, before)
    assert line == ""


def test_illusory_live_feedback_end_to_end(monkeypatch, tmp_path):
    # 自评 100 + 答错 2 题 -> 每题当题点出，地图汇总仍在
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["100\n", "0\n", "100\n", "0\n"],
        args=["--questions", "2"],
    )
    assert out.count("伪自信提示") == 2
    assert "自评 100%" in out
    assert "实际得分 0%" in out
    assert "发现 2 处失准" in out


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


def test_bloom_all_six_layers_light_up(monkeypatch, tmp_path):
    # L5/L6 题库补齐（2026-08-27）：六层全有题 -> 地图六层全亮，不再标注"暂无对应层级题目"
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    for label in ("L1 记忆", "L2 理解", "L3 应用", "L4 分析", "L5 评价", "L6 创造"):
        assert label in out
    assert "暂无对应层级题目" not in out


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
    # 用地图独有标记计数（"你的认知地图"也会出现在新用户欢迎语里，不能用来数渲染次数）
    assert out.count("当前主导层级") == 1, "拒绝后不应再渲染地图/出题"
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


def test_liminal_live_feedback_end_to_end(monkeypatch, tmp_path):
    # 逐题反馈端到端：主轮第 3 题进入 liminal（90%/3 次），练习轮第 3 题跨过（已跨越）
    answers = _LOOP_L3_ANSWERS + ["y\n"] + _LOOP_L3_ANSWERS + ["\n"]
    _, out = run_cli(
        monkeypatch, tmp_path, answers=answers,
        args=["--topic", "python.loops", "--level", "L3", "--questions", "3"],
    )
    # 逐题行（地图只在轮末重渲染，"再答对 2/1 次"只可能来自逐题反馈）
    assert "跨越进度 90%——再答对 3 次 L3+ 题即跨越" in out
    assert "再答对 2 次 L3+ 题即跨越" in out
    assert "再答对 1 次 L3+ 题即跨越" in out
    assert "已跨越！恭喜" in out


# ── 新用户上手引导 + 答题输入帮助（2026-08-27）─────────────────────


def test_new_user_shows_welcome(monkeypatch, tmp_path):
    # 首次运行：新用户见上手说明（怎么答各类题 / 怎么退出 / 怎么导出删除）
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "第一次用？很简单" in out
    assert "选择题输选项编号" in out
    assert "单独一行输入 END" in out
    assert "Ctrl-C" in out
    assert "cogmirror --export" in out


def test_choice_invalid_then_valid_reask(monkeypatch, tmp_path):
    # 无效选项编号 -> 重问而不是默默判 0；重问后答对得 1.00
    # pv-l1-01 有 4 个选项（0-3），答案 = 1；pv-l2-01 答案 = 2
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "9\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "请输入 0-3 之间的选项编号" in out
    assert out.count("得分: 1.00") == 2


# ── 合规数据命令（--export / --delete，2026-08-27）─────────────────


def _run_data_command(monkeypatch, db_path, args, stdin_text=""):
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    code = cli.main(args)
    return code, sys.stdout.getvalue()


def test_export_dumps_user_data(monkeypatch, tmp_path):
    from cogmirror.db import Database
    # 先跑一组题产生数据，再 --export -> JSON 含 user 与 responses
    run_cli(monkeypatch, tmp_path,
            answers=["80\n", "1\n", "90\n", "2\n"],
            args=["--questions", "2"])
    db_path = str(tmp_path / "cli.db")
    code, out = _run_data_command(monkeypatch, db_path, ["--user", "t1", "--db", db_path, "--export"])
    assert code == 0
    assert '"user_id": "t1"' in out
    assert '"problem_id": "pv-l1-01"' in out
    assert "导出请求" in out
    assert Database(db_path).get_user("t1")["data_export_requested_at"] is not None


def test_export_nonexistent_user(monkeypatch, tmp_path):
    db_path = str(tmp_path / "cli.db")
    code, out = _run_data_command(monkeypatch, db_path, ["--user", "ghost", "--db", db_path, "--export"])
    assert code == 0
    assert "不存在" in out


def test_delete_requires_confirmation(monkeypatch, tmp_path):
    from cogmirror.db import Database
    run_cli(monkeypatch, tmp_path,
            answers=["80\n", "1\n", "90\n", "2\n"],
            args=["--questions", "2"])
    db_path = str(tmp_path / "cli.db")
    code, out = _run_data_command(monkeypatch, db_path,
                                  ["--user", "t1", "--db", db_path, "--delete"], stdin_text="nope\n")
    assert code == 0
    assert "已取消" in out
    assert len(Database(db_path).load_responses("t1")) == 2, "未确认不应删除数据"


def test_delete_confirmed_clears_data(monkeypatch, tmp_path):
    from cogmirror.db import Database
    run_cli(monkeypatch, tmp_path,
            answers=["80\n", "1\n", "90\n", "2\n"],
            args=["--questions", "2"])
    db_path = str(tmp_path / "cli.db")
    code, out = _run_data_command(monkeypatch, db_path,
                                  ["--user", "t1", "--db", db_path, "--delete"], stdin_text="DELETE\n")
    assert code == 0
    assert "已删除" in out
    db = Database(db_path)
    assert db.load_responses("t1") == [], "确认后应清空作答"
    assert db.get_user("t1")["data_delete_requested_at"] is not None, "删除请求应记录在用户档案"


def test_delete_eof_is_cancel(monkeypatch, tmp_path):
    # 非交互/输入耗尽（EOF）-> 视为取消，不删除
    from cogmirror.db import Database
    run_cli(monkeypatch, tmp_path,
            answers=["80\n", "1\n", "90\n", "2\n"],
            args=["--questions", "2"])
    db_path = str(tmp_path / "cli.db")
    code, out = _run_data_command(monkeypatch, db_path,
                                  ["--user", "t1", "--db", db_path, "--delete"], stdin_text="")
    assert code == 0
    assert "已取消" in out
    assert len(Database(db_path).load_responses("t1")) == 2


# ── 认知地图可读性/呈现优化（2026-08-27）───────────────────────────


def test_map_has_reading_guide_and_percent(monkeypatch, tmp_path):
    # 标题下读图说明 + 数值改百分比；Bloom 标题补语义
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "（怎么看：每行条形" in out
    assert "[Bloom 六层分布]（各层掌握概率）" in out


def test_dominant_layer_shows_chinese_label(monkeypatch, tmp_path):
    # 主导层级显示中文层名（L2 理解）而非英文枚举名（UNDERSTAND）
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n", "90\n", "2\n"],
        args=["--questions", "2"],
    )
    assert "当前主导层级: L2 理解" in out
    assert "当前主导层级: UNDERSTAND" not in out


def test_bar_shows_percent_plain_when_not_tty(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert cli._bar(0.62).endswith("62%")
    assert "\x1b[" not in cli._bar(0.9)


def test_bar_colors_by_tier_when_tty(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert "\x1b[32m" in cli._bar(0.9)   # >=80% 绿
    assert "\x1b[33m" in cli._bar(0.7)   # >=60% 黄
    assert "\x1b[31m" in cli._bar(0.4)   # <60% 红
    assert "\x1b[0m" in cli._bar(0.9)    # 复位


def test_session_eof_midway_ends_gracefully(monkeypatch, tmp_path):
    # 输入流中途结束（EOF）：提前结束答题、不吐 traceback，仍渲染地图
    _, out = run_cli(
        monkeypatch, tmp_path,
        answers=["80\n", "1\n"],   # 只够答 1 题，第 2 题自评处 EOF
        args=["--questions", "3"],
    )
    assert "输入已结束" in out
    assert "Traceback" not in out
    assert "你的认知地图" in out
