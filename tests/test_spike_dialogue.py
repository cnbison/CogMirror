"""锚定对话引擎测试：anchor 强制、P 确定性执行、transcript 顺序、轮次上限."""

import json

import pytest

from cogmirror.questions import QuestionBank

from spike.dialogue import (
    DialogueEngine,
    _parse_interviewer_response,
    extract_json_object,
)
from spike.graph import build_graph
from spike.llm import FakeLLM

LOOPS_TOPIC = "python.loops"
ALL_LOOPS = ["loops-L1-S1-K", "loops-L2-S2-C", "loops-L3-S3-P", "loops-L4-S4-S"]


@pytest.fixture
def graph():
    return build_graph()


@pytest.fixture
def bank():
    return QuestionBank()


def _engine(graph, bank, responder, topics=(LOOPS_TOPIC,), max_rounds=30):
    return DialogueEngine(FakeLLM(responder), graph, bank, list(topics),
                          max_rounds=max_rounds)


def _valid_responder(question_text="请回答。"):
    """永远返回合法 anchor（回显给定的节点）."""

    def responder(system, user):
        if "评卷判据" in system:
            return "{}"
        node_id = user.split("本轮要探测的节点 id：")[1].split("\n")[0]
        return json.dumps({"anchor": node_id, "question": question_text})

    return responder


class TestAnchorEnforcement:
    def test_invalid_anchor_retried_once_then_succeeds(self, graph, bank):
        """第一次返回非法 anchor -> 重试一次后成功，不跳节点."""
        calls = {"n": 0}

        def responder(system, user):
            if "评卷判据" in system:
                return "{}"
            calls["n"] += 1
            node_id = user.split("本轮要探测的节点 id：")[1].split("\n")[0]
            if calls["n"] == 1:
                return json.dumps({"anchor": "not-a-real-node", "question": "x"})
            return json.dumps({"anchor": node_id, "question": "请回答。"})

        engine = _engine(graph, bank, responder)
        state = engine.run("u1", ask=lambda node: "答案")
        # 第一个非 CODE 节点被重试一次（2 次调用），全部节点仍被覆盖
        assert state.covered_nodes == set(ALL_LOOPS)
        assert state.skipped_anchors == []
        # 3 个非 CODE 节点 = 2（L1 重试）+ 1（L2）+ 1（L4）
        assert calls["n"] == 4

    def test_double_invalid_skipped_and_recorded(self, graph, bank):
        """两次都非法 -> 记录 skipped 并跳过该节点，避免死循环."""

        def responder(system, user):
            return json.dumps({"anchor": "still-invalid", "question": "x"})

        engine = _engine(graph, bank, responder)
        state = engine.run("u1", ask=lambda node: "答案")
        # 3 个非 CODE 节点被跳过，CODE 节点（L3）正常覆盖
        assert set(state.skipped_anchors) == {"loops-L1-S1-K", "loops-L2-S2-C",
                                              "loops-L4-S4-S"}
        assert "loops-L3-S3-P" in state.covered_nodes


class TestExtractJsonObject:
    def test_extracts_balanced_object_from_preamble(self):
        raw = (' thinking... response\n'
               '{"anchor": "loops-L1-S1-K", "question": "请解释 range(5) 的输出。"}')
        assert extract_json_object(raw) == (
            '{"anchor": "loops-L1-S1-K", "question": "请解释 range(5) 的输出。"}')

    def test_braces_inside_string_not_confused(self):
        raw = '前导 {"a": "}", "b": 1} 结尾'
        assert extract_json_object(raw) == '{"a": "}", "b": 1}'

    def test_nested_objects(self):
        raw = 'x {"outer": {"inner": 1}, "k": 2}'
        assert extract_json_object(raw) == '{"outer": {"inner": 1}, "k": 2}'

    def test_no_object_returns_none(self):
        assert extract_json_object("完全没有 JSON 花括号") is None


class TestParseInterviewerResponse:
    def test_thinking_preamble_parsed(self):
        """MiniMax-M3 思维链前导 + JSON 在末尾——anchor/question 仍须提取出来."""
        raw = (' thinkingThe learner seems uncertain. response\n'
               '{"anchor": "loops-L2-S2-C", "question": "range(1,5) 和 range(5) 有何不同？"}')
        assert _parse_interviewer_response(raw) == {
            "anchor": "loops-L2-S2-C",
            "question": "range(1,5) 和 range(5) 有何不同？",
        }

    def test_garbage_returns_none(self):
        assert _parse_interviewer_response("完全不是 JSON 也没有花括号") is None


class TestTrivialQuestionGuard:
    def test_trivial_question_retried_then_succeeds(self, graph, bank):
        """第一次返回 '...' 退化提问 -> 重试一次后成功，节点不被跳过."""
        calls = {"n": 0}

        def responder(system, user):
            if "评卷判据" in system:
                return "{}"
            calls["n"] += 1
            node_id = user.split("本轮要探测的节点 id：")[1].split("\n")[0]
            if calls["n"] == 1:
                return json.dumps({"anchor": node_id, "question": "..."})
            return json.dumps({"anchor": node_id, "question": "请回答。"})

        engine = _engine(graph, bank, responder)
        state = engine.run("u1", ask=lambda node: "答案")
        assert state.covered_nodes == set(ALL_LOOPS)
        assert state.skipped_anchors == []
        assert calls["n"] == 4  # L1 退化重试(2) + L2(1) + L4(1)

    def test_persistent_trivial_question_skipped(self, graph, bank):
        """该节点一直返回 '...' -> 跳过并记录；其余节点不受影响."""

        def responder(system, user):
            node_id = user.split("本轮要探测的节点 id：")[1].split("\n")[0]
            if node_id == "loops-L1-S1-K":
                return json.dumps({"anchor": node_id, "question": "..."})
            return json.dumps({"anchor": node_id, "question": "请回答。"})

        engine = _engine(graph, bank, responder)
        state = engine.run("u1", ask=lambda node: "答案")
        assert "loops-L1-S1-K" in state.skipped_anchors
        assert "loops-L1-S1-K" not in state.covered_nodes
        # 其余节点正常覆盖
        assert {"loops-L2-S2-C", "loops-L3-S3-P", "loops-L4-S4-S"} <= state.covered_nodes


RECURSION_TOPIC = "python.recursion"


def _recursion_engine(graph, bank, responder):
    return _engine(graph, bank, responder, topics=(RECURSION_TOPIC,))


class TestPCodeExecution:
    def test_factorial_correct_code_full_score(self, graph, bank):
        engine = _recursion_engine(graph, bank, _valid_responder())
        correct = ("def factorial(n):\n"
                   "    if n <= 1:\n"
                   "        return 1\n"
                   "    return n * factorial(n-1)")
        state = engine.run("u1", ask=lambda node: correct if node.node_id ==
                           "recursion-L3-S4-P" else "答案")
        er = [e for e in state.exec_results
              if e.node_id == "recursion-L3-S4-P"][0]
        q = bank.get("pr-l3-01")
        expected, _ = bank.grade_answer(q, correct)
        assert er.score == expected == 1.0
        assert er.executed

    def test_factorial_wrong_code_partial_score(self, graph, bank):
        """P 用确定性判分：错误代码得分与 grade_answer 完全一致."""
        engine = _recursion_engine(graph, bank, _valid_responder())
        wrong = "def factorial(n):\n    return 1"
        state = engine.run("u1", ask=lambda node: wrong if node.node_id ==
                           "recursion-L3-S4-P" else "答案")
        er = [e for e in state.exec_results
              if e.node_id == "recursion-L3-S4-P"][0]
        q = bank.get("pr-l3-01")
        expected, _ = bank.grade_answer(q, wrong)
        assert er.score == expected
        assert 0.0 < er.score < 1.0  # 2/3 用例通过

    def test_score_not_in_transcript(self, graph, bank):
        """score 数值只进数据不进 prompt：transcript 的 P 结果块不含得分."""
        engine = _recursion_engine(graph, bank, _valid_responder())
        wrong = "def factorial(n):\n    return 1"
        state = engine.run("u1", ask=lambda node: wrong if node.node_id ==
                           "recursion-L3-S4-P" else "答案")
        system_text = "".join(t.text for t in state.transcript
                              if t.role == "system")
        assert "得分" not in system_text
        assert "0.67" not in system_text


class TestTranscriptInvariant:
    def test_no_consecutive_assistant_turns(self, graph, bank):
        engine = _engine(graph, bank, _valid_responder())
        state = engine.run("u1", ask=lambda node: "答案")
        for i in range(len(state.transcript) - 1):
            assert not (state.transcript[i].role == "assistant"
                        and state.transcript[i + 1].role == "assistant")

    def test_covered_nodes_match_transcript_anchors(self, graph, bank):
        engine = _engine(graph, bank, _valid_responder())
        state = engine.run("u1", ask=lambda node: "答案")
        anchored = {t.anchor for t in state.transcript if t.anchor}
        assert anchored == state.covered_nodes


class TestRoundLimit:
    def test_max_rounds_stops(self, graph, bank):
        engine = _engine(graph, bank, _valid_responder(),
                         topics=(LOOPS_TOPIC, "python.variables"), max_rounds=2)
        state = engine.run("u1", ask=lambda node: "答案")
        # 2 轮：第一个 topic 覆盖 2 个节点，第二个 topic 一个都没覆盖
        assert len(state.covered_nodes) == 2
        assert all("loops" in n for n in state.covered_nodes)

    def test_full_coverage_terminates(self, graph, bank):
        engine = _engine(graph, bank, _valid_responder(),
                         topics=(LOOPS_TOPIC, "python.variables"))
        state = engine.run("u1", ask=lambda node: "答案")
        assert state.covered_nodes == {
            "loops-L1-S1-K", "loops-L2-S2-C", "loops-L3-S3-P", "loops-L4-S4-S",
            "variables-L1-S1-K", "variables-L2-S3-C", "variables-L3-S3-P",
            "variables-L4-S4-X",
        }
