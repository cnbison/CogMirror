"""评分器测试：用户侧视角剥离、JSON 容错解析、P 只认执行结果."""

import json

import pytest

from cogmirror.belief_state import BloomLevel

from spike.dialogue import AnchorTurn, ExecResult
from spike.graph import DimensionId, build_graph
from spike.llm import FakeLLM
from spike.scorer import (
    build_user_perspective_transcript,
    parse_scorer_output,
    score_session,
)


class TestUserPerspectiveTranscript:
    def test_strips_anchor_and_strategy(self):
        """用户侧视角必须剔除 anchor 节点 id，评分器无法反推面试协议."""
        transcript = [
            AnchorTurn(role="assistant", anchor="loops-L3-S3-P",
                       text="请写一个累加函数。"),
            AnchorTurn(role="user", anchor=None, text="def f(n): return n"),
            AnchorTurn(role="system", anchor=None, text="【代码执行结果】未通过"),
        ]
        text = build_user_perspective_transcript(transcript)
        assert "loops-L3-S3-P" not in text
        assert "请写一个累加函数。" in text
        assert "【代码执行结果】" in text

    def test_preserves_order(self):
        transcript = [
            AnchorTurn(role="assistant", anchor="a", text="q1"),
            AnchorTurn(role="user", anchor=None, text="a1"),
            AnchorTurn(role="assistant", anchor="b", text="q2"),
            AnchorTurn(role="user", anchor=None, text="a2"),
        ]
        text = build_user_perspective_transcript(transcript)
        assert text.index("q1") < text.index("a1") < text.index("q2") < text.index("a2")


class TestParseScorerOutput:
    def test_normal_json(self):
        raw = json.dumps({
            "five_d": {"K": 0.6, "P": 0.4, "S": 0.5, "C": 0.5, "X": 0.5},
            "bloom": {"REMEMBER": 0.7, "UNDERSTAND": 0.6,
                      "APPLY": 0.4, "ANALYZE": 0.5},
            "solo": {"loops": 3.0},
            "overall": 0.55,
            "insufficient": [],
        })
        out = parse_scorer_output(raw)
        assert out.five_d[DimensionId.K] == 0.6
        assert out.bloom[BloomLevel.UNDERSTAND] == 0.6
        assert out.solo["loops"] == 3.0
        assert out.overall == 0.55
        assert out.insufficient == []
        assert "parse_warning" not in out.evidence_notes

    def test_json_block_fallback(self):
        raw = '前面有文字\n```json\n{"five_d": {"K": 0.9, "P": 0.1, "S": 0.2, "C": 0.3, "X": 0.4}, "bloom": {"REMEMBER": 0.5, "UNDERSTAND": 0.5, "APPLY": 0.5, "ANALYZE": 0.5}, "overall": 0.4, "insufficient": []}\n```\n后面有文字'
        out = parse_scorer_output(raw)
        assert out.five_d[DimensionId.K] == 0.9
        assert "parse_warning" in out.evidence_notes

    def test_text_kv_fallback(self):
        raw = "K=0.7 P=0.4 整体感觉一般 X=0.3"
        out = parse_scorer_output(raw)
        assert out.five_d[DimensionId.K] == 0.7
        assert out.five_d[DimensionId.P] == 0.4
        assert out.five_d[DimensionId.X] == 0.3
        assert "parse_warning" in out.evidence_notes

    def test_clamp_values(self):
        raw = json.dumps({
            "five_d": {"K": 1.5, "P": -0.2, "S": 0.5, "C": 0.5, "X": 0.5},
            "bloom": {"REMEMBER": 2.0, "UNDERSTAND": -1.0,
                      "APPLY": 0.5, "ANALYZE": 0.5},
            "overall": 3.0,
            "insufficient": [],
        })
        out = parse_scorer_output(raw)
        assert out.five_d[DimensionId.K] == 1.0
        assert out.five_d[DimensionId.P] == 0.0
        assert out.bloom[BloomLevel.REMEMBER] == 1.0
        assert out.bloom[BloomLevel.UNDERSTAND] == 0.0
        assert out.overall == 1.0

    def test_bad_input_no_raise(self):
        out = parse_scorer_output("完全不是 JSON 也没有 K= 这样的键值")
        assert out.insufficient == ["parsing_failed"]
        assert "parse_warning" in out.evidence_notes

    def test_thinking_preamble_json_extract(self):
        """MiniMax-M3 思维链前导：content 以 thinking 开头，JSON 在末尾——须能解析."""
        raw = (' thinkingThe user wants me to grade the learner. response\n'
               '\n{"five_d": {"K": 0.5, "P": 0.4, "S": 0.5, "C": 0.5, "X": 0.5}, '
               '"bloom": {"REMEMBER": 0.5, "UNDERSTAND": 0.5, '
               '"APPLY": 0.5, "ANALYZE": 0.5}, "solo": {}, '
               '"overall": 0.5, "insufficient": []}')
        out = parse_scorer_output(raw)
        assert out.five_d[DimensionId.K] == 0.5
        assert out.overall == 0.5
        assert out.insufficient == []
        assert "json_extract" in out.evidence_notes["parse_warning"]

    def test_missing_fields_added_to_insufficient(self):
        raw = json.dumps({
            "five_d": {"K": 0.5},
            "bloom": {"REMEMBER": 0.5},
            "overall": 0.5,
            "insufficient": [],
        })
        out = parse_scorer_output(raw)
        # 缺失的 P/S/C/X 与 UNDERSTAND/APPLY/ANALYZE 计入 insufficient
        assert set(out.insufficient) >= {"P", "S", "C", "X",
                                         "UNDERSTAND", "APPLY", "ANALYZE"}
        assert out.five_d[DimensionId.P] == 0.0


class TestScoreSession:
    def test_score_session_with_fake(self):
        """FakeLLM 走完整评分链路：system 含 rubric，输出被解析."""

        def responder(system, user):
            assert "评卷判据" in system
            assert "loops-L3-S3-P" not in user  # 用户侧视角不含 anchor
            return json.dumps({
                "five_d": {"K": 0.5, "P": 0.5, "S": 0.5, "C": 0.5, "X": 0.5},
                "bloom": {"REMEMBER": 0.5, "UNDERSTAND": 0.5,
                          "APPLY": 0.5, "ANALYZE": 0.5},
                "solo": {},
                "overall": 0.5,
                "insufficient": [],
            })

        graph = build_graph()
        transcript = [AnchorTurn(role="assistant", anchor="loops-L3-S3-P",
                                 text="请写累加函数"),
                      AnchorTurn(role="user", anchor=None, text="def f(): pass")]
        exec_results = [ExecResult(node_id="loops-L3-S3-P", submitted_code="x",
                                   score=0.0, details=[], executed=True)]
        out = score_session(FakeLLM(responder), graph, transcript, exec_results)
        assert out.overall == 0.5
        assert "exec_results" in out.evidence_notes

    def test_covered_topic_without_solo_goes_to_insufficient(self):
        """对话覆盖到 recursion，但评分器没给 solo——确定性兜底计入 insufficient."""

        def responder(system, user):
            return json.dumps({
                "five_d": {"K": 0.5, "P": 0.5, "S": 0.5, "C": 0.5, "X": 0.5},
                "bloom": {"REMEMBER": 0.5, "UNDERSTAND": 0.5,
                          "APPLY": 0.5, "ANALYZE": 0.5},
                "solo": {"loops": 3.0},  # 只有 loops，缺 recursion
                "overall": 0.5,
                "insufficient": [],
            })

        graph = build_graph()
        transcript = [
            AnchorTurn(role="assistant", anchor="loops-L1-S1-K", text="q1"),
            AnchorTurn(role="user", anchor=None, text="a1"),
            AnchorTurn(role="assistant", anchor="recursion-L1-S1-K", text="q2"),
            AnchorTurn(role="user", anchor=None, text="a2"),
        ]
        out = score_session(FakeLLM(responder), graph, transcript)
        assert "solo:recursion" in out.insufficient
        assert "solo:loops" not in out.insufficient
        assert "missing_solo" in out.evidence_notes
