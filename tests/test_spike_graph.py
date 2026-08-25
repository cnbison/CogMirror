"""技能图谱测试：完整性、锚点坐标系、检索 API、rubric."""

import pytest

from cogmirror.belief_state import BloomLevel

from spike.graph import (
    DimensionId,
    GraphNode,
    ProbeKind,
    SOLOLevel,
    TOPIC_ID_TO_SHORT,
    build_graph,
)


@pytest.fixture
def graph():
    return build_graph()


class TestGraphIntegrity:
    def test_20_nodes_unique_ids(self, graph):
        nodes = graph.all_nodes()
        assert len(nodes) == 20
        ids = [n.node_id for n in nodes]
        assert len(ids) == len(set(ids))

    def test_each_topic_covers_l1_l4(self, graph):
        from collections import defaultdict
        by_topic = defaultdict(set)
        for n in graph.all_nodes():
            by_topic[n.topic].add(n.bloom_level)
        assert len(by_topic) == 5
        for topic, levels in by_topic.items():
            assert levels == {BloomLevel.REMEMBER, BloomLevel.UNDERSTAND,
                              BloomLevel.APPLY, BloomLevel.ANALYZE}, topic

    def test_every_dimension_has_node(self, graph):
        dims = {n.dimension for n in graph.all_nodes()}
        assert dims == set(DimensionId)

    def test_each_topic_l3_has_code_anchor(self, graph):
        """P 维度硬锚点：每个 topic 的 L3 至少一个 CODE 节点（PRD 8b）."""
        from collections import defaultdict
        by_topic = defaultdict(list)
        for n in graph.all_nodes():
            by_topic[n.topic].append(n)
        for topic, nodes in by_topic.items():
            l3_code = [n for n in nodes
                       if n.bloom_level == BloomLevel.APPLY
                       and n.probe_kind == ProbeKind.CODE]
            assert l3_code, f"{topic} 缺少 L3 CODE 锚点"
            for n in l3_code:
                assert n.dimension == DimensionId.P

    def test_all_code_seeds_reference_bank_problems(self, graph):
        """CODE 节点 question_seed 必须是题库里的 problem_id（P 执行确定性）."""
        from cogmirror.questions import QuestionBank
        bank = QuestionBank()
        for n in graph.all_nodes():
            if n.probe_kind == ProbeKind.CODE:
                assert bank.get(n.question_seed) is not None, n.node_id


class TestNodeIdFormat:
    def test_node_id_parses_back(self, graph):
        """node_id 自描述：{topic}-L{bloom}-S{solo}-{dim}，可被 compare 反解."""
        for n in graph.all_nodes():
            parts = n.node_id.split("-")
            assert len(parts) == 4
            assert parts[1] == f"L{int(n.bloom_level.value)}"
            assert parts[2] == f"S{int(n.solo_level)}"
            assert parts[3] == n.dimension.value
            assert parts[0] in TOPIC_ID_TO_SHORT.values()


class TestSOLOLevel:
    def test_ordinal_order(self):
        assert int(SOLOLevel.PRE_STRUCTURAL) < int(SOLOLevel.UNI)
        assert int(SOLOLevel.UNI) < int(SOLOLevel.MULTI)
        assert int(SOLOLevel.MULTI) < int(SOLOLevel.RELATIONAL)
        assert int(SOLOLevel.RELATIONAL) < int(SOLOLevel.EXTENDED_ABSTRACT)
        assert SOLOLevel.RELATIONAL > SOLOLevel.MULTI  # IntEnum 可直接比较

    def test_labels_present(self):
        for level in SOLOLevel:
            assert level.label


class TestGraphRetrieval:
    def test_get_valid_node(self, graph):
        node = graph.get("loops-L3-S3-P")
        assert isinstance(node, GraphNode)
        assert node.dimension == DimensionId.P

    def test_get_invalid_raises_keyerror(self, graph):
        with pytest.raises(KeyError):
            graph.get("not-a-real-node")

    def test_has(self, graph):
        assert graph.has("loops-L3-S3-P")
        assert not graph.has("nope")

    def test_nodes_for_topic(self, graph):
        loops = graph.nodes_for_topic("python.loops")
        assert len(loops) == 4
        assert all(n.topic == "python.loops" for n in loops)

    def test_nodes_for_dimension(self, graph):
        p_nodes = graph.nodes_for_dimension(DimensionId.P)
        assert len(p_nodes) == 5
        assert all(n.dimension == DimensionId.P for n in p_nodes)

    def test_cx_probe_lookup(self, graph):
        probe = graph.cx_probe(DimensionId.C, ProbeKind.ANALOGY)
        assert probe is not None
        assert probe.success_evidence


class TestRubric:
    def test_rubric_contains_tc_signals(self, graph):
        text = graph.rubric_text()
        assert "liminal_signals" in text
        assert "crossing_indicators" in text
        assert "TC_python_loops" in text

    def test_rubric_contains_misconceptions(self, graph):
        text = graph.rubric_text()
        assert "Misconception" in text
        assert "M1" in text

    def test_rubric_contains_solo_and_bloom(self, graph):
        text = graph.rubric_text()
        assert "S1" in text
        assert "S5" in text
        assert "Bloom 目标成功标准" in text
