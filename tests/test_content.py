"""第1类直接迁移文件的独立可用性测试（脱离 ECOS 主框架）."""

import pytest

from cogmirror.belief_state import BloomLevel
from cogmirror.content import (
    PythonBasicsBloomLibrary,
    PythonBasicsMisconceptionLibrary,
    PythonThresholdConceptLibrary,
    PYTHON_BASICS_MISCONCEPTION_LIBRARY_STR,
)


class TestBloomLibrary:
    def test_20_entries(self):
        lib = PythonBasicsBloomLibrary()
        entries = lib.all_entries()
        assert len(entries) == 20

    def test_goal_ids_unique(self):
        lib = PythonBasicsBloomLibrary()
        ids = [e.goal_id for e in lib.all_entries()]
        assert len(ids) == len(set(ids))

    def test_5_topics_4_levels(self):
        lib = PythonBasicsBloomLibrary()
        for topic in ("python.variables", "python.loops", "python.functions",
                      "python.recursion", "python.scope"):
            goals = lib.get_goals_by_topic(topic)
            assert len(goals) == 4, topic
            levels = {g.bloom_level for g in goals}
            assert levels == {BloomLevel.REMEMBER, BloomLevel.UNDERSTAND,
                              BloomLevel.APPLY, BloomLevel.ANALYZE}

    def test_get_by_id(self):
        lib = PythonBasicsBloomLibrary()
        e = lib.get("python.variables-L2")
        assert e is not None
        assert e.bloom_level == BloomLevel.UNDERSTAND
        assert lib.get("nonexistent") is None

    def test_prerequisite_chain_resolvable(self):
        lib = PythonBasicsBloomLibrary()
        for e in lib.all_entries():
            for pre in e.prerequisite_goals:
                assert lib.get(pre) is not None, f"{e.goal_id} -> {pre} 悬空"


class TestMisconceptionLibrary:
    def test_8_entries_unique_ids(self):
        lib = PythonBasicsMisconceptionLibrary()
        entries = lib.all_entries()
        assert len(entries) == 8
        ids = [e.misc_id for e in entries]
        assert len(ids) == len(set(ids))
        assert set(ids) == {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"}

    def test_get(self):
        lib = PythonBasicsMisconceptionLibrary()
        assert lib.get("M3").name == "for 循环 off-by-one"
        assert lib.get("M9") is None

    def test_keyword_detection(self):
        lib = PythonBasicsMisconceptionLibrary()
        hit = lib.detect_by_keywords("range(5) 应该是 0 到 5 吧？最后一个怎么不对")
        assert hit is not None and hit.misc_id == "M3"
        assert lib.detect_by_keywords("我完全明白了") is None

    def test_library_str_nonempty(self):
        assert "M8" in PYTHON_BASICS_MISCONCEPTION_LIBRARY_STR


class TestThresholdConceptLibrary:
    def test_5_python_entries(self):
        lib = PythonThresholdConceptLibrary()
        entries = lib.all_entries()
        assert len(entries) == 5
        # 全部是 Python 学科条目（去非 Python 条目的迁移要求）
        for e in entries:
            assert all(s.startswith("python.") for s in e.skill_ids), e.tc_id

    def test_ids_unique_and_lookup(self):
        lib = PythonThresholdConceptLibrary()
        ids = lib.all_tc_ids()
        assert len(ids) == len(set(ids))
        assert lib.get("TC_python_variables") is not None
        assert lib.get("TC_function") is None  # ECOS 数学条目不得混入

    def test_liminal_signals_present(self):
        lib = PythonThresholdConceptLibrary()
        for e in lib.all_entries():
            assert len(e.liminal_signals) >= 2, e.tc_id
            assert e.pre_conception and e.post_conception
