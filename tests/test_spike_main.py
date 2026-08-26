"""spike CLI 题库锚点测验守卫测试：空答案重问、skip 不计入锚点."""

import pytest

from cogmirror.questions import QuestionBank

from spike.__main__ import _collect_bank_anchors, _read_bank_answer
from spike.dialogue import SKIP


@pytest.fixture
def bank():
    return QuestionBank()


class TestReadBankAnswer:
    def test_empty_choice_reprompts_then_answer(self, bank, monkeypatch):
        """选择题空输入 -> 重问 -> 作答."""
        q = bank.get("pl-l2-01")
        answers = iter(["", "1"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
        assert _read_bank_answer(q) == "1"

    def test_choice_skip_returns_sentinel(self, bank, monkeypatch):
        q = bank.get("pl-l2-01")
        monkeypatch.setattr("builtins.input", lambda *a, **k: "skip")
        assert _read_bank_answer(q) is SKIP

    def test_fill_empty_reprompts_then_skip(self, bank, monkeypatch):
        """填空空输入 -> 重问 -> skip 返回哨兵."""
        q = bank.get("pl-l1-01")
        answers = iter(["", "skip"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
        assert _read_bank_answer(q) is SKIP

    def test_code_empty_submission_returns_skip(self, bank, monkeypatch):
        """代码题只敲 END（没写代码）-> SKIP，不再硬判 0 分."""
        q = bank.get("pl-l3-01")
        monkeypatch.setattr("builtins.input", lambda *a, **k: "END")
        assert _read_bank_answer(q) is SKIP

    def test_code_real_submission_returns_code(self, bank, monkeypatch):
        q = bank.get("pl-l3-01")
        answers = iter(["def f(n):", "    return n", "END"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
        assert _read_bank_answer(q) == "def f(n):\n    return n"


class TestCollectBankAnchors:
    def test_skipped_questions_excluded_from_anchors(self, bank, monkeypatch):
        """skip 的题不进 answers 字典 -> 不计入分母（不是按错算）."""
        qs = [q for q in bank.all_questions() if q.topic == "python.loops"]
        skip_ids = {qs[0].problem_id, qs[2].problem_id}

        def fake_read(q):
            return SKIP if q.problem_id in skip_ids else "1"

        monkeypatch.setattr("spike.__main__._read_bank_answer", fake_read)
        gt = _collect_bank_anchors(bank, ["python.loops"])
        assert gt.per_topic_correct["python.loops"]["answered"] == len(qs) - len(skip_ids)
