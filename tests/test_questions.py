"""静态题库与确定性 partial credit 判分测试."""

import numpy as np
import pytest

from cogmirror.belief_state import BloomLevel
from cogmirror.questions import QuestionBank, TestCase, grade, grade_code


@pytest.fixture
def bank() -> QuestionBank:
    return QuestionBank()


class TestBank:
    def test_ids_unique_and_nonempty(self, bank):
        qs = bank.all_questions()
        assert len(qs) >= 20
        ids = [q.problem_id for q in qs]
        assert len(ids) == len(set(ids))

    def test_covers_all_topics_and_levels(self, bank):
        topics = {q.topic for q in bank.all_questions()}
        assert topics == {"python.variables", "python.loops", "python.functions",
                          "python.recursion", "python.scope"}
        levels = {q.bloom_level for q in bank.all_questions()}
        assert levels >= {BloomLevel.REMEMBER, BloomLevel.UNDERSTAND,
                          BloomLevel.APPLY, BloomLevel.ANALYZE}

    def test_mirt_items_loadings_differ(self, bank):
        """载荷矩阵必须维度间可分，否则 5D 估计退化（Phase 0 实测教训）."""
        items = bank.mirt_items()
        assert len(items) == len(bank.all_questions())
        a_matrix = np.array([it.a_specialized for it in items])
        # 不同题目的载荷向量不能全部相同
        assert len({tuple(np.round(a, 6)) for a in a_matrix}) > 1
        # K/P/S 至少各被一组题加载
        k_load = a_matrix[:, 0].max()
        p_load = a_matrix[:, 1].max()
        s_load = a_matrix[:, 2].max()
        assert k_load > 0 and p_load > 0 and s_load > 0


class TestGrading:
    def test_choice_right_wrong(self, bank):
        q = bank.get("pv-l1-01")
        assert grade(q, "1")[0] == 1.0
        assert grade(q, "0")[0] == 0.0
        assert grade(q, "abc")[0] == 0.0

    def test_fill_normalization(self, bank):
        q = bank.get("pl-l1-01")
        assert grade(q, "0, 1, 2, 3, 4")[0] == 1.0
        assert grade(q, "0,1,2,3,4,5")[0] == 0.0

    def test_code_full_and_zero(self, bank):
        q = bank.get("pl-l3-01")
        good = "def sum_to(n):\n    total = 0\n    for i in range(1, n+1):\n        total += i\n    return total"
        score, details = bank.grade_answer(q, good)
        assert score == 1.0
        score, _ = bank.grade_answer(q, "def sum_to(n):\n    return 0")
        assert score == 0.0

    def test_code_recursion_global_lookup(self, bank):
        """回归：递归代码的全局名字查找（exec 拆 globals/locals 曾误判 NameError）."""
        q = bank.get("pr-l3-01")
        good = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"
        score, details = bank.grade_answer(q, good)
        assert score == 1.0, details

    def test_code_partial_credit(self, bank):
        """核心场景：代码逻辑对但有小瑕疵（部分用例失败）不得判全错."""
        q = bank.get("pl-l3-01")
        # 只处理了 n>=1 的累加但忘了 n=0 的语义边界（返回了 n 而非和）——用例 2 过、其余错
        partial = "def sum_to(n):\n    return 55 if n == 10 else 0"
        score, details = bank.grade_answer(q, partial)
        assert 0.0 < score < 1.0
        assert score == pytest.approx(1 / 3)

    def test_code_syntax_error(self, bank):
        q = bank.get("pf-l3-01")
        score, details = bank.grade_answer(q, "def is_even(n)\n    return True")
        assert score == 0.0
        assert any("语法错误" in str(d.get("error", "")) for d in details)

    def test_code_missing_function(self, bank):
        q = bank.get("pf-l3-01")
        score, details = bank.grade_answer(q, "x = 1")
        assert score == 0.0
        assert any("未找到函数" in str(d.get("error", "")) for d in details)

    def test_code_timeout_infinite_loop(self, bank):
        q = bank.get("pl-l3-01")
        bad = "def sum_to(n):\n    while True:\n        pass"
        score, details = bank.grade_answer(q, bad, timeout_sec=1)
        assert score == 0.0
        assert any("超时" in str(d) for d in details)

    def test_closure_question(self, bank):
        q = bank.get("ps-l3-01")
        good = (
            "def make_counter():\n"
            "    count = 0\n"
            "    def counter():\n"
            "        nonlocal count\n"
            "        count += 1\n"
            "        return count\n"
            "    return counter"
        )
        score, _ = bank.grade_answer(q, good)
        assert score == 1.0
        bad = "def make_counter():\n    return lambda: 1"
        score, details = bank.grade_answer(q, bad)
        assert score == 0.0

    def test_runtime_exception_in_tests(self, bank):
        q = bank.get("pf-l3-01")
        score, details = bank.grade_answer(q, "def is_even(n):\n    return 10 / n == 0 or n % 2 == 0")
        assert 0.0 <= score < 1.0  # n=0 用例会 ZeroDivisionError
