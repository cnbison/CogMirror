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

    def test_covers_all_six_bloom_levels(self, bank):
        """L5/L6 补齐（2026-08-27）：六层全链覆盖，认知地图六层全亮."""
        levels = {q.bloom_level for q in bank.all_questions()}
        assert levels == set(BloomLevel)

    def test_each_topic_has_l5_and_l6(self, bank):
        for topic in ("python.variables", "python.loops", "python.functions",
                      "python.recursion", "python.scope"):
            tq = bank.by_topic(topic)
            l5 = sum(1 for q in tq if q.bloom_level == BloomLevel.EVALUATE)
            l6 = sum(1 for q in tq if q.bloom_level == BloomLevel.CREATE)
            assert l5 >= 1, f"{topic} 缺 L5 题"
            assert l6 >= 1, f"{topic} 缺 L6 题"

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

    def test_code_references_exist_and_pass(self, bank):
        """答错揭晓正解（web 辅导链路）：每道 code 题都有参考答案，
        且参考答案能通过自身全部测试用例（含 make_counter/make_adder
        特殊判分与 global 题的跨调用状态）。"""
        code_qs = [q for q in bank.all_questions() if q.qtype == "code"]
        assert len(code_qs) >= 19
        for q in code_qs:
            assert q.reference, f"{q.problem_id} 缺参考答案"
            score, details = bank.grade_answer(q, q.reference)
            assert score == 1.0, f"{q.problem_id} 参考答案未满分: {details}"

    def test_each_topic_has_six_l3_plus(self, bank):
        """F10 收口：每 topic 至少 6 道 L3+，且保留 L1/L2 基础题.

        临界概念三态端到端可达的前提：pre->liminal 需 3 次 L3+ 正确、
        liminal->post 需再连续 3 次（共 6 次），因此每 topic 需 >=6 道 L3+。
        """
        for topic in ("python.variables", "python.loops", "python.functions",
                      "python.recursion", "python.scope"):
            tq = bank.by_topic(topic)
            l3_plus = sum(1 for q in tq if q.bloom_level.value >= BloomLevel.APPLY.value)
            assert l3_plus >= 6, f"{topic} 只有 {l3_plus} 道 L3+，不足 6"
            l1 = sum(1 for q in tq if q.bloom_level == BloomLevel.REMEMBER)
            l2 = sum(1 for q in tq if q.bloom_level == BloomLevel.UNDERSTAND)
            assert l1 >= 1 and l2 >= 1, f"{topic} 缺少 L1/L2 基础题"

    def test_every_choice_question_has_option_explanations(self, bank):
        """④ 题目讲解深度：所有选择题逐选项讲解齐全且与选项一一对应."""
        choice = [q for q in bank.all_questions() if q.qtype == "choice"]
        assert choice, "题库应包含选择题"
        for q in choice:
            assert len(q.option_explanations) == len(q.options), \
                f"{q.problem_id} 讲解数 {len(q.option_explanations)} != 选项数 {len(q.options)}"
            assert all(e.strip() for e in q.option_explanations), \
                f"{q.problem_id} 存在空讲解"
        # 非选择题不应误带逐选项讲解
        for q in bank.all_questions():
            if q.qtype != "choice":
                assert not q.option_explanations, f"{q.problem_id} 非选择题不应有逐选项讲解"


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

    def test_make_adder_closure_factory(self, bank):
        """L6 闭包工厂题：返回值是函数，走专用判分器."""
        q = bank.get("ps-l6-01")
        good = (
            "def make_adder(n):\n"
            "    def adder(x):\n"
            "        return n + x\n"
            "    return adder"
        )
        score, _ = bank.grade_answer(q, good)
        assert score == 1.0
        bad = "def make_adder(n):\n    return lambda x: x"
        score, details = bank.grade_answer(q, bad)
        assert score == 0.0

    @pytest.mark.parametrize(
        ("problem_id", "correct_idx"),
        [
            ("pv-l5-01", 2),
            ("pl-l5-01", 1),
            ("pf-l5-01", 2),
            ("pr-l5-01", 1),
            ("ps-l5-01", 3),
        ],
    )
    def test_l5_choice_questions_grade(self, bank, problem_id, correct_idx):
        q = bank.get(problem_id)
        assert q is not None, f"{problem_id} 不存在"
        assert q.qtype == "choice" and q.bloom_level == BloomLevel.EVALUATE
        assert bank.grade_answer(q, str(correct_idx))[0] == 1.0
        assert bank.grade_answer(q, str((correct_idx + 1) % len(q.options)))[0] == 0.0

    @pytest.mark.parametrize(
        ("problem_id", "correct_code"),
        [
            ("pv-l6-01", 'def dedupe(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out'),
            ("pl-l6-01", 'def make_star_triangle(n):\n    return "\\n".join("*" * i for i in range(1, n + 1))'),
            ("pf-l6-01", "def apply_twice(f, x):\n    return f(f(x))"),
            ("pr-l6-01", "def reverse_str(s):\n    if len(s) <= 1:\n        return s\n    return reverse_str(s[1:]) + s[0]"),
            ("ps-l6-01", "def make_adder(n):\n    def adder(x):\n        return n + x\n    return adder"),
        ],
    )
    def test_l6_code_questions_grade_full(self, bank, problem_id, correct_code):
        """L6 创造题（2026-08-27 补齐）：已知正确解必须判 1.0."""
        q = bank.get(problem_id)
        assert q is not None, f"{problem_id} 不存在"
        assert q.qtype == "code" and q.bloom_level == BloomLevel.CREATE
        score, details = bank.grade_answer(q, correct_code)
        assert score == 1.0, f"{problem_id} 正确解判分错误: {details}"

    def test_runtime_exception_in_tests(self, bank):
        q = bank.get("pf-l3-01")
        score, details = bank.grade_answer(q, "def is_even(n):\n    return 10 / n == 0 or n % 2 == 0")
        assert 0.0 <= score < 1.0  # n=0 用例会 ZeroDivisionError

    @pytest.mark.parametrize(
        ("problem_id", "correct_code"),
        [
            ("pv-l3-03", "def repeat_word(s):\n    return s * 2"),
            ("pl-l3-03", "def count_even(nums):\n    c = 0\n    for n in nums:\n        if n % 2 == 0:\n            c += 1\n    return c"),
            ("pl-l3-04", "def sum_range(a, b):\n    total = 0\n    for i in range(a, b + 1):\n        total += i\n    return total"),
            ("pf-l3-03", "def first_last(nums):\n    return (nums[0], nums[-1])"),
            ("pf-l3-04", "def sum_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total"),
            ("pr-l3-02", "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)"),
            ("ps-l3-02", "count = 0\ndef step():\n    global count\n    count += 1\n    return count"),
        ],
    )
    def test_new_code_questions_grade_full(self, bank, problem_id, correct_code):
        """F10 新增 code 题：已知正确解必须判 1.0（含 global 计数跨用例持久）. """
        q = bank.get(problem_id)
        assert q is not None, f"{problem_id} 不存在"
        score, details = bank.grade_answer(q, correct_code)
        assert score == 1.0, f"{problem_id} 正确解判分错误: {details}"
