"""静态 Python 基础题库 + 确定性 partial credit 判分.

ECOS 没有静态题库（题目由 LLM 生成、LLM 判分）。新项目 MVP 采用
静态题库 + 确定性判分（用户确认的方案）：结果可复现、无 API 依赖，
partial credit 由代码题的测试用例通过率自然产生。

MIRT 载荷（loadings）：每题声明其在 5D 维度上的区分度，这是让 5D
估计不退化为"全部相同"的必要条件--所有题共用默认参数时，后验在
维度间对称，五维 θ 必然相等（Phase 0 关卡问题实测发现）。

题型：
- choice: 单选，答对 1.0 / 否则 0.0
- fill: 填空，标准化文本匹配，1.0 / 0.0
- code: 写函数，本地运行测试用例，score = 通过用例数 / 总用例数
"""

from __future__ import annotations

import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .belief_state import BloomLevel
from .mirt import MIRTItemParams

DIM_CHARS = ("K", "P", "S", "C", "X")

# Bloom 层 -> MIRT 难度（L1 易，L4 难）
BLOOM_DIFFICULTY = {
    BloomLevel.REMEMBER: -1.0,
    BloomLevel.UNDERSTAND: -0.3,
    BloomLevel.APPLY: 0.3,
    BloomLevel.ANALYZE: 1.0,
}


@dataclass(frozen=True)
class TestCase:
    """code 题的单个测试用例."""

    __test__ = False  # 告知 pytest 这不是测试类

    args: tuple  # 位置参数
    kwargs: dict = field(default_factory=dict)
    expected: Any = None


@dataclass(frozen=True)
class Question:
    """单道题目.

    Attributes:
        problem_id: 唯一标识（同时用作 MIRT item id）
        skill_id: 关联知识点 ID（BKT / TC 粒度，如 python.variables）
        topic: 主题（python.variables 等 5 个）
        bloom_level: Bloom 层级
        qtype: "choice" / "fill" / "code"
        prompt: 题面
        options: choice 题的选项列表
        answer: choice 题的正确选项下标
        accepted: fill 题的可接受答案（标准化后匹配）
        func_name: code 题要求定义的函数名
        tests: code 题的测试用例
        loadings: 5D 载荷 {"K": 1.2, "P": 0.3, ...}（未列维度为 0）
        explanation: 判分后给用户看的知识点解释
    """

    problem_id: str
    skill_id: str
    topic: str
    bloom_level: BloomLevel
    qtype: str
    prompt: str
    loadings: dict[str, float]
    options: tuple[str, ...] = ()
    answer: int = -1
    accepted: tuple[str, ...] = ()
    func_name: str = ""
    tests: tuple[TestCase, ...] = ()
    explanation: str = ""


# ── 判分 ───────────────────────────────────────────────────────────


class GradingTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise GradingTimeout()


def _normalize(text: str) -> str:
    return "".join(text.split()).lower()


def grade_fill(question: Question, user_answer: str) -> float:
    if _normalize(user_answer) in {_normalize(a) for a in question.accepted}:
        return 1.0
    return 0.0


def grade_code(question: Question, user_code: str, timeout_sec: int = 5) -> tuple[float, list[dict]]:
    """运行测试用例判分，返回 (score, 每个用例的通过详情)."""
    # globals/locals 用同一个 dict：拆开会让用户函数 __globals__ 落在独立 dict，
    # 递归/全局名字查找失败（自测发现：正确的递归代码被判 NameError）
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(timeout_sec)
        try:
            exec(user_code, namespace, namespace)  # noqa: S102 - 本地单用户学习工具
        except GradingTimeout:
            return 0.0, [{"error": f"代码执行超时（>{timeout_sec}s），疑似死循环，所有用例未通过"}]
        except SyntaxError as e:
            return 0.0, [{"error": f"语法错误: {e}"}]
        except Exception as e:  # noqa: BLE001
            return 0.0, [{"error": f"定义阶段异常: {type(e).__name__}: {e}"}]

        func = namespace.get(question.func_name)
        if not callable(func):
            return 0.0, [{"error": f"未找到函数 {question.func_name}"}]

        details = []
        passed = 0
        for tc in question.tests:
            try:
                signal.alarm(timeout_sec)
                got = func(*tc.args, **tc.kwargs)
                signal.alarm(0)
                ok = got == tc.expected
            except GradingTimeout:
                details.append({"args": tc.args, "expected": tc.expected, "got": "超时", "passed": False})
                continue
            except Exception as e:  # noqa: BLE001
                details.append({"args": tc.args, "expected": tc.expected, "got": f"异常: {e}", "passed": False})
                continue
            if ok:
                passed += 1
            details.append({"args": tc.args, "expected": tc.expected, "got": got, "passed": ok})
        score = passed / len(question.tests) if question.tests else 0.0
        return score, details
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def grade(question: Question, user_answer: str, timeout_sec: int = 5) -> tuple[float, list[dict]]:
    """统一判分入口：返回 (score 0-1, 详情)."""
    if question.qtype == "choice":
        try:
            idx = int(user_answer.strip())
        except ValueError:
            return 0.0, [{"error": "无效选项"}]
        return (1.0 if idx == question.answer else 0.0), []
    if question.qtype == "fill":
        return grade_fill(question, user_answer), []
    if question.qtype == "code":
        return grade_code(question, user_answer, timeout_sec=timeout_sec)
    raise ValueError(f"未知题型: {question.qtype}")


# ── 题库 ───────────────────────────────────────────────────────────
# 载荷约定（K 知识 / P 程序技能 / S 策略-诊断能力）：
#   记忆/再认类题 -> K；写代码题 -> P；找错/分析题 -> S


def _bank() -> list[Question]:
    q: list[Question] = [
        # ─── 变量与赋值 ────────────────────────────────────────────
        Question(
            problem_id="pv-l1-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.REMEMBER, qtype="choice", loadings={"K": 1.2},
            prompt="下列哪个是合法的 Python 变量赋值语句？",
            options=("x == 5", "x = 5", "int x = 5", "x: = 5"), answer=1,
            explanation="Python 赋值用单个等号 =，== 是比较运算符。",
        ),
        Question(
            problem_id="pv-l2-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.UNDERSTAND, qtype="choice", loadings={"K": 1.0, "S": 0.4},
            prompt="执行 x = 3; x = x + 1 后，x 的值是？",
            options=("报错，等式不成立", "3", "4", "1"), answer=2,
            explanation="赋值先计算右边表达式（3+1=4），再把名字 x 绑定到 4。",
        ),
        Question(
            problem_id="pv-l2-02", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.UNDERSTAND, qtype="fill", loadings={"K": 1.0},
            prompt="执行 a = [1, 2]; b = a; b.append(3) 后，a 的值是什么？（按 Python 列表字面量书写，如 [1]）",
            accepted=("[1, 2, 3]",),
            explanation="b = a 让 b 与 a 指向同一个列表对象，append 对两者都可见。",
        ),
        Question(
            problem_id="pv-l3-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2, "K": 0.2},
            prompt="定义函数 swap_values(a, b)，返回交换后的元组 (b, a)。",
            func_name="swap_values",
            tests=(
                TestCase(args=(1, 2), expected=(2, 1)),
                TestCase(args=("x", "y"), expected=("y", "x")),
            ),
            explanation="Python 支持多重赋值 a, b = b, a 直接交换。",
        ),
        Question(
            problem_id="pv-l4-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="代码：a = [1, 2]; b = a[:]; b.append(3)。执行后 a 的值是？",
            options=("[1, 2, 3]", "[1, 2]", "报错", "不确定"), answer=1,
            explanation="a[:] 创建了新列表（浅拷贝），b 与 a 不再指向同一对象。",
        ),
        # ─── 循环 ──────────────────────────────────────────────────
        Question(
            problem_id="pl-l1-01", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.REMEMBER, qtype="fill", loadings={"K": 1.2},
            prompt="range(5) 依次产生哪些整数？（逗号分隔，如 0,1,2）",
            accepted=("0,1,2,3,4",),
            explanation="range(n) 从 0 开始到 n-1 结束，左闭右开，共 n 个数。",
        ),
        Question(
            problem_id="pl-l2-01", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.UNDERSTAND, qtype="choice", loadings={"K": 1.0},
            prompt="range(1, 5, 2) 产生的序列是？",
            options=("1, 3, 5", "1, 3", "1, 2, 3, 4, 5", "2, 4"), answer=1,
            explanation="start=1，stop=5（不含），step=2 -> 1, 3。",
        ),
        Question(
            problem_id="pl-l3-01", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="定义函数 sum_to(n)，用循环返回 1+2+...+n 的结果。",
            func_name="sum_to",
            tests=(
                TestCase(args=(1,), expected=1),
                TestCase(args=(10,), expected=55),
                TestCase(args=(100,), expected=5050),
            ),
            explanation="累加器初值为 0，循环内 total += i。",
        ),
        Question(
            problem_id="pl-l3-02", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.1, "S": 0.3},
            prompt="定义函数 max_of(nums)，返回列表中的最大值（不用内置 max）。",
            func_name="max_of",
            tests=(
                TestCase(args=([3, 7, 2, 9],), expected=9),
                TestCase(args=([5],), expected=5),
                TestCase(args=([-1, -5, -3],), expected=-1),
            ),
            explanation="用第一个元素做初始最大值，逐个比较更新。",
        ),
        Question(
            problem_id="pl-l4-01", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="下列 while 循环为什么是死循环？\n    i = 0\n    while i < 5:\n        print(i)",
            options=(
                "while 的条件写错了",
                "循环体内没有更新 i，i 永远是 0，条件永远为真",
                "print 不能用在 while 里",
                "i 应该初始化为 1",
            ), answer=1,
            explanation="循环三要素：初始化、终止条件、状态更新。缺状态更新必然死循环。",
        ),
        # ─── 函数 ──────────────────────────────────────────────────
        Question(
            problem_id="pf-l1-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.REMEMBER, qtype="choice", loadings={"K": 1.2},
            prompt="定义带返回值函数的正确语法是？",
            options=(
                "def f(x): return x * 2",
                "function f(x) { return x * 2 }",
                "def f(x) -> return x * 2",
                "define f(x): return x * 2",
            ), answer=0,
            explanation="Python 用 def 定义函数，return 返回值。",
        ),
        Question(
            problem_id="pf-l2-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.UNDERSTAND, qtype="choice", loadings={"K": 1.0, "S": 0.4},
            prompt="def f(x): print(x * 2)，则 y = f(3) 之后 y 的值是？",
            options=("6", "None", "报错", "3"), answer=1,
            explanation="print 只是输出副作用，没有 return 的函数返回 None。",
        ),
        Question(
            problem_id="pf-l3-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="定义函数 is_even(n)：n 为偶数返回 True，奇数返回 False。",
            func_name="is_even",
            tests=(
                TestCase(args=(2,), expected=True),
                TestCase(args=(7,), expected=False),
                TestCase(args=(0,), expected=True),
            ),
            explanation="n % 2 == 0 判断偶数。",
        ),
        Question(
            problem_id="pf-l3-02", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.1, "S": 0.3},
            prompt="定义函数 count_vowels(s)，返回字符串中元音字母（a/e/i/o/u，不区分大小写）的个数。",
            func_name="count_vowels",
            tests=(
                TestCase(args=("hello",), expected=2),
                TestCase(args=("AEIOU",), expected=5),
                TestCase(args=("",), expected=0),
                TestCase(args=("xyz",), expected=0),
            ),
            explanation="遍历字符串，逐字符判断是否在 'aeiou'（统一小写后）中。",
        ),
        Question(
            problem_id="pf-l4-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="def f(lst): lst.append(4)。执行 a = [1]; f(a) 后 a 的值是？",
            options=("[1]", "[1, 4]", "None", "报错"), answer=1,
            explanation="列表作为参数传的是引用，函数内 append 影响调用方的列表。",
        ),
        # ─── 递归 ──────────────────────────────────────────────────
        Question(
            problem_id="pr-l1-01", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.REMEMBER, qtype="choice", loadings={"K": 1.2},
            prompt="递归函数必须包含哪两个要素？",
            options=(
                "for 循环和 if 判断",
                "函数调用自身 + 基准情形（base case）",
                "print 语句和 return 语句",
                "全局变量和局部变量",
            ), answer=1,
            explanation="没有基准情形的递归会无限调用直至栈溢出。",
        ),
        Question(
            problem_id="pr-l2-01", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.UNDERSTAND, qtype="choice", loadings={"K": 1.0},
            prompt="递归和循环的本质区别是？",
            options=(
                "递归更快",
                "递归把问题化为同类子问题，循环重复执行代码块",
                "递归不需要终止条件",
                "没有区别",
            ), answer=1,
            explanation="递归核心是化归：分解为更小的同类子问题直到基准情形。",
        ),
        Question(
            problem_id="pr-l3-01", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="用递归定义函数 factorial(n)，返回 n!（0! = 1）。",
            func_name="factorial",
            tests=(
                TestCase(args=(0,), expected=1),
                TestCase(args=(1,), expected=1),
                TestCase(args=(5,), expected=120),
            ),
            explanation="基准情形 n <= 1 返回 1，否则返回 n * factorial(n-1)。",
        ),
        Question(
            problem_id="pr-l4-01", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="def f(n): return f(n-1) 调用 f(3) 会发生什么？",
            options=(
                "返回 0",
                "返回 None",
                "RecursionError（超过最大递归深度）",
                "正常结束",
            ), answer=2,
            explanation="没有基准情形，调用栈不断加深直到超过默认深度限制（约 1000）。",
        ),
        # ─── 作用域 ────────────────────────────────────────────────
        Question(
            problem_id="ps-l1-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.REMEMBER, qtype="fill", loadings={"K": 1.2},
            prompt="Python 名字查找顺序 LEGB 的四个字母分别代表哪四个作用域？（中文顿号分隔）",
            accepted=("局部、封闭、全局、内建", "局部、嵌套、全局、内建"),
            explanation="Local -> Enclosing -> Global -> Builtin。",
        ),
        Question(
            problem_id="ps-l2-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.UNDERSTAND, qtype="choice", loadings={"K": 1.0, "S": 0.4},
            prompt="x = 10\ndef f():\n    x = 5\nf()\nprint(x) 输出什么？",
            options=("5", "10", "报错", "None"), answer=1,
            explanation="函数内的 x = 5 创建的是局部变量，不影响全局 x。",
        ),
        Question(
            problem_id="ps-l3-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="定义函数 make_counter()：每次调用返回的函数计数加一并返回新计数值（利用闭包，从 1 开始计）。",
            func_name="make_counter",
            explanation="闭包内用 nonlocal 修改外层计数变量。",
        ),
        Question(
            problem_id="ps-l4-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="funcs = [lambda: i for i in range(3)]，则 [f() for f in funcs] 的结果是？",
            options=("[0, 1, 2]", "[2, 2, 2]", "[0, 0, 0]", "报错"), answer=1,
            explanation="闭包捕获的是变量 i 本身而非当时的值；循环结束后 i = 2。（注：lambda 参数默认值 i=i 可修复）",
        ),
        # ─── 变量与赋值（L3+ 扩充，2026-08-26：F10 三态端到端可达） ──
        Question(
            problem_id="pv-l3-02", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.APPLY, qtype="choice", loadings={"K": 1.0, "S": 0.3},
            prompt="执行 x = 3; x = x + 1 时，Python 实际先做什么？",
            options=(
                "先把右边 3 + 1 算出来，再把名字 x 绑定到结果 4",
                "先把 x 复制一份再修改",
                "报错，x 不能出现在等号两边",
                "先比较 x 和 x + 1 是否相等",
            ), answer=0,
            explanation="赋值语句先计算右边的表达式，再把左边的名字绑定到结果，所以 x = x + 1 合法且常见。",
        ),
        Question(
            problem_id="pv-l3-03", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="定义函数 repeat_word(s)，返回 s 重复两次的字符串（如 repeat_word('ab') 返回 'abab'）。",
            func_name="repeat_word",
            tests=(
                TestCase(args=("ab",), expected="abab"),
                TestCase(args=("x",), expected="xx"),
                TestCase(args=("",), expected=""),
            ),
            explanation="字符串乘法 s * 2 或 s + s 都能实现重复。",
        ),
        Question(
            problem_id="pv-l4-02", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="a = [1, 2]\nb = a\nb = b + [3]\n执行后 a 的值是？",
            options=("[1, 2, 3]", "[1, 2]", "报错", "[3]"), answer=1,
            explanation="b + [3] 创建了新的列表，b 重新绑定到新列表；a 仍指向原列表，不受影响。",
        ),
        Question(
            problem_id="pv-l4-03", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="a = [1, 2]\nb = a\nb += [3]\n执行后 a 的值是？",
            options=("[1, 2, 3]", "[1, 2]", "报错", "None"), answer=0,
            explanation="列表的 += 是原地修改（等价于 extend），b 与 a 共享同一对象，所以 a 也变成 [1, 2, 3]。",
        ),
        # ─── 循环（L3+ 扩充）───────────────────────────────────────
        Question(
            problem_id="pl-l3-03", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="定义函数 count_even(nums)，返回列表中偶数的个数。",
            func_name="count_even",
            tests=(
                TestCase(args=([1, 2, 3, 4],), expected=2),
                TestCase(args=([5, 7],), expected=0),
                TestCase(args=([2, 4, 6],), expected=3),
            ),
            explanation="遍历列表，用 n % 2 == 0 判断偶数并计数。",
        ),
        Question(
            problem_id="pl-l3-04", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.1, "S": 0.3},
            prompt="定义函数 sum_range(a, b)，返回从 a 到 b（含两端）所有整数的和。",
            func_name="sum_range",
            tests=(
                TestCase(args=(1, 3), expected=6),
                TestCase(args=(5, 5), expected=5),
                TestCase(args=(1, 100), expected=5050),
            ),
            explanation="range(a, b+1) 含两端；累加器求和。",
        ),
        Question(
            problem_id="pl-l4-02", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.0, "K": 0.4},
            prompt="执行 for i in range(0, 10, 3): print(i)，依次输出哪些数？",
            options=("0, 3, 6, 9", "0, 3, 6, 9, 12", "1, 4, 7, 10", "3, 6, 9"), answer=0,
            explanation="range(0, 10, 3) 从 0 开始、步长 3、不含 10，所以是 0, 3, 6, 9。",
        ),
        # ─── 函数（L3+ 扩充）───────────────────────────────────────
        Question(
            problem_id="pf-l3-03", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="定义函数 first_last(nums)，返回元组 (第一个元素, 最后一个元素)。",
            func_name="first_last",
            tests=(
                TestCase(args=([3, 7, 2, 9],), expected=(3, 9)),
                TestCase(args=(["a", "b"],), expected=("a", "b")),
                TestCase(args=([5],), expected=(5, 5)),
            ),
            explanation="用索引 0 和 -1 取首尾元素，包成元组返回。",
        ),
        Question(
            problem_id="pf-l3-04", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.1, "S": 0.3},
            prompt="定义函数 sum_list(nums)，返回列表中所有元素的和（空列表返回 0）。",
            func_name="sum_list",
            tests=(
                TestCase(args=([1, 2, 3],), expected=6),
                TestCase(args=([],), expected=0),
                TestCase(args=([-1, 1],), expected=0),
            ),
            explanation="累加器从 0 开始遍历求和。",
        ),
        Question(
            problem_id="pf-l4-02", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="def f(x, lst=[]):\n    lst.append(x)\n    return lst\n连续调用 f(1)、f(2)、f(3) 后，f(3) 的返回值是？",
            options=("[3]", "[1, 2, 3]", "[1]", "报错"), answer=1,
            explanation="默认参数列表只在函数定义时创建一次，多次调用共享同一个列表对象，元素会跨调用累积。",
        ),
        # ─── 递归（L3+ 扩充）───────────────────────────────────────
        Question(
            problem_id="pr-l3-02", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="用递归定义函数 fib(n)，返回第 n 个斐波那契数（fib(0)=0, fib(1)=1）。",
            func_name="fib",
            tests=(
                TestCase(args=(0,), expected=0),
                TestCase(args=(1,), expected=1),
                TestCase(args=(10,), expected=55),
            ),
            explanation="基准情形 n <= 1 返回 n；否则返回 fib(n-1) + fib(n-2)。",
        ),
        Question(
            problem_id="pr-l3-03", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.APPLY, qtype="choice", loadings={"K": 1.0},
            prompt="下列哪个递归函数会无限递归（缺少基准情形）？",
            options=(
                "def f(n):\n    return 1 if n <= 1 else f(n - 1)",
                "def f(n):\n    return f(n - 1)",
                "def f(n):\n    return 0 if n == 0 else n + f(n - 1)",
                "def f(n):\n    return n if n == 0 else f(n - 1)",
            ), answer=1,
            explanation="第二个函数没有任何基准情形，无论 n 是多少都继续调用 f(n-1)，最终栈溢出。",
        ),
        Question(
            problem_id="pr-l4-02", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="def g(n):\n    return 0 if n == 0 else 1 + g(n - 1)\n调用 g(3) 时，调用栈最深时有多少层？",
            options=("3 层", "4 层", "1 层", "无限"), answer=1,
            explanation="g(3)→g(2)→g(1)→g(0) 共 4 层，到 g(0) 返回 0 后再逐层展开。",
        ),
        Question(
            problem_id="pr-l4-03", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.0, "K": 0.3},
            prompt="下列哪个问题用递归解决最自然？",
            options=("遍历一棵树的所有节点", "打印 1 到 100", "计算两个整数之和", "交换两个变量的值"), answer=0,
            explanation="树的结构天然是递归的（每个子树都是一棵树），递归遍历最自然；其余问题迭代即可。",
        ),
        # ─── 作用域（L3+ 扩充）─────────────────────────────────────
        Question(
            problem_id="ps-l3-02", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="在代码顶部定义模块级变量 count = 0，再定义函数 step()：用 global 声明修改全局 count，每次调用加 1 并返回新值（第一次返回 1，第二次 2，第三次 3）。",
            func_name="step",
            tests=(
                TestCase(args=(), expected=1),
                TestCase(args=(), expected=2),
                TestCase(args=(), expected=3),
            ),
            explanation="函数内用 global count 声明后，count += 1 修改的就是模块级变量，且跨调用保持。",
        ),
        Question(
            problem_id="ps-l3-03", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.APPLY, qtype="choice", loadings={"K": 1.0, "S": 0.4},
            prompt="x = 5\ndef f():\n    global x\n    x = x + 1\nf()\nprint(x) 输出什么？",
            options=("5", "6", "报错", "None"), answer=1,
            explanation="global 声明让函数内修改的是全局 x，所以 x 变为 6。",
        ),
        Question(
            problem_id="ps-l4-02", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="def outer():\n    n = 0\n    def inner():\n        nonlocal n\n        n += 1\n        return n\n    return inner\nf = outer()\n连续调用 f() 三次，第三次的返回值是？",
            options=("1", "2", "3", "报错"), answer=2,
            explanation="nonlocal 让 inner 修改外层 outer 的 n，且跨调用保持，三次调用分别返回 1、2、3。",
        ),
        Question(
            problem_id="ps-l4-03", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.3},
            prompt="x = 10\ndef f():\n    print(x)\n    x = 5\nf() 会发生什么？",
            options=("输出 10", "输出 5", "UnboundLocalError", "NameError"), answer=2,
            explanation="函数内对 x 赋值使 x 成为局部变量，print(x) 时局部 x 尚未赋值，触发 UnboundLocalError。",
        ),
    ]
    return q


# make_counter 是闭包题，测试用例结构特殊（返回值是函数），单独处理
def _grade_make_counter(user_code: str) -> tuple[float, list[dict]]:
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(user_code, ns, ns)  # noqa: S102
        make = ns.get("make_counter")
        counter = make()
        results = [counter(), counter(), counter()]
        ok = results == [1, 2, 3]
        return (1.0 if ok else 0.0), [{"expected": [1, 2, 3], "got": results, "passed": ok}]
    except Exception as e:  # noqa: BLE001
        return 0.0, [{"error": f"{type(e).__name__}: {e}"}]


class QuestionBank:
    """静态题库 + MIRT 题目参数注册."""

    def __init__(self) -> None:
        self._questions = _bank()
        self._by_id = {q.problem_id: q for q in self._questions}
        assert len(self._questions) == len(self._by_id), "题目 ID 重复"

    def all_questions(self) -> list[Question]:
        return list(self._questions)

    def get(self, problem_id: str) -> Optional[Question]:
        return self._by_id.get(problem_id)

    def by_topic(self, topic: str) -> list[Question]:
        return [q for q in self._questions if q.topic == topic]

    def mirt_items(self) -> list[MIRTItemParams]:
        """题目 -> MIRT 参数（载荷向量 + Bloom 难度）."""
        items = []
        for q in self._questions:
            a = np.array([float(q.loadings.get(d, 0.0)) for d in DIM_CHARS])
            if a.sum() <= 0:
                a = np.ones(5) * 0.8  # 未声明载荷的兜底
            items.append(MIRTItemParams(
                problem_id=q.problem_id,
                a_specialized=a,
                a_general=0.5,
                difficulty=BLOOM_DIFFICULTY.get(q.bloom_level, 0.0),
            ))
        return items

    def grade_answer(self, question: Question, user_answer: str, timeout_sec: int = 5) -> tuple[float, list[dict]]:
        """判分入口；闭包题走专用校验."""
        if question.problem_id == "ps-l3-01":
            return _grade_make_counter(user_answer)
        return grade(question, user_answer, timeout_sec=timeout_sec)
