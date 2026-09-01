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
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .belief_state import BloomLevel
from .mirt import MIRTItemParams

DIM_CHARS = ("K", "P", "S", "C", "X")

# Bloom 层 -> MIRT 难度（L1 易，L6 难；2026-08-27 补 L5/L6 打通六层全链）
BLOOM_DIFFICULTY = {
    BloomLevel.REMEMBER: -1.0,
    BloomLevel.UNDERSTAND: -0.3,
    BloomLevel.APPLY: 0.3,
    BloomLevel.ANALYZE: 1.0,
    BloomLevel.EVALUATE: 1.6,
    BloomLevel.CREATE: 2.2,
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
        explanation: 判分后给用户看的知识点解释（fill/code 用）
        option_explanations: choice 题每个选项的讲解（长度与 options 一致）
        reference: code 题的参考答案（答错后揭晓；必须能通过自身全部测试用例，
            由 test_questions 的回归测试保证）
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
    option_explanations: tuple[str, ...] = ()
    reference: str = ""


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


def _alarm_enabled() -> bool:
    """signal.alarm 只在主线程可用：非主线程（如嵌入方在请求线程里判分）
    优雅降级为无超时保护，而不是抛 ValueError（web 真机发现）."""
    return threading.current_thread() is threading.main_thread()


def grade_code(question: Question, user_code: str, timeout_sec: int = 5) -> tuple[float, list[dict]]:
    """运行测试用例判分，返回 (score, 每个用例的通过详情)."""
    # globals/locals 用同一个 dict：拆开会让用户函数 __globals__ 落在独立 dict，
    # 递归/全局名字查找失败（自测发现：正确的递归代码被判 NameError）
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    alarm_on = _alarm_enabled()
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler) if alarm_on else None
    try:
        if alarm_on:
            signal.alarm(timeout_sec)
        try:
            exec(user_code, namespace, namespace)  # noqa: S102 - 本地单用户学习工具
        except GradingTimeout:
            return 0.0, [{"error": f"代码执行超时（>{timeout_sec}s），疑似死循环，所有用例未通过"}]
        except SyntaxError as e:
            # 语法错误是笔误不是概念信号：带行号 + 源码行，供 web 端
            # 「修正后重新提交」（grade 纯判分不落库，重交零成本）
            return 0.0, [{
                "error": f"语法错误（第 {e.lineno} 行）: {e.msg}",
                "syntax_error": True,
                "line": (e.text or "").rstrip("\n"),
            }]
        except Exception as e:  # noqa: BLE001
            return 0.0, [{"error": f"定义阶段异常: {type(e).__name__}: {e}"}]

        func = namespace.get(question.func_name)
        if not callable(func):
            return 0.0, [{"error": f"未找到函数 {question.func_name}"}]

        details = []
        passed = 0
        for tc in question.tests:
            try:
                if alarm_on:
                    signal.alarm(timeout_sec)
                got = func(*tc.args, **tc.kwargs)
                if alarm_on:
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
        if alarm_on:
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
            option_explanations=(
                "x == 5 是比较运算（判断 x 是否等于 5），不是赋值。",
                "正确。单个等号 = 是赋值，把名字 x 绑定到值 5。",
                "Python 不需要（也不允许）像 C 那样在赋值时声明类型 int。",
                "x: = 5 不是合法的赋值语法，冒号后不能直接跟等号。",
            ),
        ),
        Question(
            problem_id="pv-l2-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.UNDERSTAND, qtype="choice", loadings={"K": 1.0, "S": 0.4},
            prompt="执行 x = 3; x = x + 1 后，x 的值是？",
            options=("报错，等式不成立", "3", "4", "1"), answer=2,
            explanation="赋值先计算右边表达式（3+1=4），再把名字 x 绑定到 4。",
            option_explanations=(
                "不会报错。= 是赋值不是数学等式，x 出现在等号两边完全合法。",
                "忽略了第二次赋值——x 先被绑到 3，随后又被重新绑定到 4。",
                "正确。赋值先算右边 3+1=4，再把名字 x 绑定到 4。",
                "1 与计算无关，x = x + 1 是把 x 加 1 而不是设成 1。",
            ),
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
            reference="def swap_values(a, b):\n    return (b, a)",
        ),
        Question(
            problem_id="pv-l4-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="代码：a = [1, 2]; b = a[:]; b.append(3)。执行后 a 的值是？",
            options=("[1, 2, 3]", "[1, 2]", "报错", "不确定"), answer=1,
            explanation="a[:] 创建了新列表（浅拷贝），b 与 a 不再指向同一对象。",
            option_explanations=(
                "混淆了切片复制与直接赋值——a[:] 生成的是新列表，b 的 append 不会影响 a。",
                "正确。a[:] 是浅拷贝，b 与 a 指向不同对象，b.append(3) 只改 b。",
                "切片和 append 都是合法操作，不会报错。",
                "行为是确定的：a 始终是 [1, 2]。",
            ),
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
            option_explanations=(
                "多算了 5——range 是右开的，stop=5 不包含 5。",
                "正确。start=1、step=2，到小于 stop 为止：1、3。",
                "忽略了步长 2，也忽略了右开。",
                "把起点当成了 0，且漏掉了 1。",
            ),
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
            reference="def sum_to(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total",
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
            reference="def max_of(nums):\n    biggest = nums[0]\n    for n in nums:\n        if n > biggest:\n            biggest = n\n    return biggest",
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
            option_explanations=(
                "条件 i < 5 本身没问题，问题出在循环体里没有改变 i。",
                "正确。i 永远是 0，条件 i < 5 永远为真，循环无法退出。",
                "print 完全可以放在 while 循环里。",
                "初始值不是关键，缺少对 i 的更新才是死循环的根因。",
            ),
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
            option_explanations=(
                "正确。def 定义函数，return 把结果返回给调用方。",
                "function f(x) { ... } 是 JavaScript 风格，Python 用 def + 冒号。",
                "-> 是类型注解语法，不能跟 return 组合。",
                "Python 没有 define 关键字，定义函数一律用 def。",
            ),
        ),
        Question(
            problem_id="pf-l2-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.UNDERSTAND, qtype="choice", loadings={"K": 1.0, "S": 0.4},
            prompt="def f(x): print(x * 2)，则 y = f(3) 之后 y 的值是？",
            options=("6", "None", "报错", "3"), answer=1,
            explanation="print 只是输出副作用，没有 return 的函数返回 None。",
            option_explanations=(
                "6 被 print 显示到屏幕上，但函数没有 return，它的返回值是 None，不是 6。",
                "正确。函数没有 return 时默认返回 None。",
                "print(x * 2) 是合法调用，不会报错。",
                "3 是参数，与返回值无关。",
            ),
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
            reference="def is_even(n):\n    return n % 2 == 0",
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
            reference="def count_vowels(s):\n    count = 0\n    for ch in s.lower():\n        if ch in \"aeiou\":\n            count += 1\n    return count",
        ),
        Question(
            problem_id="pf-l4-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="def f(lst): lst.append(4)。执行 a = [1]; f(a) 后 a 的值是？",
            options=("[1]", "[1, 4]", "None", "报错"), answer=1,
            explanation="列表作为参数传的是引用，函数内 append 影响调用方的列表。",
            option_explanations=(
                "误以为传参是复制。列表作为参数传的是引用，函数内修改会反映到原列表。",
                "正确。f(a) 把列表引用传进去，lst.append(4) 直接改了这个列表，所以 a 变成 [1, 4]。",
                "None 是 f 的返回值，不是 a 的值。",
                "把列表传给函数完全合法，不会报错。",
            ),
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
            option_explanations=(
                "递归靠函数调用自身推进，不依赖 for 循环。",
                "正确。递归函数必须调用自身，并且有能终止的基准情形（base case）。",
                "print 与递归的定义无关，return 才是让值传回去的关键。",
                "递归同样可以用局部变量，它们不是递归的必要要素。",
            ),
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
            option_explanations=(
                "递归通常更慢（每层调用有函数调用开销），不是更快。",
                "正确。递归把问题化为更小的同类子问题（化归）；循环是重复执行同一段代码。",
                "恰恰相反，递归必须有终止条件（基准情形）。",
                "有本质区别：一个是化归，一个是重复。",
            ),
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
            reference="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
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
            option_explanations=(
                "没有基准情形，f 永远不会返回 0 或其他值。",
                "不会返回 None——它根本没机会返回，直接超出递归深度。",
                "正确。f 无限调用 f(n-1)，直到超过默认递归深度上限（约 1000），抛 RecursionError。",
                "不会正常结束，必然报错。",
            ),
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
            option_explanations=(
                "函数内的 x = 5 是新建的局部变量，并不会改写全局 x。",
                "正确。f 里的 x = 5 只在函数内生效，外面的 print(x) 读到的是全局 x = 10。",
                "函数内给局部变量赋值是完全合法的，不会报错。",
                "print(x) 输出的是 x 的值 10，不是 None。",
            ),
        ),
        Question(
            problem_id="ps-l3-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.APPLY, qtype="code", loadings={"P": 1.2},
            prompt="定义函数 make_counter()：每次调用返回的函数计数加一并返回新计数值（利用闭包，从 1 开始计）。",
            func_name="make_counter",
            explanation="闭包内用 nonlocal 修改外层计数变量。",
            reference="def make_counter():\n    count = 0\n    def counter():\n        nonlocal count\n        count += 1\n        return count\n    return counter",
        ),
        Question(
            problem_id="ps-l4-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="funcs = [lambda: i for i in range(3)]，则 [f() for f in funcs] 的结果是？",
            options=("[0, 1, 2]", "[2, 2, 2]", "[0, 0, 0]", "报错"), answer=1,
            explanation="闭包捕获的是变量 i 本身而非当时的值；循环结束后 i = 2。（注：lambda 参数默认值 i=i 可修复）",
            option_explanations=(
                "若 lambda 捕获的是当时的值才会得到 [0, 1, 2]；实际捕获的是变量 i 本身。",
                "正确。所有 lambda 共享同一个 i，循环结束后 i = 2，所以每个都返回 2。（修复：lambda i=i: i）",
                "不是 [0, 0, 0]——三个 lambda 读的是循环结束后的同一个 i，值是 2。",
                "lambda 捕获循环变量不会报错，只是行为与直觉不同。",
            ),
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
            option_explanations=(
                "正确。赋值分两步：先算右边 3 + 1 = 4，再把名字 x 绑定到 4。",
                "Python 没有「复制一份再修改」的赋值语义，名字直接绑定到计算结果。",
                "合法。= 不是数学等号，x 同时出现在两边没有问题。",
                "= 是赋值，== 才是比较；赋值语句不会先比较相等。",
            ),
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
            reference="def repeat_word(s):\n    return s * 2",
        ),
        Question(
            problem_id="pv-l4-02", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="a = [1, 2]\nb = a\nb = b + [3]\n执行后 a 的值是？",
            options=("[1, 2, 3]", "[1, 2]", "报错", "[3]"), answer=1,
            explanation="b + [3] 创建了新的列表，b 重新绑定到新列表；a 仍指向原列表，不受影响。",
            option_explanations=(
                "b + [3] 生成的是新列表，b 重新绑定到它；a 仍然指向原来的 [1, 2]，不会被改。",
                "正确。b = b + [3] 相当于先算新列表再重新绑定 b，原列表（a 所指）不变。",
                "列表 + 列表是合法操作，不会报错。",
                "[3] 不是结果，b 重新绑定到了 [1, 2, 3]。",
            ),
        ),
        Question(
            problem_id="pv-l4-03", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="a = [1, 2]\nb = a\nb += [3]\n执行后 a 的值是？",
            options=("[1, 2, 3]", "[1, 2]", "报错", "None"), answer=0,
            explanation="列表的 += 是原地修改（等价于 extend），b 与 a 共享同一对象，所以 a 也变成 [1, 2, 3]。",
            option_explanations=(
                "正确。列表的 += 是原地修改（等价于 extend），b 与 a 指向同一个列表，a 也变成 [1, 2, 3]。",
                "把 += 和 + 搞混了：b + [3] 生成新列表，而 b += [3] 是原地修改共享对象。",
                "列表的 += 完全合法，不会报错。",
                "b += [3] 后 b 是 [1, 2, 3]，不是 None。",
            ),
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
            reference="def count_even(nums):\n    count = 0\n    for n in nums:\n        if n % 2 == 0:\n            count += 1\n    return count",
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
            reference="def sum_range(a, b):\n    total = 0\n    for i in range(a, b + 1):\n        total += i\n    return total",
        ),
        Question(
            problem_id="pl-l4-02", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.0, "K": 0.4},
            prompt="执行 for i in range(0, 10, 3): print(i)，依次输出哪些数？",
            options=("0, 3, 6, 9", "0, 3, 6, 9, 12", "1, 4, 7, 10", "3, 6, 9"), answer=0,
            explanation="range(0, 10, 3) 从 0 开始、步长 3、不含 10，所以是 0, 3, 6, 9。",
            option_explanations=(
                "正确。start=0、step=3、stop=10 右开不含 10：0, 3, 6, 9。",
                "12 已经超过 stop=10，不在范围内。",
                "起点是 0 不是 1，且 stop 不含 10。",
                "漏掉了起点 0。",
            ),
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
            reference="def first_last(nums):\n    return (nums[0], nums[-1])",
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
            reference="def sum_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total",
        ),
        Question(
            problem_id="pf-l4-02", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="def f(x, lst=[]):\n    lst.append(x)\n    return lst\n连续调用 f(1)、f(2)、f(3) 后，f(3) 的返回值是？",
            options=("[3]", "[1, 2, 3]", "[1]", "报错"), answer=1,
            explanation="默认参数列表只在函数定义时创建一次，多次调用共享同一个列表对象，元素会跨调用累积。",
            option_explanations=(
                "若每次调用都新建默认列表才会返回 [3]；实际默认列表只在定义时创建一次，被所有调用共享。",
                "正确。lst=[] 只创建一次，f(1) 后是 [1]，f(2) 后是 [1, 2]，f(3) 返回 [1, 2, 3]。",
                "与 [1] 无关，元素是跨调用累积的。",
                "这是 Python 的经典陷阱，但不会报错。",
            ),
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
            reference="def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)",
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
            option_explanations=(
                "有基准情形 n <= 1 时返回 1，递归会终止。",
                "正确。函数体只有 f(n-1)，没有基准情形，任何输入都会无限递归直到栈溢出。",
                "有基准情形 n == 0 时返回 0，会终止。",
                "有基准情形 n == 0 时返回 n，会终止。",
            ),
        ),
        Question(
            problem_id="pr-l4-02", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="def g(n):\n    return 0 if n == 0 else 1 + g(n - 1)\n调用 g(3) 时，调用栈最深时有多少层？",
            options=("3 层", "4 层", "1 层", "无限"), answer=1,
            explanation="g(3)→g(2)→g(1)→g(0) 共 4 层，到 g(0) 返回 0 后再逐层展开。",
            option_explanations=(
                "只数到 g(2) 是 3 层，但还会继续压入 g(1)、g(0)，最深时是 4 层。",
                "正确。调用链 g(3) → g(2) → g(1) → g(0)，g(0) 是基准，栈最深时有 4 层。",
                "低估了深度——每次调用都压栈，不止 1 层。",
                "g(0) 是基准情形，递归会终止，不会无限。",
            ),
        ),
        Question(
            problem_id="pr-l4-03", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.0, "K": 0.3},
            prompt="下列哪个问题用递归解决最自然？",
            options=("遍历一棵树的所有节点", "打印 1 到 100", "计算两个整数之和", "交换两个变量的值"), answer=0,
            explanation="树的结构天然是递归的（每个子树都是一棵树），递归遍历最自然；其余问题迭代即可。",
            option_explanations=(
                "正确。树的每个子树都是一棵树，递归遍历与结构天然对应；其他三个问题迭代（循环）更直接。",
                "打印 1 到 100 用循环即可，递归没有优势还多一层栈开销。",
                "两个数相加直接算，不需要递归。",
                "交换两个变量一行多重赋值即可。",
            ),
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
            reference="count = 0\ndef step():\n    global count\n    count += 1\n    return count",
        ),
        Question(
            problem_id="ps-l3-03", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.APPLY, qtype="choice", loadings={"K": 1.0, "S": 0.4},
            prompt="x = 5\ndef f():\n    global x\n    x = x + 1\nf()\nprint(x) 输出什么？",
            options=("5", "6", "报错", "None"), answer=1,
            explanation="global 声明让函数内修改的是全局 x，所以 x 变为 6。",
            option_explanations=(
                "忽略了函数开头的 global x——它让函数内改的就是全局变量。",
                "正确。global x 声明后，x = x + 1 修改的是全局 x，从 5 变 6。",
                "有 global 声明，不会报 UnboundLocalError。",
                "print(x) 输出的是全局 x 的值 6，不是 None。",
            ),
        ),
        Question(
            problem_id="ps-l4-02", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2},
            prompt="def outer():\n    n = 0\n    def inner():\n        nonlocal n\n        n += 1\n        return n\n    return inner\nf = outer()\n连续调用 f() 三次，第三次的返回值是？",
            options=("1", "2", "3", "报错"), answer=2,
            explanation="nonlocal 让 inner 修改外层 outer 的 n，且跨调用保持，三次调用分别返回 1、2、3。",
            option_explanations=(
                "1 是第一次调用的返回值；第三次调用时 n 已经累加到 3。",
                "2 是第二次调用的返回值。",
                "正确。nonlocal 让 inner 修改并保持外层 outer 的 n，三次调用依次返回 1、2、3。",
                "nonlocal 声明合法，不会报错。",
            ),
        ),
        Question(
            problem_id="ps-l4-03", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.ANALYZE, qtype="choice", loadings={"S": 1.2, "K": 0.3},
            prompt="x = 10\ndef f():\n    print(x)\n    x = 5\nf() 会发生什么？",
            options=("输出 10", "输出 5", "UnboundLocalError", "NameError"), answer=2,
            explanation="函数内对 x 赋值使 x 成为局部变量，print(x) 时局部 x 尚未赋值，触发 UnboundLocalError。",
            option_explanations=(
                "函数体内有 x = 5，x 就被视为局部变量；print(x) 在它赋值之前读取，读不到 10。",
                "print(x) 在 x = 5 之前执行，而且它读取的是未绑定的局部 x，不会输出 5。",
                "正确。函数内对 x 赋值使其成为局部变量，print 时局部 x 尚未赋值 → UnboundLocalError。",
                "是 UnboundLocalError（局部变量未绑定），不是全局名字未定义的 NameError。",
            ),
        ),
        # ─── L5/L6 补齐（2026-08-27：打通 Bloom 六层全链）────────────
        # L5 评价 = choice（判断写法/取舍的优劣，S 载荷）；L6 创造 = code（P 载荷）。
        # 追加在末尾以保持原有题目顺序（CLI 默认取 selected[:n]）。
        Question(
            problem_id="pv-l5-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.EVALUATE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="要交换两个变量的值（a、b 均为数字），下列哪种写法最不推荐（有隐蔽问题）？",
            options=(
                "a, b = b, a",
                "tmp = a; a = b; b = tmp",
                "a = a + b; b = a - b; a = a - b（用加减法交换）",
                "a, b = (b, a)",
            ), answer=2,
            explanation="加减法交换只在数值且无精度问题时成立：浮点数会引入精度误差、数值大时可能溢出，可读性也差。Python 原生多重赋值 a, b = b, a 既简洁又无副作用。",
            option_explanations=(
                "正确且推荐：Python 原生多重赋值，简洁、无临时变量、无副作用。",
                "用临时变量三步交换，清晰可靠，没有任何隐蔽问题。",
                "正确（要选的就是它）。加减法交换只在数值且无精度问题时成立：浮点数有精度误差、大整数可能溢出，可读性也差。",
                "a, b = (b, a) 与 a, b = b, a 等价，同样正确。",
            ),
        ),
        Question(
            problem_id="pl-l5-01", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.EVALUATE, qtype="choice", loadings={"S": 1.2, "K": 0.3},
            prompt="已知要精确执行 N 次循环，选 for i in range(N) 而不是 while 的最主要原因是？",
            options=(
                "for 的执行速度更快",
                "for 自带计数变量的初始化与更新，少了一个手动维护出错的点（如忘更新导致死循环）",
                "while 不能用于数字循环",
                "for 只能搭配 range 使用",
            ), answer=1,
            explanation="已知次数时 for 更安全：迭代变量的初始化与步进由循环结构管理，避开'忘更新计数变量→死循环'这类错误。while 适合次数未知、靠条件退出（如读输入直到 EOF）的场景。",
            option_explanations=(
                "不是速度问题；真正的差别在安全性：for 少一个手动维护计数变量的出错点。",
                "正确。for 自动管理计数变量的初始化与步进，避免了忘更新变量导致的死循环。",
                "while 完全可以做数字循环（while i < N），只是要自己维护 i。",
                "for 可以迭代任何可迭代对象（列表、字符串、字典……），不只是 range。",
            ),
        ),
        Question(
            problem_id="pf-l5-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.EVALUATE, qtype="choice", loadings={"S": 1.1, "K": 0.5},
            prompt="定义一个返回计算结果的函数 calc(n) 时，为什么应该用 return 而不是在函数里 print 结果？",
            options=(
                "函数里不能同时使用 print 和 return",
                "return 的代码执行得更快",
                "return 把结果交给调用方继续使用（赋值、传参），print 只是输出副作用，结果无法复用",
                "return 只能返回整数",
            ), answer=2,
            explanation="return 让函数成为可组合的'计算单元'：结果可赋值、可传参、可测试；print 只是把值显示到屏幕，把 print 当返回值用的函数无法被调用方继续计算。",
            option_explanations=(
                "函数里可以同时用 print 和 return，两者职责不同。",
                "与执行速度无关；差别在结果是否可复用。",
                "正确。return 把结果交给调用方（可赋值、传参、测试）；print 只是显示副作用，结果无法复用。",
                "return 可以返回任何对象：数字、字符串、列表、函数……不限整数。",
            ),
        ),
        Question(
            problem_id="pr-l5-01", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.EVALUATE, qtype="choice", loadings={"S": 1.2, "K": 0.3},
            prompt="关于递归与循环的选择，下列哪个说法正确？",
            options=(
                "递归总是比循环更高效，因为代码更短",
                "递归用调用栈自然表达'自己调用自己'，适合树/嵌套结构，但有栈开销与深度限制",
                "递归不需要基准情形，靠解释器兜底",
                "递归能解决的问题循环一定不能解决",
            ), answer=1,
            explanation="递归贴近问题结构（如遍历树），但每层调用占栈内存，递归过深会 RecursionError；深递归通常改写为显式栈或循环。",
            option_explanations=(
                "递归每层调用都有栈开销，通常更慢，不是更高效。",
                "正确。递归用调用栈自然表达「自己调用自己」，适合树/嵌套结构，但有栈开销和深度限制。",
                "递归必须自己写基准情形，不能靠解释器兜底。",
                "任何递归都可以改写成循环（显式栈），所以不是递归独有、循环不能做的。",
            ),
        ),
        Question(
            problem_id="ps-l5-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.EVALUATE, qtype="choice", loadings={"S": 1.2, "K": 0.4},
            prompt="关于函数与全局变量，Python 社区推荐的最佳实践是？",
            options=(
                "在函数里尽量多用 global 直接修改全局变量，省去传参",
                "把所有变量都定义成全局变量，方便任何函数访问",
                "函数内永远不能读取任何全局变量",
                "需要共享/修改的状态尽量作为参数传入，或用类/闭包封装，避免滥用 global",
            ), answer=3,
            explanation="滥用 global 会让函数依赖外部可变状态、顺序敏感、难以测试与复用。读取全局常量没有问题；需要修改共享状态时，优先用参数/类/闭包封装。",
            option_explanations=(
                "滥用 global 是坏实践：函数依赖外部可变状态、调用顺序敏感、难测试难复用。",
                "把所有变量都设为全局会让程序难以追踪与维护，不是推荐做法。",
                "函数内读取全局常量完全没问题（如常量、配置），只有修改共享状态才需要谨慎。",
                "正确。需要共享/修改的状态尽量作为参数传入，或用类/闭包封装，避免滥用 global。",
            ),
        ),
        Question(
            problem_id="pv-l6-01", skill_id="python.variables", topic="python.variables",
            bloom_level=BloomLevel.CREATE, qtype="code", loadings={"P": 1.2, "S": 0.4},
            prompt="定义函数 dedupe(items)，返回去重后的新列表，保持元素第一次出现的顺序（如 dedupe([3, 1, 3, 2, 1]) 返回 [3, 1, 2]）。",
            func_name="dedupe",
            tests=(
                TestCase(args=([3, 1, 3, 2, 1],), expected=[3, 1, 2]),
                TestCase(args=([],), expected=[]),
                TestCase(args=(["a", "b", "a", "c"],), expected=["a", "b", "c"]),
            ),
            explanation="用集合 seen 记录已出现元素，遍历时第一次遇到才加入结果——既去重又保持原顺序。",
            reference="def dedupe(items):\n    seen = set()\n    result = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
        ),
        Question(
            problem_id="pl-l6-01", skill_id="python.loops", topic="python.loops",
            bloom_level=BloomLevel.CREATE, qtype="code", loadings={"P": 1.2, "S": 0.3},
            prompt="定义函数 make_star_triangle(n)，返回 n 行由 * 组成的直角三角形（第 i 行 i 个 *），行间用换行分隔。如 make_star_triangle(3) 返回 '*\\n**\\n***'。",
            func_name="make_star_triangle",
            tests=(
                TestCase(args=(3,), expected="*\n**\n***"),
                TestCase(args=(1,), expected="*"),
            ),
            explanation="每行 '*'.repeat(i)（或 '*' * i），再用换行拼接各行。",
            reference="def make_star_triangle(n):\n    return \"\\n\".join(\"*\" * i for i in range(1, n + 1))",
        ),
        Question(
            problem_id="pf-l6-01", skill_id="python.functions", topic="python.functions",
            bloom_level=BloomLevel.CREATE, qtype="code", loadings={"P": 1.1, "S": 0.5},
            prompt="定义函数 apply_twice(f, x)，返回 f(f(x))——把 x 传给 f，再把结果传给 f 一次。如 apply_twice(lambda n: n + 1, 5) 返回 7。",
            func_name="apply_twice",
            tests=(
                TestCase(args=(lambda n: n + 1, 5), expected=7),
                TestCase(args=(lambda s: s.upper(), "ab"), expected="AB"),
                TestCase(args=(lambda n: n * n, 3), expected=81),
            ),
            explanation="函数是一等公民，可作为参数传递：return f(f(x)) 先调一次再调一次。",
            reference="def apply_twice(f, x):\n    return f(f(x))",
        ),
        Question(
            problem_id="pr-l6-01", skill_id="python.recursion", topic="python.recursion",
            bloom_level=BloomLevel.CREATE, qtype="code", loadings={"P": 1.2},
            prompt="用递归定义函数 reverse_str(s)，返回字符串反转（如 reverse_str('abc') 返回 'cba'）。",
            func_name="reverse_str",
            tests=(
                TestCase(args=("abc",), expected="cba"),
                TestCase(args=("",), expected=""),
                TestCase(args=("a",), expected="a"),
                TestCase(args=("hello",), expected="olleh"),
            ),
            explanation="基准情形 len(s) <= 1 直接返回 s；否则把首字符移到尾部：reverse_str(s[1:]) + s[0]。",
            reference="def reverse_str(s):\n    if len(s) <= 1:\n        return s\n    return reverse_str(s[1:]) + s[0]",
        ),
        Question(
            problem_id="ps-l6-01", skill_id="python.scope", topic="python.scope",
            bloom_level=BloomLevel.CREATE, qtype="code", loadings={"P": 1.1, "S": 0.5},
            prompt="定义函数 make_adder(n)，返回一个把参数加上 n 的函数（闭包）。如 add5 = make_adder(5); add5(3) 返回 8。",
            func_name="make_adder",
            explanation="闭包捕获外层参数 n，返回的函数在调用时把 n 加到自己参数上（n 只读不改，无需 nonlocal）。",
            reference="def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder",
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


# make_adder 是闭包工厂题，返回值是函数，测试用例结构特殊，单独处理
def _grade_make_adder(user_code: str) -> tuple[float, list[dict]]:
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(user_code, ns, ns)  # noqa: S102
        make = ns.get("make_adder")
        add5 = make(5)
        add10 = make(10)
        results = [add5(3), add5(-1), add10(7)]
        ok = results == [8, 4, 17]
        return (1.0 if ok else 0.0), [{"expected": [8, 4, 17], "got": results, "passed": ok}]
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
        if question.problem_id == "ps-l6-01":
            return _grade_make_adder(user_answer)
        return grade(question, user_answer, timeout_sec=timeout_sec)
