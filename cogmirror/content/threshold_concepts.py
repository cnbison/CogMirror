"""Python 学科 Threshold Concept（临界概念）库.

迁移说明：ECOS `ecos/cta/content/threshold_concepts.py` 的 8 条条目全部是
初中数学内容（函数/负数/分数等），按 MIGRATION.md 第1类要求"去掉非Python
学科的条目"后为空。Python 侧的 TC 定义实际存于 ECOS `tc_detector.py` 的
DEFAULT_TC_LIBRARY（5 条）。本文件将两者合并：沿用 ECOS 的条目结构
（pre/liminal/post 三态 + liminal 信号），内容取自 DEFAULT_TC_LIBRARY
并按其语义补全三态描述。这是新写内容，不是已验证资产，标注为初版。

TC（Threshold Concept）特性：
  - 不可逆（一旦跨越难以退回）
  - 变革性（理解后学习者 worldview 发生质变）
  - 整合性（连接多个先前分离的概念）
  - 渐进性（有 liminal 中间态）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from ..belief_state import BloomLevel


class TCStatus(Enum):
    """TC 跨越状态."""

    PRE_LIMITINAL = "pre_liminal"   # 未接触
    LIMINAL = "liminal"             # 中间态（似懂非懂）
    POST_LIMITINAL = "post_liminal"  # 已跨越


@dataclass(frozen=True)
class ThresholdConceptEntry:
    """单条 Threshold Concept 条目.

    Attributes:
        tc_id: 唯一标识，如 "TC_python_variables"
        name: 中文名称
        description: 1-2 句描述该 TC 的核心内涵
        pre_conception: 跨越前的典型前概念
        liminal_signals: 处于 liminal 态时的典型表现
        post_conception: 跨越后的典型理解
        crossing_indicators: 客观可观测的跨越信号
        skill_ids: 关联的知识点 ID 列表
        bloom_layer: 跨越所需的 Bloom 层级
    """

    tc_id: str
    name: str
    description: str
    pre_conception: str
    liminal_signals: tuple[str, ...]
    post_conception: str
    crossing_indicators: tuple[str, ...]
    skill_ids: tuple[str, ...]
    bloom_layer: BloomLevel


class PythonThresholdConceptLibrary:
    """Python 基础 Threshold Concept 库（5 条，初版）。

    用法：
        library = PythonThresholdConceptLibrary()
        tc = library.get("TC_python_variables")
    """

    _entries: ClassVar[list[ThresholdConceptEntry]] = [
        ThresholdConceptEntry(
            tc_id="TC_python_variables",
            name="变量是标签而非盒子",
            description="从「变量是装值的盒子」的直觉，过渡到「变量是指向对象的标签」的引用语义。",
            pre_conception="变量是一个盒子，赋值就是把值装进盒子；x = x + 1 在数学上无解",
            liminal_signals=(
                "我知道 x = x + 1 能跑，但说不清为什么不是矛盾",
                "a 和 b 指向同一个列表，改 a 就是改 b？这不对吧",
                "赋值到底拷贝了什么？",
            ),
            post_conception="变量是绑定到对象的名字，赋值是让名字指向新对象；可变对象的多个名字共享同一个对象",
            crossing_indicators=(
                "能解释 x = x + 1 的「先算右边再绑定」语义",
                "能预测 a = [1,2]; b = a; b.append(3) 之后 a 的值",
                "能区分可变对象共享与不可变对象重绑定",
            ),
            skill_ids=(
                "python.variables.assignment",
                "python.variables.reference",
            ),
            bloom_layer=BloomLevel.ANALYZE,
        ),
        ThresholdConceptEntry(
            tc_id="TC_python_loops",
            name="循环是受控的重复",
            description="从「照着写 for/while」的模仿，过渡到「循环不变式 + 终止条件」的结构化理解。",
            pre_conception="for 循环就是把一段代码抄 n 遍；range(5) 是 0 到 5",
            liminal_signals=(
                "range(1, 5) 为什么不到 5？我知道开区间但常忘",
                "while 循环的条件我都写了，为什么还死循环",
                "我能看懂循环，但自己写时不知道初始值设多少",
            ),
            post_conception="循环由初始状态、终止条件、状态更新三要素构成；range 是左闭右开区间；漏掉状态更新必然死循环",
            crossing_indicators=(
                "能正确计算 range(0, 10, 2) 的输出序列",
                "能用循环独立完成累加/查找/求最大值",
                "能诊断死循环代码中缺失的状态更新",
            ),
            skill_ids=(
                "python.loops.for_range",
                "python.loops.while_condition",
            ),
            bloom_layer=BloomLevel.APPLY,
        ),
        ThresholdConceptEntry(
            tc_id="TC_python_functions",
            name="函数是输入-输出映射",
            description="从「函数是一段有名字的代码」的直觉，过渡到「函数是参数到返回值的映射，调用是表达式」。",
            pre_conception="函数就是给一段代码起个名字；print 出来的就是返回值",
            liminal_signals=(
                "print() 和 return 有什么区别？我总是混",
                "没有 return 的函数算什么？它返回了什么？",
                "为什么我的函数返回 None？",
            ),
            post_conception="函数通过 return 返回值（无 return 返回 None），print 只是副作用；调用函数的表达式可以参与运算",
            crossing_indicators=(
                "能说出 return 和 print 的区别",
                "能写出带参数和返回值的函数并在表达式中调用",
                "能解释 void 函数的合理使用场景",
            ),
            skill_ids=(
                "python.functions.return_value",
            ),
            bloom_layer=BloomLevel.UNDERSTAND,
        ),
        ThresholdConceptEntry(
            tc_id="TC_python_recursion",
            name="递归是化归",
            description="从「递归是另一种循环」的混淆，过渡到「递归是把问题化为同类子问题 + 基准情形」的化归思维。",
            pre_conception="递归就是函数里的循环；没有终止也无所谓",
            liminal_signals=(
                "递归和 for 循环有什么区别？说不上来",
                "我知道要有 base case，但不知道该放在哪、怎么写",
                "我能照抄阶乘的例子，换个题就不会了",
            ),
            post_conception="递归的核心是化归：把问题分解为更小的同类子问题，直到落到 base case；调用栈深度有限所以必须收敛",
            crossing_indicators=(
                "能独立实现阶乘/斐波那契的递归版本",
                "能指出递归代码中缺失的 base case",
                "能对比递归与迭代各自适合的场景",
            ),
            skill_ids=(
                "python.functions.recursion",
            ),
            bloom_layer=BloomLevel.ANALYZE,
        ),
        ThresholdConceptEntry(
            tc_id="TC_python_scope",
            name="作用域是名字的查找规则",
            description="从「变量就是全局的」的直觉，过渡到「LEGB 查找规则 + 赋值即局部」的作用域模型。",
            pre_conception="变量定义了到处都能用；函数里赋值会改外面的变量",
            liminal_signals=(
                "为什么函数里改不了外面的 x？我试过有时行有时不行",
                "global 关键字什么时候要加？",
                "UnboundLocalError 是什么？我明明在外面定义了",
            ),
            post_conception="名字按 LEGB 顺序查找；函数内的赋值默认创建局部变量；修改外层名字需要 global/nonlocal 声明",
            crossing_indicators=(
                "能解释函数内赋值不影响全局变量的原因",
                "能正确使用 global/nonlocal 完成跨作用域读写",
                "能识别闭包陷阱（循环中创建函数引用同一变量）",
            ),
            skill_ids=(
                "python.scope.global_local",
            ),
            bloom_layer=BloomLevel.ANALYZE,
        ),
    ]

    def __init__(self) -> None:
        self._by_id: dict[str, ThresholdConceptEntry] = {e.tc_id: e for e in self._entries}

    def get(self, tc_id: str) -> ThresholdConceptEntry | None:
        return self._by_id.get(tc_id)

    def all_entries(self) -> list[ThresholdConceptEntry]:
        return list(self._entries)

    def all_tc_ids(self) -> list[str]:
        return [e.tc_id for e in self._entries]
