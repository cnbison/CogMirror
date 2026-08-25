"""技能图谱——对话锚定坐标系.

对话不是自由聊天，而是锚定在本图谱上的追问：每一轮由面试官根据一个
GraphNode（anchor）生成用户可见的提问，系统内部保留 anchor -> 图谱节点的
可追溯映射，不把判分逻辑交给 LLM 自由发挥（PRD 8a）。

坐标系复用 MVP 内容库：
- Bloom 层级复用 cogmirror.belief_state.BloomLevel（L1-L4）
- topic 复用 cogmirror.content.python_basics.PythonBasicsTopic
- 诊断信号复用 threshold_concepts（liminal/crossing）与 misconceptions

5D 语义采用 PRD 8b 的 L1 潜变量基准（C=概念联结、X=元认知），
与 cogmirror.belief_state.DimensionId（C=置信度/X=外部支架）不同，
见 GOVERNANCE 规则7 与 PRD 8b——本 spike 就是为此语义采集数据的实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

from cogmirror.belief_state import BloomLevel
from cogmirror.content.misconceptions import PYTHON_BASICS_MISCONCEPTION_LIBRARY_STR
from cogmirror.content.python_basics import PythonBasicsBloomLibrary, PythonBasicsTopic
from cogmirror.content.threshold_concepts import PythonThresholdConceptLibrary


class SOLOLevel(IntEnum):
    """SOLO 分类法五层结构（代码里此前完全没有，从零设计）.

    SOLO（Structure of Observed Learning Outcomes）用于判断对具体概念的
    掌握结构复杂度，捕捉 Bloom 分不清的中间状态（PRD 8b）。
    """

    PRE_STRUCTURAL = 1    # 前结构：无回应 / 答非所问
    UNI = 2               # 单一结构：抓住单个相关点但不完整
    MULTI = 3             # 多元结构：列出多个相关点但未建立联系
    RELATIONAL = 4        # 关联结构：整合多个点并解释其关系
    EXTENDED_ABSTRACT = 5 # 抽象拓展：抽象归纳并迁移到新情境

    @property
    def label(self) -> str:
        return SOLO_LABELS[self]


SOLO_LABELS: dict[SOLOLevel, str] = {
    SOLOLevel.PRE_STRUCTURAL: "前结构（无回应或答非所问）",
    SOLOLevel.UNI: "单一结构（抓住单个相关点，不完整）",
    SOLOLevel.MULTI: "多元结构（列出多个相关点，未建立联系）",
    SOLOLevel.RELATIONAL: "关联结构（整合多个点并解释关系）",
    SOLOLevel.EXTENDED_ABSTRACT: "抽象拓展（抽象归纳并迁移到新情境）",
}


class DimensionId(str, Enum):
    """5D 维度（PRD 8b L1 潜变量语义，区别于 cogmirror 内的 L3 证据语义）.

    C=概念联结（图式/结构知识）、X=元认知（Flavell）——对话模式能采到的
    差异化信号，静态题库采不到（PRD 8b 可测性排序）。
    """

    K = "K"  # 知识：我知道什么（陈述性）
    P = "P"  # 程序技能：我能做什么（程序性，必须提交可执行代码锚点）
    S = "S"  # 策略：我应该怎么做（条件性）
    C = "C"  # 概念联结：这些知识之间如何连接（图式/结构知识）
    X = "X"  # 元认知：我如何知道自己是否知道（Flavell）

    @property
    def label(self) -> str:
        return DIM_LABELS[self]


DIM_LABELS: dict[DimensionId, str] = {
    DimensionId.K: "知识（陈述性）",
    DimensionId.P: "程序技能（产出物）",
    DimensionId.S: "策略（条件性）",
    DimensionId.C: "概念联结（图式/结构）",
    DimensionId.X: "元认知（自我监控）",
}


class ProbeKind(str, Enum):
    """单轮追问的探测类型."""

    EXPLAIN = "explain"                # K：用自己的话解释概念
    ANALOGY = "analogy"                # C：类比说明概念间关系
    TRANSFER = "transfer"              # S/C：把做法迁移到新场景
    RELATION_JUDGE = "relation_judge"  # C：判断概念/代码片段的关系
    PREDICT_VERIFY = "predict_verify"  # X：先预测再验证、比较差异
    REFLECT = "reflect"                # X：回顾过程、定位易错点
    REGULATE = "regulate"              # X：预期与实际不符时修正理解
    CODE = "code"                      # P：提交可执行代码（硬锚点）


# topic 短名 -> PythonBasicsTopic.value（node_id 用短名，便于人读）
TOPIC_SHORT_TO_ID: dict[str, str] = {
    "variables": PythonBasicsTopic.VARIABLES.value,
    "loops": PythonBasicsTopic.LOOPS.value,
    "functions": PythonBasicsTopic.FUNCTIONS.value,
    "recursion": PythonBasicsTopic.RECURSION.value,
    "scope": PythonBasicsTopic.SCOPE.value,
}

TOPIC_ID_TO_SHORT: dict[str, str] = {v: k for k, v in TOPIC_SHORT_TO_ID.items()}


@dataclass(frozen=True)
class GraphNode:
    """单个对话锚点节点.

    Attributes:
        node_id: 唯一标识，格式 {topic}-L{bloom}-S{solo}-{dim}
        topic: 知识点 ID（PythonBasicsTopic.value，如 "python.loops"）
        bloom_level: Bloom 认知层级
        solo_level: SOLO 结构层级
        dimension: 5D 维度（L1 潜变量语义）
        probe_kind: 探测类型
        question_seed: CODE 节点 = 题库 problem_id（如 "pr-l3-01"）；
                       其余 = 面试官据此生成用户可见提问的中文指令
    """

    node_id: str
    topic: str
    bloom_level: BloomLevel
    solo_level: SOLOLevel
    dimension: DimensionId
    probe_kind: ProbeKind
    question_seed: str


@dataclass(frozen=True)
class SoloProbe:
    """某 topic 在某 SOLO 层的探测模板与成功证据.

    probe_text 是面试官通用的探测角度；success_evidence 供评分器 rubric
    判断"什么样的回答算达到了该 SOLO 层"。
    """

    topic: str
    solo_level: SOLOLevel
    probe_text: str
    success_evidence: str


@dataclass(frozen=True)
class CXProbe:
    """C/X 专属探测模板（静态题库采不到的差异化信号，PRD 8b）."""

    dimension: DimensionId
    probe_kind: ProbeKind
    probe_text: str
    success_evidence: str


def _node(short_topic: str, bloom: BloomLevel, solo: int, dim: DimensionId,
          kind: ProbeKind, seed: str) -> GraphNode:
    b = int(bloom.value)
    return GraphNode(
        node_id=f"{short_topic}-L{b}-S{solo}-{dim.value}",
        topic=TOPIC_SHORT_TO_ID[short_topic],
        bloom_level=bloom,
        solo_level=SOLOLevel(solo),
        dimension=dim,
        probe_kind=kind,
        question_seed=seed,
    )


def _build_nodes() -> list[GraphNode]:
    """20 个节点 = 5 topic × L1-L4。

    维度分布：K=5、P=5、C=4、X=3、S=2，五个维度都至少一个节点；
    每个 topic 的 L3 都是 CODE 节点（P 维度硬锚点，PRD 8b）。
    """
    R, U, A, N = (BloomLevel.REMEMBER, BloomLevel.UNDERSTAND,
                  BloomLevel.APPLY, BloomLevel.ANALYZE)
    nodes = [
        # ── 变量与赋值 ──────────────────────────────────────────────
        _node("variables", R, 1, DimensionId.K, ProbeKind.EXPLAIN,
              "请用自己的话解释：在 Python 里变量是什么？x = 5 这个语句到底做了什么？"),
        _node("variables", U, 3, DimensionId.C, ProbeKind.RELATION_JUDGE,
              "x = x + 1 这个语句为什么不是矛盾？请解释 '=' 在这里和数学等号的区别。"),
        _node("variables", A, 3, DimensionId.P, ProbeKind.CODE, "pv-l3-01"),
        _node("variables", N, 4, DimensionId.X, ProbeKind.PREDICT_VERIFY,
              "请先预测：执行 a = [1, 2]; b = a; b.append(3) 之后 a 的值是什么？"
              "然后说明依据；如果方便可以运行验证，看看是否如你所料。"),
        # ── 循环 ────────────────────────────────────────────────────
        _node("loops", R, 1, DimensionId.K, ProbeKind.EXPLAIN,
              "请解释 range(5) 会产生哪些整数？为什么最后一个不是 5？"),
        _node("loops", U, 2, DimensionId.C, ProbeKind.EXPLAIN,
              "range(1, 5) 和 range(5) 有什么区别？请解释 range 的 start/stop/step 三段式，"
              "特别是 stop 为什么是不包含的。"),
        _node("loops", A, 3, DimensionId.P, ProbeKind.CODE, "pl-l3-01"),
        _node("loops", N, 4, DimensionId.S, ProbeKind.TRANSFER,
              "这个 while 循环为什么是死循环？\n    i = 0\n    while i < 5:\n        print(i)\n"
              "先诊断问题，再说明你自己写循环时会如何设置终止条件避免死循环。"),
        # ── 函数 ────────────────────────────────────────────────────
        _node("functions", R, 1, DimensionId.K, ProbeKind.EXPLAIN,
              "请解释 def 定义函数的语法：参数和 return 各起什么作用？"),
        _node("functions", U, 2, DimensionId.C, ProbeKind.ANALOGY,
              "return 和 print() 有什么区别？有说法是 'return 是递东西给调用者，print 是喊一嗓子'，"
              "你同意吗？请说说你的理解。"),
        _node("functions", A, 3, DimensionId.P, ProbeKind.CODE, "pf-l3-01"),
        _node("functions", N, 4, DimensionId.S, ProbeKind.RELATION_JUDGE,
              "def f(lst): lst.append(4)，执行 a = [1]; f(a) 后 a 变成什么？为什么？"
              "这属于哪种参数传递？"),
        # ── 递归 ────────────────────────────────────────────────────
        _node("recursion", R, 1, DimensionId.K, ProbeKind.EXPLAIN,
              "请解释递归的定义：为什么递归必须同时具备 '调用自身' 和 '基准情形' 两个要素？"),
        _node("recursion", U, 3, DimensionId.C, ProbeKind.ANALOGY,
              "递归和 for 循环的本质区别是什么？请用 '化归' 的思路说明："
              "把问题化成同类子问题直到落到基准情形。"),
        _node("recursion", A, 4, DimensionId.P, ProbeKind.CODE, "pr-l3-01"),
        _node("recursion", N, 4, DimensionId.X, ProbeKind.PREDICT_VERIFY,
              "def f(n): return f(n-1)，调用 f(3) 会发生什么？请先预测结果，再解释为什么。"),
        # ── 作用域 ──────────────────────────────────────────────────
        _node("scope", R, 1, DimensionId.K, ProbeKind.EXPLAIN,
              "请说出 Python 名字查找顺序 LEGB 是哪四个作用域，并举一个简单例子。"),
        _node("scope", U, 3, DimensionId.C, ProbeKind.RELATION_JUDGE,
              "x = 10\ndef f():\n    x = 5\nf()\nprint(x) 输出什么？为什么函数内的 x = 5 "
              "不影响外面的 x？global 关键字什么时候用？"),
        _node("scope", A, 3, DimensionId.P, ProbeKind.CODE, "ps-l3-01"),
        _node("scope", N, 5, DimensionId.X, ProbeKind.REGULATE,
              "funcs = [lambda: i for i in range(3)]，执行 [f() for f in funcs] 结果是什么？"
              "如果你一开始的预测和实际不符，说明哪里和预期不同、你会怎么修正自己的理解。"),
    ]
    return nodes


def _build_solo_probes() -> list[SoloProbe]:
    """每 topic × SOLO 层的探测模板与成功证据（图里实际用到的层）. """

    def p(topic: str, solo: int, probe_text: str, evidence: str) -> SoloProbe:
        return SoloProbe(
            topic=TOPIC_SHORT_TO_ID[topic], solo_level=SOLOLevel(solo),
            probe_text=probe_text, success_evidence=evidence,
        )

    return [
        # ── variables ────────────────────────────────────────────────
        p("variables", 1, "让学习者用自己的话复述变量的定义与赋值动作。",
          "能说清赋值是把名字绑定到对象，而非数学等式。"),
        p("variables", 3, "让学习者解释赋值与引用的多个侧面（自增、多重赋值、共享）。",
          "能同时正确解释自增与引用共享，且能说明各自边界。"),
        p("variables", 4, "让学习者判断引用/拷贝在不同代码片段下的关系。",
          "能整合引用、拷贝、可变对象规则并正确预测结果。"),
        # ── loops ───────────────────────────────────────────────────
        p("loops", 1, "让学习者复述 range 与 for/while 的基本语法。",
          "能说出 range(5) 产生 0..4，左闭右开。"),
        p("loops", 2, "让学习者解释 range 三段式的单一侧面。",
          "能解释 stop 不包含，但可能忽略 step。"),
        p("loops", 3, "让学习者给出循环的多个构成要素。",
          "能同时列出初始状态、终止条件、状态更新三要素。"),
        p("loops", 4, "让学习者诊断死循环并把结论迁移到自己的写法。",
          "能定位缺失的状态更新，并说出自写循环时的防护做法。"),
        # ── functions ────────────────────────────────────────────────
        p("functions", 1, "让学习者复述 def/参数/return 语法。",
          "能说出函数定义与 return 返回值的骨架。"),
        p("functions", 2, "让学习者解释 return 与副作用（print）的区别。",
          "能说清 print 是副作用、无 return 返回 None。"),
        p("functions", 3, "让学习者列举函数设计与调用的多个要点。",
          "能同时涉及参数、返回值、多函数协作。"),
        p("functions", 4, "让学习者分析参数传递与作用域的交互。",
          "能说清可变对象按引用共享、函数内修改影响调用方。"),
        # ── recursion ───────────────────────────────────────────────
        p("recursion", 1, "让学习者复述递归定义与两个必要要素。",
          "能说出调用自身 + 基准情形。"),
        p("recursion", 3, "让学习者解释递归的化归思想与循环对比。",
          "能把递归说成化归：化为同类子问题直到基准情形。"),
        p("recursion", 4, "让学习者分析调用栈与递归边界。",
          "能整合调用栈、深度限制、基准情形并判断递归适用性。"),
        # ── scope ───────────────────────────────────────────────────
        p("scope", 1, "让学习者复述 LEGB 查找规则。",
          "能说出四个作用域层级。"),
        p("scope", 3, "让学习者解释作用域遮蔽与 global/nonlocal 的多个侧面。",
          "能同时解释遮蔽、赋值即局部、global 的使用场景。"),
        p("scope", 5, "让学习者把作用域规则抽象到闭包陷阱并归纳修法。",
          "能把闭包捕获规则抽象成一般结论并给出通用修正（默认参数/工厂函数）。"),
    ]


def _build_cx_probes() -> list[CXProbe]:
    """C/X 专属探测模板（PRD 8b 差异化信号，静态题库采不到）. """

    def p(dim: DimensionId, kind: ProbeKind, probe_text: str, evidence: str) -> CXProbe:
        return CXProbe(dimension=dim, probe_kind=kind,
                       probe_text=probe_text, success_evidence=evidence)

    return [
        # ── C 概念联结 ───────────────────────────────────────────────
        p(DimensionId.C, ProbeKind.EXPLAIN,
          "让学习者解释概念本质，并指出关键边界（如 range 左闭右开、变量是标签）。",
          "概念解释清晰、能主动点出边界条件。"),
        p(DimensionId.C, ProbeKind.ANALOGY,
          "让学习者用类比说明概念间或与熟悉事物的关系。",
          "类比贴切，且能指出类比的边界（哪里像、哪里不像）。"),
        p(DimensionId.C, ProbeKind.RELATION_JUDGE,
          "让学习者判断概念/代码片段的关系并说明判据。",
          "能给出正确判断并说出依据（如引用共享、作用域遮蔽、闭包捕获）。"),
        p(DimensionId.C, ProbeKind.TRANSFER,
          "让学习者把已知做法迁移到新场景，说明要改哪里。",
          "能识别需要改动的最小点并给出正确迁移。"),
        # ── X 元认知 ─────────────────────────────────────────────────
        p(DimensionId.X, ProbeKind.PREDICT_VERIFY,
          "让学习者先预测结果，再（如可能）运行验证，比较差异。",
          "预测有明确依据；验证后能说明差异来源并修正。"),
        p(DimensionId.X, ProbeKind.REFLECT,
          "让学习者回顾自己的解题/学习过程，定位易错点。",
          "能定位自己的易错点并说出应对策略。"),
        p(DimensionId.X, ProbeKind.REGULATE,
          "当预期与实际不符时，让学习者说明如何修正自己的理解。",
          "能描述预期-实际差异并给出修正后的理解。"),
    ]


def _build_rubric(bloom_lib: PythonBasicsBloomLibrary,
                  tc_lib: PythonThresholdConceptLibrary) -> str:
    lines: list[str] = ["# 评卷判据（Rubric）"]

    lines.append("\n## SOLO 五层结构")
    for level in SOLOLevel:
        lines.append(f"- S{int(level)} {level.label}")

    lines.append("\n## Bloom 目标成功标准（topic × 层）")
    for entry in bloom_lib.all_entries():
        lines.append(f"- {entry.topic} L{int(entry.bloom_level.value)} "
                     f"[{entry.goal_id}]: {entry.success_criteria}")

    lines.append("\n## 临界概念诊断信号（TC：liminal 表现 / 跨越信号）")
    for tc in tc_lib.all_entries():
        lines.append(f"- {tc.name}（{tc.tc_id}）")
        lines.append(f"  - liminal_signals: {'；'.join(tc.liminal_signals)}")
        lines.append(f"  - crossing_indicators: {'；'.join(tc.crossing_indicators)}")

    lines.append("\n## 常见错误模式（Misconception）")
    lines.append(PYTHON_BASICS_MISCONCEPTION_LIBRARY_STR.rstrip())

    return "\n".join(lines)


class Graph:
    """技能图谱：节点 + 检索 + rubric."""

    def __init__(self, nodes: list[GraphNode],
                 solo_probes: list[SoloProbe],
                 cx_probes: list[CXProbe],
                 rubric: str) -> None:
        self._nodes: dict[str, GraphNode] = {n.node_id: n for n in nodes}
        assert len(self._nodes) == len(nodes), "GraphNode node_id 重复"
        self._solo_probes: dict[tuple[str, SOLOLevel], SoloProbe] = {
            (p.topic, p.solo_level): p for p in solo_probes
        }
        self._cx_probes: dict[tuple[DimensionId, ProbeKind], CXProbe] = {
            (p.dimension, p.probe_kind): p for p in cx_probes
        }
        self._rubric = rubric
        self._ordered = sorted(
            nodes, key=lambda n: (n.bloom_level.value, int(n.solo_level), n.dimension.value))
        self._by_topic: dict[str, list[GraphNode]] = {}
        self._by_dimension: dict[DimensionId, list[GraphNode]] = {}
        for n in nodes:
            self._by_topic.setdefault(n.topic, []).append(n)
            self._by_dimension.setdefault(n.dimension, []).append(n)

    def get(self, node_id: str) -> GraphNode:
        """按 node_id 取节点；非法 id 抛 KeyError（dialogue 靠它强制 anchor 合法）."""
        return self._nodes[node_id]

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes_for_topic(self, topic: str) -> list[GraphNode]:
        return list(self._by_topic.get(topic, []))

    def nodes_for_dimension(self, dimension: DimensionId) -> list[GraphNode]:
        return list(self._by_dimension.get(dimension, []))

    def all_nodes(self) -> list[GraphNode]:
        """全部节点，按 bloom -> solo -> dimension 排序（dialogue 推进顺序）."""
        return list(self._ordered)

    def solo_probe(self, topic: str, solo_level: SOLOLevel) -> SoloProbe | None:
        return self._solo_probes.get((topic, solo_level))

    def cx_probe(self, dimension: DimensionId, probe_kind: ProbeKind) -> CXProbe | None:
        return self._cx_probes.get((dimension, probe_kind))

    def rubric_text(self) -> str:
        return self._rubric


def build_graph(bloom_lib: PythonBasicsBloomLibrary | None = None,
                tc_lib: PythonThresholdConceptLibrary | None = None) -> Graph:
    """构建完整技能图谱（默认复用 MVP 内容库实例）."""
    bloom_lib = bloom_lib or PythonBasicsBloomLibrary()
    tc_lib = tc_lib or PythonThresholdConceptLibrary()
    return Graph(
        nodes=_build_nodes(),
        solo_probes=_build_solo_probes(),
        cx_probes=_build_cx_probes(),
        rubric=_build_rubric(bloom_lib, tc_lib),
    )
