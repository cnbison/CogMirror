"""锚定追问对话引擎.

流程（每轮）：
1. 从选定 topics 里选下一个未覆盖节点（topic 内按 bloom -> solo -> dimension 推进）
2. 对非 CODE 节点：调面试官 LLM 生成 {"anchor", "question"}；
   anchor 必须落在技能图谱上（graph.get），非法则重试一次，仍非法记录并跳过
   ——这是"锚定可追溯"的强制机制，不把追问交给 LLM 自由发挥（PRD 8a）
3. CODE 节点（P 维度硬锚点）：直接复用题库题面（确定性，不经过 LLM），
   用 QuestionBank.grade_answer 判分得到 ExecResult
4. P 的执行结果以 system 文本块逐字合入 transcript，但 **score 数值只进数据
   不进 prompt**——面试官下一轮看不到得分，防止用分数引导
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from cogmirror.questions import QuestionBank

from .graph import DimensionId, Graph, GraphNode, ProbeKind
from .llm import LLMClient

DEFAULT_MAX_ROUNDS = 30

# 面试官结构化输出约束（Sonnet 4.6 走 output_config.format）
INTERVIEWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "anchor": {
            "type": "string",
            "description": "本轮探测的图谱节点 node_id，必须原样返回系统给定的节点",
        },
        "question": {"type": "string", "description": "给学习者看的中文提问"},
    },
    "required": ["anchor", "question"],
}

INTERVIEWER_SYSTEM: str = (
    "你是一名 Python 编程认知诊断的面试官，正在进行一轮锚定在技能图谱上的"
    "追问式对话。\n"
    "任务：根据系统给定的「下一探测节点」，生成一句给学习者的中文提问。\n"
    "纪律：\n"
    "1. 只针对给定的节点探测，不要跳到其他知识点。\n"
    "2. 提问要自然、像人话，不要出现「5D 维度」「Bloom 层级」「SOLO」这类内部术语。\n"
    "3. 不要直接给出答案、代码或提示，让学习者自己回答。\n"
    "4. 只输出 JSON：{\"anchor\": 原样返回给定的节点 id, \"question\": 提问文本}，"
    "不要输出其他任何内容。"
)


@dataclass
class AnchorTurn:
    """对话轮次.

    role: "assistant"（面试官提问）/ "user"（学习者作答）/ "system"（客观执行结果块）
    anchor: 非 None 表示该轮来自哪个图谱节点；用户作答轮为 None
    """

    role: str
    text: str
    anchor: str | None = None


@dataclass
class ExecResult:
    """P 维度的客观执行锚点（score 只进数据，不进 prompt）."""

    node_id: str
    submitted_code: str
    score: float
    details: list[dict]
    executed: bool = True


@dataclass
class DialogueState:
    user_id: str
    transcript: list[AnchorTurn] = field(default_factory=list)
    exec_results: list[ExecResult] = field(default_factory=list)
    covered_nodes: set[str] = field(default_factory=set)
    skipped_anchors: list[str] = field(default_factory=list)


def _extract_exec_result_block(er: ExecResult, include_score: bool) -> str:
    """P 执行结果块文本（score 数值只进数据不进 prompt）."""
    lines = ["【代码执行结果】由系统自动判定，非面试官评价", ""]
    lines.append("你的提交：")
    lines.append("```python")
    lines.append(er.submitted_code)
    lines.append("```")
    lines.append("")
    if er.details:
        lines.append("运行结果（逐个测试用例）：")
        for i, d in enumerate(er.details, 1):
            if "error" in d:
                lines.append(f"- 用例 {i}: {d['error']}")
            else:
                status = "通过" if d.get("passed") else "未通过"
                lines.append(f"- 用例 {i}: {status}（期望 {d.get('expected')}，"
                             f"得到 {d.get('got')}）")
    else:
        lines.append("运行结果：通过全部测试用例。")
    if include_score:
        lines.append(f"得分: {er.score:.2f}")
    return "\n".join(lines)


def default_ask(node: GraphNode) -> str:
    """从 stdin 读用户输入：CODE 节点多行（END 结束），其余单行."""
    if node.probe_kind == ProbeKind.CODE:
        print("（输入代码，单独一行输入 END 结束）")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines)
    return input().strip()


def _parse_interviewer_response(raw: str) -> dict | None:
    """解析面试官输出 {"anchor", "question"}；解析失败返回 None（按非法 anchor 处理）."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        data = json.loads(m.group(1)) if m else None
    if not isinstance(data, dict):
        return None
    anchor = data.get("anchor")
    question = data.get("question")
    if not anchor or not question:
        return None
    return {"anchor": str(anchor), "question": str(question)}


class DialogueEngine:
    """锚定追问对话引擎.

    Args:
        llm: 面试官 LLM 客户端（FakeLLM 或 OpenAICompatClient）
        graph: 技能图谱（anchor 合法性的唯一来源）
        bank: 题库（CODE 节点题面 + 确定性判分）
        topics: 要诊断的 topic 列表（PythonBasicsTopic.value）
        max_rounds: 对话轮次上限（防止无界会话）
    """

    def __init__(self, llm: LLMClient, graph: Graph, bank: QuestionBank,
                 topics: list[str], max_rounds: int = DEFAULT_MAX_ROUNDS) -> None:
        self.llm = llm
        self.graph = graph
        self.bank = bank
        self.topics = topics
        self.max_rounds = max_rounds

    def run(self, user_id: str,
            ask: Callable[[GraphNode], str] | None = None) -> DialogueState:
        """跑一轮完整诊断对话."""
        ask = ask or default_ask
        state = DialogueState(user_id=user_id)

        for _ in range(self.max_rounds):
            node = self._pick_next(state)
            if node is None:
                break  # 全部覆盖

            if node.probe_kind == ProbeKind.CODE:
                question = self._bank_prompt(node)
                anchor = node.node_id  # 合法 by construction，无需 LLM
            else:
                picked = self._ask_interviewer(state, node)
                if picked is None:
                    # 两次非法 anchor：记录并跳过该节点，避免死循环
                    state.skipped_anchors.append(node.node_id)
                    state.covered_nodes.add(node.node_id)
                    continue
                anchor = picked["anchor"]
                question = picked["question"]

            self._emit_question(question)
            answer = ask(node)

            state.transcript.append(
                AnchorTurn(role="assistant", anchor=anchor, text=question))
            if node.probe_kind == ProbeKind.CODE:
                er = self._execute(node, answer)
                state.exec_results.append(er)
                # score 数值只进数据不进 prompt
                state.transcript.append(AnchorTurn(
                    role="system", anchor=None,
                    text=_extract_exec_result_block(er, include_score=False)))
            else:
                state.transcript.append(
                    AnchorTurn(role="user", anchor=None, text=answer))

            state.covered_nodes.add(node.node_id)

        return state

    # ── 内部 ──────────────────────────────────────────────────────────

    def _pick_next(self, state: DialogueState) -> GraphNode | None:
        for topic in self.topics:
            for node in self.graph.all_nodes():  # 已按 bloom->solo->dim 排序
                if node.topic == topic and node.node_id not in state.covered_nodes:
                    return node
        return None

    def _bank_prompt(self, node: GraphNode) -> str:
        q = self.bank.get(node.question_seed)
        if q is None:
            raise KeyError(f"CODE 节点 {node.node_id} 引用的题库题目 "
                           f"{node.question_seed} 不存在")
        return q.prompt

    def _emit_question(self, question: str) -> None:
        print(question)

    def _interview_payload(self, state: DialogueState, node: GraphNode) -> str:
        parts = [
            f"本轮要探测的节点 id：{node.node_id}",
            f"节点探测指令（据此生成提问）：{node.question_seed}",
            "",
            "已发生的对话（供你避免重复提问）：",
        ]
        for turn in state.transcript:
            if turn.role == "system":
                parts.append(f"[执行结果] {turn.text}")
            elif turn.role == "assistant":
                parts.append(f"[面试官] {turn.text}")
            else:
                parts.append(f"[学习者] {turn.text}")
        return "\n".join(parts)

    def _ask_interviewer(self, state: DialogueState, node: GraphNode) -> dict | None:
        """调面试官生成提问；anchor 非法重试一次，仍非法返回 None."""
        for _ in range(2):
            raw = self.llm.complete(
                INTERVIEWER_SYSTEM,
                self._interview_payload(state, node),
                cache_breakpoint=True,
                json_schema=INTERVIEWER_SCHEMA,
            )
            picked = _parse_interviewer_response(raw)
            if picked and self.graph.has(picked["anchor"]):
                return picked
        return None

    def _execute(self, node: GraphNode, code: str) -> ExecResult:
        q = self.bank.get(node.question_seed)
        if q is None:
            raise KeyError(f"CODE 节点 {node.node_id} 引用的题库题目 "
                           f"{node.question_seed} 不存在")
        score, details = self.bank.grade_answer(q, code)
        return ExecResult(node_id=node.node_id, submitted_code=code,
                          score=score, details=details, executed=True)
