"""独立评分器——与面试官分离，切断自证回路.

去偏设计（GOVERNANCE 规则2 的方法论落地）：
- 评分器与面试官是两次独立 LLM 调用
- 评分器只拿"用户侧视角"的 transcript：保留用户作答 + 用户可见的提问/执行结果块，
  **剔除 anchor 意图与追问策略**——评分器无法反推面试协议
- 评分器看不到题库答案与 ground truth
- P 维度只认【代码执行结果】块里的客观执行结果，口头"我会写"不算 P 证据（PRD 8b）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from cogmirror.belief_state import BloomLevel

from .dialogue import AnchorTurn, ExecResult, extract_json_object
from .graph import DimensionId, Graph
from .llm import LLMClient

DIM_ORDER = ["K", "P", "S", "C", "X"]
BLOOM_ORDER = ["REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE"]

# 评分器结构化输出约束（bloom 只覆盖题库实际有的 L1-L4）
SCORER_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "five_d": {
            "type": "object",
            "properties": {d: {"type": "number", "minimum": 0, "maximum": 1}
                           for d in DIM_ORDER},
            "required": DIM_ORDER,
        },
        "bloom": {
            "type": "object",
            "properties": {b: {"type": "number", "minimum": 0, "maximum": 1}
                           for b in BLOOM_ORDER},
            "required": BLOOM_ORDER,
        },
        "solo": {
            "type": "object",
            "description": "topic 短名 -> SOLO 层级估计（1-5，允许小数）",
            "additionalProperties": {"type": "number", "minimum": 1, "maximum": 5},
        },
        "overall": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence_notes": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "insufficient": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["five_d", "bloom", "overall", "insufficient"],
}

GRADER_ROLE: str = (
    "你是独立的编程认知评估员。你只根据一份「对话记录」判断一个成年自学者的"
    "认知状态，为 5D 维度（K 知识 / P 程序技能 / S 策略 / C 概念联结 / X 元认知）、"
    "Bloom 层级（L1 记忆-L4 分析）、SOLO 层级给出 0-1 的估计。\n"
    "你不是面试官，没见过面试协议、题库与标准答案，只能依据对话记录本身。\n"
    "纪律：\n"
    "1. 证据分层：区分「学习者明确说出的」与「你推断的」，只把有依据的写进估计。\n"
    "2. P 维度：只认【代码执行结果】块里的客观执行结果；口头说『我会写』不算 P 证据。"
    "没有代码执行结果时，把 P 标进 insufficient。\n"
    "3. 诚实：证据不足的维度写进 insufficient 列表，宁可缺失不猜，不要用 0.5 填数。\n"
    "4. C 判据（概念联结）：概念解释是否清晰、能否类比/迁移、能否判断概念间关系、"
    "能否自我修正。\n"
    "5. X 判据（元认知）：是否有『预测->行动->结果->比较->修正』闭环、自评是否校准、"
    "是否表现出犹豫/自我监控/修正行为。\n"
    "6. 对话中实际讨论过的每个 topic 都必须给出 solo 估计；某个 topic 证据不足"
    "判断不出来时，把该 topic 名写进 insufficient（如 \"solo: recursion\"），不要"
    "省略——省略会被当成『该 topic 没被对话覆盖』。\n"
    "7. 只输出一个 JSON 对象，直接输出、不要放进代码块，严格按下面模板（键名与嵌套"
    "结构必须一致，数值在 [0,1]，solo 在 [1,5]）：\n"
    "{\n"
    '  "five_d": {"K": 0.5, "P": 0.4, "S": 0.5, "C": 0.5, "X": 0.5},\n'
    '  "bloom": {"REMEMBER": 0.5, "UNDERSTAND": 0.5, "APPLY": 0.4, "ANALYZE": 0.5},\n'
    '  "solo": {"loops": 3.0},\n'
    '  "overall": 0.55,\n'
    '  "evidence_notes": {"依据": "说明"},\n'
    '  "insufficient": []\n'
    "}"
)


@dataclass
class ScorerInput:
    transcript: list[AnchorTurn]
    exec_results: list[ExecResult]
    rubric: str


@dataclass
class ScorerOutput:
    five_d: dict[DimensionId, float]
    bloom: dict[BloomLevel, float]
    solo: dict[str, float]
    overall: float
    evidence_notes: dict[str, str]
    insufficient: list[str]


def build_user_perspective_transcript(transcript: list[AnchorTurn]) -> str:
    """只保留用户可见内容：提问 / 作答 / 执行结果块，剔除 anchor 与追问策略."""
    lines: list[str] = ["# 对话记录（学习者视角）"]
    for turn in transcript:
        if turn.role == "assistant":
            lines.append(f"[提问] {turn.text}")
        elif turn.role == "user":
            lines.append(f"[学习者] {turn.text}")
        else:  # system：P 执行结果块，客观、用户可见
            lines.append(f"[执行结果] {turn.text}")
    return "\n\n".join(lines)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_scorer_output(raw: str) -> ScorerOutput:
    """容错解析：正常 JSON -> ```json``` 块 -> 文本 K=0.7 型兜底；任何失败不抛错."""
    data: dict | None = None
    source = "json"

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            data = parsed
    except (ValueError, TypeError):
        pass

    if data is None:
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict):
                    data = parsed
                    source = "json_block"
            except (ValueError, TypeError):
                pass

    if data is None:
        # MiniMax-M3 会在 content 里带 thinking 思维链前导——提取第一个平衡 JSON 对象
        obj = extract_json_object(raw)
        if obj:
            try:
                parsed = json.loads(obj)
                if isinstance(parsed, dict):
                    data = parsed
                    source = "json_extract"
            except (ValueError, TypeError):
                pass

    if data is None:
        kv: dict[str, float] = {}
        for m in re.finditer(r"\b([KPSCX])\s*[=:]\s*([0-9]*\.?[0-9]+)", raw):
            v = _to_float(m.group(2))
            if v is not None:
                kv[m.group(1)] = v
        if kv:
            data = {"five_d": kv}
            source = "text_kv"

    notes: dict[str, str] = {}
    if data is None:
        return ScorerOutput(
            five_d={}, bloom={}, solo={}, overall=0.0,
            evidence_notes={"parse_warning": "无法从评分器输出中解析出结构化结果"},
            insufficient=["parsing_failed"],
        )

    five_d_raw = data.get("five_d") if isinstance(data.get("five_d"), dict) else {}
    five_d: dict[DimensionId, float] = {}
    missing_dims: list[str] = []
    for d in DIM_ORDER:
        v = _to_float(five_d_raw.get(d))
        if v is None:
            missing_dims.append(d)
            v = 0.0
        five_d[DimensionId(d)] = _clamp01(v)

    bloom_raw = data.get("bloom") if isinstance(data.get("bloom"), dict) else {}
    bloom: dict[BloomLevel, float] = {}
    missing_bloom: list[str] = []
    for b in BLOOM_ORDER:
        v = _to_float(bloom_raw.get(b))
        if v is None:
            missing_bloom.append(b)
            v = 0.0
        bloom[BloomLevel[b]] = _clamp01(v)

    solo_raw = data.get("solo") if isinstance(data.get("solo"), dict) else {}
    solo: dict[str, float] = {}
    for k, v in solo_raw.items():
        fv = _to_float(v)
        if fv is not None:
            solo[str(k)] = max(1.0, min(5.0, fv))

    overall = _clamp01(_to_float(data.get("overall")) or 0.0)

    notes_raw = data.get("evidence_notes") if isinstance(data.get("evidence_notes"), dict) else {}
    evidence_notes: dict[str, str] = {str(k): str(v) for k, v in notes_raw.items()}

    insufficient_raw = data.get("insufficient")
    insufficient: list[str] = []
    if isinstance(insufficient_raw, list):
        insufficient = [str(x) for x in insufficient_raw]
    insufficient.extend(missing_dims)
    insufficient.extend(missing_bloom)
    if source != "json":
        evidence_notes.setdefault("parse_warning", f"评分器输出经 {source} 容错解析")
    if missing_dims or missing_bloom:
        evidence_notes.setdefault(
            "missing_fields",
            f"缺失字段已置 0 并计入 insufficient: {missing_dims + missing_bloom}",
        )

    return ScorerOutput(
        five_d=five_d, bloom=bloom, solo=solo, overall=overall,
        evidence_notes=evidence_notes, insufficient=insufficient,
    )


def build_scorer_system(graph: Graph) -> str:
    """稳定 system（可缓存）：评委角色 + 评卷判据 + C/X 判据."""
    return "\n\n".join([GRADER_ROLE, graph.rubric_text()])


def score_session(llm: LLMClient, graph: Graph,
                  transcript: list[AnchorTurn],
                  exec_results: list[ExecResult] | None = None) -> ScorerOutput:
    """对一段对话做独立评分."""
    user_text = build_user_perspective_transcript(transcript)
    raw = llm.complete(
        build_scorer_system(graph),
        user_text,
        cache_breakpoint=True,
        json_schema=SCORER_JSON_SCHEMA,
    )
    out = parse_scorer_output(raw)
    if exec_results:
        out.evidence_notes.setdefault(
            "exec_results",
            f"{len(exec_results)} 个代码执行结果已作为 P 维度客观证据提供",
        )
    # 确定性兜底（不依赖 LLM 自觉）：对话覆盖到的 topic 必须有 solo 估计，否则计 insufficient
    covered_topics = {t.anchor.split("-")[0] for t in transcript if t.anchor}
    missing_solo = sorted(covered_topics - set(out.solo))
    if missing_solo:
        out.insufficient.extend(f"solo:{t}" for t in missing_solo)
        out.evidence_notes.setdefault(
            "missing_solo",
            "对话覆盖但评分器未给 solo 估计的 topic 已计入 insufficient: "
            + ", ".join(missing_solo),
        )
    return out
