"""n=1 比对分析 + 预留的统计工具（5-8 人轮用）.

比对纪律（GOVERNANCE 规则2/5）：
- 只比对对话**实际覆盖**的 topic / Bloom 层；未覆盖的显式标注「对话未覆盖，不比对」，
  不能拿空数据当 0 分
- 逐 topic 的对话估计来自评分器输出的 SOLO 层级，经 (solo-1)/4 归一化为 0-1
  的结构性掌握代理指标——这是代理，不是直接测量，结论里会标注
- C/X/S 维度静态题库没有 ground truth（这正是 spike 要验证的），只做五维估计
  摘要与诚实标注，不做"与锚点一致性"的判定
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from cogmirror.belief_state import BloomLevel

from .graph import DimensionId, TOPIC_SHORT_TO_ID, TOPIC_ID_TO_SHORT
from .protocol import SessionRecord

DIM_ORDER = ["K", "P", "S", "C", "X"]
BLOOM_ORDER = ["REMEMBER", "UNDERSTAND", "APPLY", "ANALYZE"]

# topic 中文短名（与 cogmirror/cli.py 保持一致）
TOPIC_LABELS: dict[str, str] = {
    "python.variables": "变量赋值",
    "python.loops": "循环",
    "python.functions": "函数",
    "python.recursion": "递归",
    "python.scope": "作用域",
}

DELTA_TOLERANCE = 0.15  # |delta| < 0.15 视为「与锚点一致」


@dataclass
class ComparisonReport:
    per_topic: list[dict]
    per_bloom: list[dict]
    five_d_summary: dict[str, object]
    overall_bank: float | None
    overall_dialogue: float | None
    overall_delta: float | None
    agreement_notes: list[str]


# ── 统计工具（本阶段只写函数 + 已知数据单测，5-8 人轮才接真数据） ──────────


def _average_ranks(a: np.ndarray) -> np.ndarray:
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n) + 1.0
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = float(np.mean(ranks[order[i:j + 1]]))
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(xs, ys) -> float:
    """Spearman 秩相关系数（手写，含平局平均秩）.

    样本 < 2 或某侧无方差时返回 NaN（不能把无方差误判为强相关）。
    """
    x = np.asarray(xs, dtype=float).ravel()
    y = np.asarray(ys, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = np.sqrt(float((dx * dx).sum() * (dy * dy).sum()))
    if denom == 0:
        return float("nan")
    return float((dx * dy).sum() / denom)


def five_d_corr_matrix(records: list[SessionRecord]) -> np.ndarray | None:
    """5×5 五维估计相关系数矩阵（验 PRD 8b 维度区分度，S/X、C/K 边界）.

    < 2 条记录或无方差时对应位置为 NaN；全部不足则返回 None。
    """
    rows: list[list[float]] = []
    for rec in records:
        if rec.estimate and rec.estimate.five_d:
            rows.append([rec.estimate.five_d.get(DimensionId(d), float("nan"))
                         for d in DIM_ORDER])
    if len(rows) < 2:
        return None
    m = np.array(rows, dtype=float)
    corr = np.full((5, 5), float("nan"))
    for i in range(5):
        for j in range(5):
            mask = ~(np.isnan(m[:, i]) | np.isnan(m[:, j]))
            if mask.sum() >= 2 and np.std(m[mask, i]) > 0 and np.std(m[mask, j]) > 0:
                corr[i, j] = np.corrcoef(m[mask, i], m[mask, j])[0, 1]
    return corr


# ── n=1 比对 ─────────────────────────────────────────────────────────────


def _parse_node_id(node_id: str) -> tuple[str, BloomLevel] | None:
    """从 node_id（{topic}-L{bloom}-S{solo}-{dim}）解析出短 topic 与 Bloom 层.

    解析失败返回 None（不抛错，坏 anchor 不阻塞比对）。
    """
    m = re.match(r"^([a-z]+)-L(\d)-S(\d)-([KPSCX])$", node_id)
    if not m:
        return None
    short_topic = m.group(1)
    bloom = int(m.group(2))
    if short_topic not in TOPIC_SHORT_TO_ID or bloom not in range(1, 7):
        return None
    return short_topic, BloomLevel(bloom)


def _covered_sets(rec: SessionRecord) -> tuple[set[str], set[BloomLevel]]:
    covered_topics: set[str] = set()
    covered_blooms: set[BloomLevel] = set()
    for turn in rec.transcript:
        if turn.anchor:
            parsed = _parse_node_id(turn.anchor)
            if parsed:
                short, bloom = parsed
                covered_topics.add(TOPIC_SHORT_TO_ID[short])
                covered_blooms.add(bloom)
    return covered_topics, covered_blooms


def _solo_to_dialogue_est(rec: SessionRecord, topic: str) -> float | None:
    """逐 topic 对话估计：(solo-1)/4 归一化（结构性掌握代理指标）."""
    if not rec.estimate:
        return None
    short = TOPIC_ID_TO_SHORT.get(topic)
    if not short:
        return None
    solo = rec.estimate.solo.get(short)
    if solo is None:
        return None
    return max(0.0, min(1.0, (float(solo) - 1.0) / 4.0))


def compare_n1(rec: SessionRecord) -> ComparisonReport:
    """单会话比对：对话估计 vs 确定性题库 ground truth 锚点."""
    gt = rec.ground_truth
    est = rec.estimate
    covered_topics, covered_blooms = _covered_sets(rec)
    notes: list[str] = []

    per_topic: list[dict] = []
    if gt and gt.per_topic_bank:
        for topic in gt.per_topic_bank:
            bank_anchor = gt.per_topic_bank[topic]
            if topic in covered_topics:
                dial = _solo_to_dialogue_est(rec, topic)
                delta = (dial - bank_anchor) if dial is not None else None
                per_topic.append({
                    "topic": topic,
                    "bank_anchor": bank_anchor,
                    "dialogue_est": dial,
                    "delta": delta,
                    "covered": True,
                    "note": "",
                })
            else:
                per_topic.append({
                    "topic": topic,
                    "bank_anchor": bank_anchor,
                    "dialogue_est": None,
                    "delta": None,
                    "covered": False,
                    "note": "对话未覆盖，不比对",
                })

    per_bloom: list[dict] = []
    if gt and gt.per_bloom_bank:
        for name in BLOOM_ORDER:
            bloom = BloomLevel[name]
            bank_anchor = gt.per_bloom_bank.get(name)
            dial = est.bloom.get(bloom) if est else None
            covered = bloom in covered_blooms
            delta = (dial - bank_anchor) if (
                covered and dial is not None and bank_anchor is not None) else None
            per_bloom.append({
                "level": name,
                "bank_anchor": bank_anchor,
                "dialogue_est": dial,
                "delta": delta,
                "covered": covered,
                "note": "" if covered else "对话未覆盖，不比对",
            })

    five_d_summary: dict[str, object] = {"insufficient": []}
    if est:
        for dim in DimensionId:
            v = est.five_d.get(dim)
            five_d_summary[dim.value] = v if v is not None else None
        five_d_summary["insufficient"] = list(est.insufficient)

    overall_bank: float | None = None
    if gt and gt.per_topic_bank:
        vals = list(gt.per_topic_bank.values())
        if vals:
            overall_bank = float(np.mean(vals))
    overall_dialogue = float(est.overall) if est else None
    overall_delta = (overall_dialogue - overall_bank) if (
        overall_dialogue is not None and overall_bank is not None) else None

    # ── 中文描述性结论 ───────────────────────────────────────────────
    notes.append("逐 topic 的对话估计来自评分器输出的 SOLO 层级，(solo-1)/4 归一化，"
                 "是结构性掌握代理指标，非直接测量。")
    for row in per_topic:
        if not row["covered"] or row["delta"] is None:
            continue
        label = TOPIC_LABELS.get(row["topic"], row["topic"])
        delta = row["delta"]
        if abs(delta) < DELTA_TOLERANCE:
            notes.append(f"「{label}」topic 估计与锚点一致（Δ={delta:+.2f}）。")
        elif delta >= DELTA_TOLERANCE:
            notes.append(f"「{label}」topic 估计偏高（Δ={delta:+.2f}，锚点 "
                         f"{row['bank_anchor']:.2f} vs 对话 {row['dialogue_est']:.2f}）。")
        else:
            notes.append(f"「{label}」topic 估计偏低（Δ={delta:+.2f}，锚点 "
                         f"{row['bank_anchor']:.2f} vs 对话 {row['dialogue_est']:.2f}）。")
    for row in per_bloom:
        if not row["covered"] or row["delta"] is None:
            continue
        if abs(row["delta"]) >= DELTA_TOLERANCE:
            direction = "偏高" if row["delta"] > 0 else "偏低"
            notes.append(f"Bloom {row['level']} 层估计{direction}（Δ={row['delta']:+.2f}）。")
    if est and est.insufficient:
        notes.append("以下维度证据不足、估计缺失（评分器诚实标注）："
                     + "、".join(est.insufficient) + "。")
    if gt and gt.source == "bank_deterministic":
        notes.append("C/X/S 维度静态题库没有对应 ground truth（这正是 spike 要验证的），"
                     "五维摘要仅作记录，不判定与锚点的一致性；相关结论须等 5-8 人轮。")

    return ComparisonReport(
        per_topic=per_topic,
        per_bloom=per_bloom,
        five_d_summary=five_d_summary,
        overall_bank=overall_bank,
        overall_dialogue=overall_dialogue,
        overall_delta=overall_delta,
        agreement_notes=notes,
    )


def render_comparison(report: ComparisonReport) -> str:
    """把比对报告渲染成终端可读文本."""
    lines: list[str] = []
    lines.append("=" * 56)
    lines.append("会话比对报告（对话估计 vs 确定性题库锚点）")
    lines.append("=" * 56)

    lines.append("\n[逐 topic]")
    lines.append(f"  {'topic':<20}{'锚点':>8}{'对话估计':>10}{'Δ':>8}  说明")
    for row in report.per_topic:
        label = TOPIC_LABELS.get(row["topic"], row["topic"])
        est_s = f"{row['dialogue_est']:.2f}" if row["dialogue_est"] is not None else "  -"
        delta_s = f"{row['delta']:+.2f}" if row["delta"] is not None else "  -"
        lines.append(f"  {label:<20}{row['bank_anchor']:>8.2f}{est_s:>10}"
                     f"{delta_s:>8}  {row['note']}")

    lines.append("\n[逐 Bloom 层]")
    lines.append(f"  {'level':<12}{'锚点':>8}{'对话估计':>10}{'Δ':>8}  说明")
    for row in report.per_bloom:
        bank_s = f"{row['bank_anchor']:.2f}" if row["bank_anchor"] is not None else "  -"
        est_s = f"{row['dialogue_est']:.2f}" if row["dialogue_est"] is not None else "  -"
        delta_s = f"{row['delta']:+.2f}" if row["delta"] is not None else "  -"
        lines.append(f"  {row['level']:<12}{bank_s:>8}{est_s:>10}"
                     f"{delta_s:>8}  {row['note']}")

    lines.append("\n[五维估计摘要（对话）]")
    for dim in DIM_ORDER:
        v = report.five_d_summary.get(dim)
        v_s = f"{v:.2f}" if isinstance(v, float) else "  -"
        lines.append(f"  {dim} = {v_s}")
    ins = report.five_d_summary.get("insufficient")
    lines.append(f"  证据不足维度: {', '.join(ins) if ins else '无'}")

    lines.append("\n[整体]")
    ob = f"{report.overall_bank:.2f}" if report.overall_bank is not None else "  -"
    od = f"{report.overall_dialogue:.2f}" if report.overall_dialogue is not None else "  -"
    odel = f"{report.overall_delta:+.2f}" if report.overall_delta is not None else "  -"
    lines.append(f"  题库锚点整体: {ob}   对话估计整体: {od}   Δ: {odel}")

    lines.append("\n[结论]")
    for note in report.agreement_notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)
