"""黄金回归 runner（P1）：replay 序列 -> 客观 scorer + baseline 数值对比.

移植 PersonalAGI `eval/gauntlet_regression.py` 的纪律（docs/implementation-plan.md
第 2.2/2.4 节）：

- 固定黄金集（sequences.py），每次变更跑全量；
- 客观 scorer 逐 case 判 pass（expect 窗口断言）；
- baseline 存数值摘要（不存整份 state，避免无关字段差异造成假回归），
  对比带容差（scipy/BLAS 版本间 1e-9 级微漂移，全等断言会碎）；
- 回归判定 = 本次 FAIL 且 baseline 中该 case 曾 PASS；冷启动（baseline 未
  收录）不报回归；恢复（PASS）自然撤回告警；
- 永不自动回滚，只 FAIL 告警。

复用产品公开接口（BeliefEngine.update / next_suggestion / map_interpretation /
suggested_practice），不驱动 CLI 交互流。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cogmirror.belief_engine import BeliefEngine, Observation
from cogmirror.cli import (
    map_interpretation,
    next_suggestion,
    practice_command,
    suggested_practice,
)
from cogmirror.questions import QuestionBank

from .sequences import SEQUENCES

GOLDEN_DIR = Path(__file__).parent
BASELINE_PATH = GOLDEN_DIR / "baseline.json"

# 数值摘要对比容差：覆盖跨 scipy/BLAS 版本的 1e-9 级微漂移，
# 同时远小于任何有语义变化的量级（折扣/掌握概率的最小有意义变化 >1e-3）
TOL = 1e-6

# 全部观测时间固定为该基准 + 序号分钟数（Observation 默认 datetime.now() 不可复现）
BASE_TIME = datetime(2026, 1, 1, 12, 0, 0)

_BLOOM_FIELDS = ("remember", "understand", "apply", "analyze", "evaluate", "create")
_DIMS = ("K", "P", "S", "C", "X")


@dataclass
class CaseResult:
    """单条黄金序列的 replay 结果."""

    name: str
    engine: BeliefEngine
    state: Any
    summary: dict
    expect_failures: list[str]


def replay(sequence: dict) -> CaseResult:
    """构造全新引擎，按 steps 依次 replay（skill/bloom 从题库查，与产品同源）."""
    bank = QuestionBank()
    by_id = {q.problem_id: q for q in bank.all_questions()}
    engine = BeliefEngine()
    engine.l2.register_items_bulk(bank.mirt_items())
    state = engine.create_initial_state("golden-" + sequence["name"])
    for i, step in enumerate(sequence["steps"]):
        q = by_id[step["problem_id"]]
        obs = Observation(
            skill_id=q.skill_id,
            problem_id=q.problem_id,
            score=step["score"],
            bloom_level=q.bloom_level,
            self_confidence=step.get("self_confidence"),
            explanation_text=step.get("explanation_text", ""),
            user_answer=step.get("user_answer", ""),
            correct_answer=step.get("correct_answer", ""),
            timestamp=BASE_TIME + timedelta(minutes=i),
        )
        state = engine.update(state, obs)
    return CaseResult(
        name=sequence["name"],
        engine=engine,
        state=state,
        summary=numeric_summary(engine, state),
        expect_failures=score_expect(sequence, engine, state),
    )


# ── 客观 scorer：expect 窗口断言 ─────────────────────────────────

def score_expect(sequence: dict, engine: BeliefEngine, state: Any) -> list[str]:
    """逐项检查 expect 窗口，返回失败描述列表（空 = 全过）."""
    failures: list[str] = []
    expect = sequence.get("expect", {})

    for dim, (lo, hi) in expect.get("mastery", {}).items():
        v = getattr(state, dim).mastery_prob
        if not (lo <= v <= hi):
            failures.append(f"mastery[{dim}]={v:.4f} 不在窗口 [{lo}, {hi}]")

    for skill, (lo, hi) in expect.get("bkt", {}).items():
        v = engine.get_bkt_mastery(skill)
        if not (lo <= v <= hi):
            failures.append(f"bkt[{skill}]={v:.4f} 不在窗口 [{lo}, {hi}]")

    if "dominant_bloom" in expect:
        actual = state.bloom_profile.dominant_layer.name
        if actual != expect["dominant_bloom"]:
            failures.append(f"dominant_bloom={actual}，期望 {expect['dominant_bloom']}")

    for skill, want in expect.get("tc", {}).items():
        tc = state.C.tc_states.get(skill)
        actual = tc.status if tc is not None else "(未创建)"
        if actual != want:
            failures.append(f"tc[{skill}]={actual}，期望 {want}")

    for key in ("illusory_hits", "misc_hits"):
        if key in expect:
            actual = (len(state.C.illusory_confidence_hits) if key == "illusory_hits"
                      else len(state.C.misconception_hits))
            if actual != expect[key]:
                failures.append(f"{key}={actual}，期望 {expect[key]}")

    if "discount_factor" in expect:
        lo, hi = expect["discount_factor"]
        v = state.C.discount_factor
        if not (lo <= v <= hi):
            failures.append(f"discount_factor={v:.4f} 不在窗口 [{lo}, {hi}]")

    suggestion = next_suggestion(engine, state)
    for kw in expect.get("suggestion_contains", []):
        if kw not in suggestion:
            failures.append(f"建议文案缺少关键词「{kw}」: {suggestion}")

    interpretation = map_interpretation(engine, state)
    for kw in expect.get("interpretation_contains", []):
        if kw not in interpretation:
            failures.append(f"整体解读缺少关键词「{kw}」: {interpretation}")

    if "practice" in expect:
        want = expect["practice"]
        actual = _practice_tuple(engine, state)
        if actual != want:
            failures.append(f"suggested_practice={actual}，期望 {want}")

    return failures


def _practice_tuple(engine: BeliefEngine, state: Any) -> tuple | None:
    """suggested_practice 归一化为可 JSON 序列化的 (topic, level_name | None)."""
    target = suggested_practice(engine, state)
    if target is None:
        return None
    topic, level = target
    return (topic, level.name if level is not None else None)


# ── baseline：数值摘要存取与对比 ─────────────────────────────────

def numeric_summary(engine: BeliefEngine, state: Any) -> dict:
    """case 的数值摘要（只存有语义的量，不存整份 state 序列化）."""
    return {
        "mastery": {d: getattr(state, d).mastery_prob for d in _DIMS},
        "theta": {d: getattr(state, d).theta for d in _DIMS},
        "discount_factor": state.C.discount_factor,
        "overall_confidence": state.overall_confidence,
        "bloom": {f: getattr(state.bloom_profile, f) for f in _BLOOM_FIELDS},
        "dominant_bloom": state.bloom_profile.dominant_layer.name,
        "bkt_mastery": {s: engine.get_bkt_mastery(s) for s in sorted(engine.l1.all_skills())},
        "tc": {
            skill: {"status": tc.status, "progress": tc.progress}
            for skill, tc in sorted(state.C.tc_states.items())
        },
        "illusory_hits": len(state.C.illusory_confidence_hits),
        "misc_hits": len(state.C.misconception_hits),
        "suggestion": next_suggestion(engine, state),
        "interpretation": map_interpretation(engine, state),
        "practice": _practice_tuple(engine, state),
        "practice_command": practice_command(engine, state),
    }


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def write_baseline(baseline: dict) -> None:
    BASELINE_PATH.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def compare_summary(cur: dict, base: dict, tol: float = TOL) -> list[str]:
    """当前数值摘要 vs 基线摘要（带容差），返回漂移描述列表."""
    drifts: list[str] = []
    for key in ("mastery", "theta", "bkt_mastery", "bloom"):
        for k, v in cur.get(key, {}).items():
            bv = base.get(key, {}).get(k)
            if bv is None:
                drifts.append(f"{key}.{k}: 基线无此条目（新字段？）")
            elif abs(float(v) - float(bv)) > tol:
                drifts.append(f"{key}.{k}: {bv:.6f} -> {v:.6f}")
    for key in ("discount_factor", "overall_confidence"):
        if key in cur:
            bv = base.get(key)
            if bv is None or abs(float(cur[key]) - float(bv)) > tol:
                drifts.append(f"{key}: {bv} -> {cur[key]}")
    for skill, tc in cur.get("tc", {}).items():
        b = base.get("tc", {}).get(skill)
        if b is None:
            drifts.append(f"tc.{skill}: 基线无此条目")
        elif b["status"] != tc["status"] or abs(b["progress"] - tc["progress"]) > tol:
            drifts.append(f"tc.{skill}: {b['status']}/{b['progress']:.3f} -> "
                          f"{tc['status']}/{tc['progress']:.3f}")
    for key in ("dominant_bloom", "suggestion", "interpretation", "practice",
                "practice_command", "illusory_hits", "misc_hits"):
        cur_v = cur.get(key)
        # JSON 往返把 tuple 变 list，对比前归一化（practice 可能是 (topic, level)）
        if isinstance(cur_v, tuple):
            cur_v = list(cur_v)
        if cur_v != base.get(key):
            drifts.append(f"{key}: {base.get(key)!r} -> {cur.get(key)!r}")
    return drifts


# ── 单 case 完整评估（expect + baseline 回归判定） ────────────────

def evaluate(sequence: dict) -> tuple[list[str], dict | None]:
    """跑一条序列，返回 (失败列表, baseline 条目)。

    失败 = expect 窗口未命中 ∪ baseline 数值漂移。
    回归判定（本次 FAIL 且 baseline 曾 PASS）在 test 层用返回的 baseline
    条目标注，这里只产出客观事实。
    """
    result = replay(sequence)
    failures = list(result.expect_failures)
    baseline = load_baseline()
    entry = baseline.get("cases", {}).get(sequence["name"]) if baseline else None
    if entry is not None:
        failures += compare_summary(result.summary, entry["summary"])
    return failures, entry


def run_all() -> list[tuple[dict, CaseResult]]:
    return [(seq, replay(seq)) for seq in SEQUENCES]
