"""会话记录与读写（JSONL）.

命名沿用 user_id（GOVERNANCE/CLAUDE.md 约定，非 student_id）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cogmirror.belief_state import BloomLevel
from cogmirror.questions import QuestionBank

from .dialogue import AnchorTurn, ExecResult
from .graph import DimensionId
from .scorer import ScorerOutput

GRAPH_VERSION = "0.1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class GroundTruthAnchors:
    """独立 ground truth 锚点（确定性题库当场判分产出）.

    source: 锚点来源（当前固定 "bank_deterministic"）
    per_topic_bank: topic -> 平均得分 0-1
    per_bloom_bank: BloomLevel.name -> 平均得分 0-1
    per_topic_correct: topic -> 原始作答计数 {answered, correct, total_score}
    """

    source: str
    per_topic_bank: dict[str, float]
    per_bloom_bank: dict[str, float]
    per_topic_correct: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "per_topic_bank": self.per_topic_bank,
            "per_bloom_bank": self.per_bloom_bank,
            "per_topic_correct": self.per_topic_correct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruthAnchors":
        return cls(
            source=str(d.get("source", "bank_deterministic")),
            per_topic_bank={str(k): float(v) for k, v in d.get("per_topic_bank", {}).items()},
            per_bloom_bank={str(k): float(v) for k, v in d.get("per_bloom_bank", {}).items()},
            per_topic_correct={str(k): dict(v)
                               for k, v in d.get("per_topic_correct", {}).items()},
        )


def build_bank_anchors(bank: QuestionBank,
                       answers_by_problem_id: dict[str, str]) -> GroundTruthAnchors:
    """用确定性题库判分生成 ground truth 锚点（逐 topic / 逐 Bloom 层）.

    answers_by_problem_id: {problem_id: 用户作答原文}；未知题目跳过。
    """
    per_topic_score: dict[str, list[float]] = defaultdict(list)
    per_bloom_score: dict[str, list[float]] = defaultdict(list)
    per_topic_correct: dict[str, dict] = defaultdict(
        lambda: {"answered": 0, "correct": 0, "total_score": 0.0})

    for pid, answer in answers_by_problem_id.items():
        q = bank.get(pid)
        if q is None:
            continue
        score, _ = bank.grade_answer(q, answer)
        per_topic_score[q.topic].append(score)
        per_bloom_score[q.bloom_level.name].append(score)
        c = per_topic_correct[q.topic]
        c["answered"] += 1
        c["total_score"] += score
        if score >= 1.0:
            c["correct"] += 1

    return GroundTruthAnchors(
        source="bank_deterministic",
        per_topic_bank={t: sum(v) / len(v) for t, v in per_topic_score.items()},
        per_bloom_bank={b: sum(v) / len(v) for b, v in per_bloom_score.items()},
        per_topic_correct={k: dict(v) for k, v in per_topic_correct.items()},
    )


@dataclass
class SessionRecord:
    """一次完整诊断会话（对话 + 锚点 + 估计）.

    date: ISO 时间戳；model: 使用的 LLM 模型名
    estimate 为 None 表示评分尚未完成/失败（诚实标注，不假装有估计）
    """

    user_id: str
    date: str
    graph_version: str
    model: str
    ground_truth: GroundTruthAnchors | None
    transcript: list[AnchorTurn]
    exec_results: list[ExecResult]
    estimate: ScorerOutput | None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "date": self.date,
            "graph_version": self.graph_version,
            "model": self.model,
            "ground_truth": self.ground_truth.to_dict() if self.ground_truth else None,
            "transcript": [
                {"role": t.role, "text": t.text, "anchor": t.anchor}
                for t in self.transcript
            ],
            "exec_results": [
                {
                    "node_id": e.node_id,
                    "submitted_code": e.submitted_code,
                    "score": e.score,
                    "details": e.details,
                    "executed": e.executed,
                }
                for e in self.exec_results
            ],
            "estimate": _estimate_to_dict(self.estimate) if self.estimate else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRecord":
        gt = GroundTruthAnchors.from_dict(d["ground_truth"]) if d.get("ground_truth") else None
        transcript = [
            AnchorTurn(role=str(t.get("role", "user")), text=str(t.get("text", "")),
                       anchor=t.get("anchor"))
            for t in d.get("transcript", [])
        ]
        exec_results = [
            ExecResult(node_id=str(e.get("node_id", "")),
                       submitted_code=str(e.get("submitted_code", "")),
                       score=float(e.get("score", 0.0)),
                       details=list(e.get("details", [])),
                       executed=bool(e.get("executed", True)))
            for e in d.get("exec_results", [])
        ]
        return cls(
            user_id=str(d.get("user_id", "")),
            date=str(d.get("date", "")),
            graph_version=str(d.get("graph_version", "")),
            model=str(d.get("model", "")),
            ground_truth=gt,
            transcript=transcript,
            exec_results=exec_results,
            estimate=_estimate_from_dict(d["estimate"]) if d.get("estimate") else None,
        )


def _estimate_to_dict(e: ScorerOutput) -> dict:
    return {
        "five_d": {d.value: float(v) for d, v in e.five_d.items()},
        "bloom": {b.name: float(v) for b, v in e.bloom.items()},
        "solo": {k: float(v) for k, v in e.solo.items()},
        "overall": float(e.overall),
        "evidence_notes": dict(e.evidence_notes),
        "insufficient": list(e.insufficient),
    }


def _estimate_from_dict(d: dict) -> ScorerOutput:
    five_d: dict[DimensionId, float] = {}
    for k, v in d.get("five_d", {}).items():
        try:
            five_d[DimensionId(str(k))] = float(v)
        except ValueError:
            continue
    bloom: dict[BloomLevel, float] = {}
    for k, v in d.get("bloom", {}).items():
        try:
            bloom[BloomLevel[str(k)]] = float(v)
        except KeyError:
            continue
    return ScorerOutput(
        five_d=five_d,
        bloom=bloom,
        solo={str(k): float(v) for k, v in d.get("solo", {}).items()},
        overall=float(d.get("overall", 0.0)),
        evidence_notes={str(k): str(v) for k, v in d.get("evidence_notes", {}).items()},
        insufficient=[str(x) for x in d.get("insufficient", [])],
    )


def save_session(path: str | Path, rec: SessionRecord) -> Path:
    """追加一条会话记录到 JSONL 文件."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    return p


def load_sessions(directory: str | Path) -> tuple[list[SessionRecord], list[str]]:
    """读取目录下所有 .jsonl 会话；坏行跳过并记录原因（不抛错）."""
    d = Path(directory)
    records: list[SessionRecord] = []
    errors: list[str] = []
    if not d.is_dir():
        return records, errors
    for path in sorted(d.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(SessionRecord.from_dict(json.loads(line)))
                except Exception as e:  # noqa: BLE001 - 坏行跳过并记录
                    errors.append(f"{path.name}:{lineno}: {type(e).__name__}: {e}")
    return records, errors
