"""5D 信念状态数据结构.

从 ECOS `ecos/cta/belief_state.py` 选择性迁移（见 MIGRATION.md 第2节）：
- 保留：5D (K/P/S/C/X) + BloomProfile + TC 状态 + 轨迹 + 校验 + 序列化
- 去掉：LearningDNA（ECOS 曾虚标完成度的模块）、Goal Ontology、Motivation、
  domain_extension、Evidence 关联、StateEngine versioning——这些耦合了
  ECOS 的 Multi-Domain 内核/教师端/防篡改机制，新项目 MVP 不需要
- 命名：student_id -> user_id（去掉"学生"隐含身份，见 MIGRATION.md 第2节）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np


class BloomLevel(Enum):
    """Bloom 认知层级 L1-L6."""

    REMEMBER = 1
    UNDERSTAND = 2
    APPLY = 3
    ANALYZE = 4
    EVALUATE = 5
    CREATE = 6


class DimensionId(Enum):
    """5D 状态的维度标识."""

    K = "K"  # Knowledge（知识掌握）
    P = "P"  # Procedure（程序技能）
    S = "S"  # Strategy（策略能力）
    C = "C"  # Confidence（认知置信度，含伪自信折扣）
    X = "X"  # External Support（外部支架）


DIM_CHARS = ("K", "P", "S", "C", "X")


@dataclass
class DimensionState:
    """单个维度的状态.

    Attributes:
        theta: MIRT 能力估计（连续值，ℝ）
        se: 标准误
        mastered: 二值掌握判定
        mastery_prob: 掌握概率
        confidence: 系统对该维度估计的置信度 0-1
        last_updated: 最近一次更新时间
        dimension: 维度字符
    """

    theta: float = 0.0
    se: float = 1.0
    mastered: bool = False
    mastery_prob: float = 0.5
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    dimension: str = "K"


@dataclass
class BloomProfileState:
    """Bloom 六层认知层级分布."""

    remember: float = 0.5
    understand: float = 0.5
    apply: float = 0.5
    analyze: float = 0.5
    evaluate: float = 0.5
    create: float = 0.5
    dominant_layer: BloomLevel = BloomLevel.UNDERSTAND
    confidence: float = 0.0
    covered_layers: set[BloomLevel] = field(default_factory=set)

    def as_vector(self) -> np.ndarray:
        """返回 6 维向量 [L1..L6] 顺序."""
        return np.array([
            self.remember,
            self.understand,
            self.apply,
            self.analyze,
            self.evaluate,
            self.create,
        ])

    def update_dominant(self) -> None:
        """根据 6 层概率重新判定 dominant_layer.

        只从 covered_layers（有题目/观测的层）里选：无覆盖的层停在先验 0.5，
        若纳入候选，全挂的学习者会被判成"主导 L5/L6"（自测发现，2026-08-22）。
        并列（多层同概率）时取最高层——ECOS 曾因 argmax 取最低层导致
        成长被低估（v0.96.4 修复），此处继承修复后的行为。
        全部 ≤ 0.5（无信号）时取最低层，避免全新画像跳到 L6。
        """
        probs = self.as_vector()
        covered_idx = sorted({layer.value - 1 for layer in self.covered_layers})
        if not covered_idx:
            self.dominant_layer = BloomLevel.REMEMBER
            return
        covered_probs = probs[covered_idx]
        max_val = float(covered_probs.max())
        max_positions = [i for i in covered_idx if probs[i] == max_val]
        idx = max(max_positions) if max_val > 0.5 else min(max_positions)
        self.dominant_layer = BloomLevel(idx + 1)

    def __post_init__(self) -> None:
        self.update_dominant()


@dataclass
class MisconceptionHit:
    """单次错误模式（misconception）命中.

    Attributes:
        misc_id: 错误模式标识，如 "M1"
        confidence: 命中置信度 0-1
        trigger_problem_id: 触发的题目 ID
        evidence_text: 触发文本
        timestamp: 命中时间
        correction_strategy: 修正策略 ID
    """

    misc_id: str
    confidence: float
    trigger_problem_id: str
    evidence_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    correction_strategy: str = ""


@dataclass
class TCState:
    """Threshold Concept（临界概念）状态.

    Attributes:
        tc_id: TC 标识，如 "TC_python_variables"
        status: "pre_liminal" / "liminal" / "post_liminal"
        progress: 0-1，跨越进度
        confidence: 系统对状态的置信度
        liminal_signals: 触发 liminal 的信号列表
        post_liminal_jump_detected: 是否检测到质变
        irreversible: TC 不可逆性
        timestamp: 状态更新时间
    """

    tc_id: str
    status: str = "pre_liminal"
    progress: float = 0.0
    confidence: float = 0.0
    liminal_signals: List[str] = field(default_factory=list)
    post_liminal_jump_detected: bool = False
    irreversible: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IllusoryConfidenceHit:
    """单次伪自信命中：自评置信度与实际表现落差过大.

    Attributes:
        problem_id: 题目 ID
        self_confidence: 答题前用户自评置信度 0-1
        score: 实际得分 0-1
        gap: self_confidence - score
        timestamp: 时间
    """

    problem_id: str
    self_confidence: float
    score: float
    gap: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConfidenceDimensionState(DimensionState):
    """C 维度扩展——含错误模式、TC 状态与伪自信记录."""

    misconception_hits: List[MisconceptionHit] = field(default_factory=list)
    tc_states: Dict[str, TCState] = field(default_factory=dict)
    illusory_confidence_hits: List[IllusoryConfidenceHit] = field(default_factory=list)
    illusory_confidence_flag: bool = False
    discount_factor: float = 1.0


@dataclass
class StateSnapshot:
    """单次状态快照（轨迹序列中的节点）."""

    timestamp: datetime
    theta_5d: np.ndarray
    bloom_profile: BloomProfileState
    confidence: float = 0.0


@dataclass
class TrajectoryState:
    """成长轨迹（时间序列）."""

    snapshots: List[StateSnapshot] = field(default_factory=list)

    def append(self, snapshot: StateSnapshot) -> None:
        self.snapshots.append(snapshot)

    def last_n(self, n: int) -> List[StateSnapshot]:
        return self.snapshots[-n:]


@dataclass
class BeliefState:
    """完整信念状态.

    Attributes:
        user_id: 用户标识
        K/P/S/C/X: 5D 各维度状态
        theta_mean: 5D 联合均值向量
        theta_cov: 5D 联合协方差矩阵 (5x5)
        bloom_profile: Bloom 六层分布
        trajectory: 时间序列轨迹
        overall_confidence: 整体置信度 0-1
        last_updated: 最近更新时间
    """

    user_id: str
    K: DimensionState = field(default_factory=lambda: DimensionState(dimension="K"))
    P: DimensionState = field(default_factory=lambda: DimensionState(dimension="P"))
    S: DimensionState = field(default_factory=lambda: DimensionState(dimension="S"))
    C: ConfidenceDimensionState = field(default_factory=lambda: ConfidenceDimensionState(dimension="C"))
    X: DimensionState = field(default_factory=lambda: DimensionState(dimension="X"))
    theta_mean: np.ndarray = field(default_factory=lambda: np.zeros(5))
    theta_cov: np.ndarray = field(default_factory=lambda: np.eye(5))
    bloom_profile: BloomProfileState = field(default_factory=BloomProfileState)
    trajectory: TrajectoryState = field(default_factory=TrajectoryState)
    overall_confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def theta_vector(self) -> np.ndarray:
        """返回 [θ_K, θ_P, θ_S, θ_C, θ_X] 5D 向量."""
        return np.array([self.K.theta, self.P.theta, self.S.theta, self.C.theta, self.X.theta])

    def mastery_vector(self) -> np.ndarray:
        """返回 5D mastery_prob 向量."""
        return np.array([
            self.K.mastery_prob,
            self.P.mastery_prob,
            self.S.mastery_prob,
            self.C.mastery_prob,
            self.X.mastery_prob,
        ])

    def snapshot(self) -> StateSnapshot:
        """生成当前状态快照（用于 trajectory 记录）."""
        return StateSnapshot(
            timestamp=self.last_updated,
            theta_5d=self.theta_vector(),
            bloom_profile=self.bloom_profile,
            confidence=self.overall_confidence,
        )

    def validate(self) -> tuple[bool, List[str]]:
        """Schema + 取值范围校验（soft，不 raise）."""
        issues: List[str] = []

        for dim_name in DIM_CHARS:
            dim = getattr(self, dim_name)
            if not (0.0 <= float(dim.mastery_prob) <= 1.0):
                issues.append(f"{dim_name}.mastery_prob={dim.mastery_prob} out of [0,1]")
            if not (0.0 <= float(dim.confidence) <= 1.0):
                issues.append(f"{dim_name}.confidence={dim.confidence} out of [0,1]")

        if not (0.0 <= float(self.C.discount_factor) <= 1.0):
            issues.append(f"C.discount_factor={self.C.discount_factor} out of [0,1]")
        for tc_id, tc in self.C.tc_states.items():
            if not (0.0 <= float(tc.progress) <= 1.0):
                issues.append(f"C.tc_states[{tc_id}].progress={tc.progress} out of [0,1]")
            if not (0.0 <= float(tc.confidence) <= 1.0):
                issues.append(f"C.tc_states[{tc_id}].confidence={tc.confidence} out of [0,1]")

        for field_name in ("remember", "understand", "apply", "analyze", "evaluate", "create", "confidence"):
            v = getattr(self.bloom_profile, field_name)
            if not (0.0 <= float(v) <= 1.0):
                issues.append(f"bloom_profile.{field_name}={v} out of [0,1]")

        if not (0.0 <= float(self.overall_confidence) <= 1.0):
            issues.append(f"overall_confidence={self.overall_confidence} out of [0,1]")

        if self.theta_mean.shape != (5,):
            issues.append(f"theta_mean.shape={self.theta_mean.shape} != (5,)")
        if self.theta_cov.shape != (5, 5):
            issues.append(f"theta_cov.shape={self.theta_cov.shape} != (5,5)")

        return (len(issues) == 0, issues)

    # ── 序列化 ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "K": _dim_to_dict(self.K),
            "P": _dim_to_dict(self.P),
            "S": _dim_to_dict(self.S),
            "C": _conf_dim_to_dict(self.C),
            "X": _dim_to_dict(self.X),
            "theta_mean": self.theta_mean.tolist(),
            "theta_cov": self.theta_cov.tolist(),
            "bloom_profile": _bloom_to_dict(self.bloom_profile),
            "trajectory": _traj_to_dict(self.trajectory),
            "overall_confidence": self.overall_confidence,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BeliefState":
        return cls(
            user_id=d.get("user_id", ""),
            K=_dim_from_dict(d.get("K", {}), default_dim="K"),
            P=_dim_from_dict(d.get("P", {}), default_dim="P"),
            S=_dim_from_dict(d.get("S", {}), default_dim="S"),
            C=_conf_dim_from_dict(d.get("C", {})),
            X=_dim_from_dict(d.get("X", {}), default_dim="X"),
            theta_mean=np.array(d.get("theta_mean", np.zeros(5).tolist())),
            theta_cov=np.array(d.get("theta_cov", np.eye(5).tolist())),
            bloom_profile=_bloom_from_dict(d.get("bloom_profile", {})),
            trajectory=_traj_from_dict(d.get("trajectory", {})),
            overall_confidence=float(d.get("overall_confidence", 0.0)),
            last_updated=_parse_iso(d.get("last_updated")),
        )


# ── 序列化 helper ──────────────────────────────────────────────────


def _iso(dt: Any) -> str:
    return dt.isoformat() if dt else ""


def _parse_iso(s: Any) -> datetime:
    if not s:
        return datetime.now()
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return datetime.now()


def _dim_to_dict(d: DimensionState) -> Dict[str, Any]:
    return {
        "theta": float(d.theta),
        "se": float(d.se),
        "mastered": bool(d.mastered),
        "mastery_prob": float(d.mastery_prob),
        "confidence": float(d.confidence),
        "last_updated": _iso(d.last_updated),
        "dimension": d.dimension,
    }


def _dim_from_dict(d: Dict[str, Any], default_dim: str = "K") -> DimensionState:
    return DimensionState(
        theta=float(d.get("theta", 0.0)),
        se=float(d.get("se", 1.0)),
        mastered=bool(d.get("mastered", False)),
        mastery_prob=float(d.get("mastery_prob", 0.5)),
        confidence=float(d.get("confidence", 0.0)),
        last_updated=_parse_iso(d.get("last_updated")),
        dimension=d.get("dimension", default_dim),
    )


def _conf_dim_to_dict(c: ConfidenceDimensionState) -> Dict[str, Any]:
    base = _dim_to_dict(c)
    base.update({
        "misconception_hits": [
            {
                "misc_id": h.misc_id,
                "confidence": float(h.confidence),
                "trigger_problem_id": h.trigger_problem_id,
                "evidence_text": h.evidence_text,
                "timestamp": _iso(h.timestamp),
                "correction_strategy": h.correction_strategy,
            }
            for h in c.misconception_hits
        ],
        "tc_states": {
            k: {
                "tc_id": v.tc_id,
                "status": v.status,
                "progress": float(v.progress),
                "confidence": float(v.confidence),
                "liminal_signals": list(v.liminal_signals),
                "post_liminal_jump_detected": bool(v.post_liminal_jump_detected),
                "irreversible": bool(v.irreversible),
                "timestamp": _iso(v.timestamp),
            }
            for k, v in c.tc_states.items()
        },
        "illusory_confidence_hits": [
            {
                "problem_id": h.problem_id,
                "self_confidence": float(h.self_confidence),
                "score": float(h.score),
                "gap": float(h.gap),
                "timestamp": _iso(h.timestamp),
            }
            for h in c.illusory_confidence_hits
        ],
        "illusory_confidence_flag": bool(c.illusory_confidence_flag),
        "discount_factor": float(c.discount_factor),
    })
    return base


def _conf_dim_from_dict(d: Dict[str, Any]) -> ConfidenceDimensionState:
    base = _dim_from_dict(d, default_dim="C")
    return ConfidenceDimensionState(
        theta=base.theta,
        se=base.se,
        mastered=base.mastered,
        mastery_prob=base.mastery_prob,
        confidence=base.confidence,
        last_updated=base.last_updated,
        dimension=base.dimension,
        misconception_hits=[
            MisconceptionHit(
                misc_id=h["misc_id"],
                confidence=float(h.get("confidence", 0.0)),
                trigger_problem_id=h.get("trigger_problem_id", ""),
                evidence_text=h.get("evidence_text", ""),
                timestamp=_parse_iso(h.get("timestamp")),
                correction_strategy=h.get("correction_strategy", ""),
            )
            for h in d.get("misconception_hits", [])
        ],
        tc_states={
            k: TCState(
                tc_id=v["tc_id"],
                status=v.get("status", "pre_liminal"),
                progress=float(v.get("progress", 0.0)),
                confidence=float(v.get("confidence", 0.0)),
                liminal_signals=list(v.get("liminal_signals", [])),
                post_liminal_jump_detected=bool(v.get("post_liminal_jump_detected", False)),
                irreversible=bool(v.get("irreversible", False)),
                timestamp=_parse_iso(v.get("timestamp")),
            )
            for k, v in d.get("tc_states", {}).items()
        },
        illusory_confidence_hits=[
            IllusoryConfidenceHit(
                problem_id=h.get("problem_id", ""),
                self_confidence=float(h.get("self_confidence", 0.0)),
                score=float(h.get("score", 0.0)),
                gap=float(h.get("gap", 0.0)),
                timestamp=_parse_iso(h.get("timestamp")),
            )
            for h in d.get("illusory_confidence_hits", [])
        ],
        illusory_confidence_flag=bool(d.get("illusory_confidence_flag", False)),
        discount_factor=float(d.get("discount_factor", 1.0)),
    )


def _bloom_to_dict(b: BloomProfileState) -> Dict[str, Any]:
    return {
        "remember": float(b.remember),
        "understand": float(b.understand),
        "apply": float(b.apply),
        "analyze": float(b.analyze),
        "evaluate": float(b.evaluate),
        "create": float(b.create),
        "dominant_layer": b.dominant_layer.name,
        "confidence": float(b.confidence),
        "covered_layers": [l.name for l in sorted(b.covered_layers, key=lambda l: l.value)],
    }


def _bloom_from_dict(d: Dict[str, Any]) -> BloomProfileState:
    try:
        dominant = BloomLevel[d.get("dominant_layer", "UNDERSTAND")]
    except KeyError:
        dominant = BloomLevel.UNDERSTAND
    covered = set()
    for name in d.get("covered_layers", []):
        try:
            covered.add(BloomLevel[name])
        except KeyError:
            continue
    return BloomProfileState(
        remember=float(d.get("remember", 0.5)),
        understand=float(d.get("understand", 0.5)),
        apply=float(d.get("apply", 0.5)),
        analyze=float(d.get("analyze", 0.5)),
        evaluate=float(d.get("evaluate", 0.5)),
        create=float(d.get("create", 0.5)),
        dominant_layer=dominant,
        confidence=float(d.get("confidence", 0.0)),
        covered_layers=covered,
    )


def _traj_to_dict(t: TrajectoryState) -> Dict[str, Any]:
    return {
        "snapshots": [
            {
                "timestamp": _iso(s.timestamp),
                "theta_5d": s.theta_5d.tolist() if hasattr(s.theta_5d, "tolist") else list(s.theta_5d),
                "bloom_profile": _bloom_to_dict(s.bloom_profile),
                "confidence": float(s.confidence),
            }
            for s in t.snapshots
        ]
    }


def _traj_from_dict(d: Dict[str, Any]) -> TrajectoryState:
    snapshots = []
    for s in d.get("snapshots", []):
        snapshots.append(StateSnapshot(
            timestamp=_parse_iso(s.get("timestamp")),
            theta_5d=np.array(s.get("theta_5d", np.zeros(5).tolist())),
            bloom_profile=_bloom_from_dict(s.get("bloom_profile", {})),
            confidence=float(s.get("confidence", 0.0)),
        ))
    return TrajectoryState(snapshots=snapshots)
