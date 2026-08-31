"""5D 信念引擎（简化版）.

从 ECOS `ecos/cta/belief_engine.py` 及其 4 层架构
（observation_engine / feature_extractor / inference_engine / belief_updater）
合并改造而来（见 MIGRATION.md 第2节）：

- 保留：partial credit 判分派生（score >= 0.6 -> correct）、响应历史累积、
  BKT 更新、5D MIRT MAP 估计、Bloom 六层更新、TC 状态检测、
  misconception 折扣、整体置信度、轨迹快照
- 去掉：warm-up/探针题状态机（K12 体验设计）、LLM Critic 感知层
  （新项目 MVP 不依赖 LLM）、EventLog / StateEngine / EvidenceEngine
  （ECOS 的防篡改与审计基建，MVP 不需要）
- 新增：自评置信度采集与伪自信检测（PRD 5.1 P0）--答题前用户自评
  置信度，与实际得分对比，落差过大记为 illusory confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .belief_state import (
    BeliefState,
    BloomLevel,
    DIM_CHARS,
    IllusoryConfidenceHit,
    MisconceptionHit,
)
from .bkt import BKTModel, BKTEvolutionLayer, EvolutionConfig
from .calibration import CalibrationCurve, CalibrationCurveComputer
from .content.misconceptions import PythonBasicsMisconceptionLibrary
from .mirt import BiFactorMIRT5D, MIRTConfig
from .misconception_tracker import MisconceptionTracker
from .tc import TCStateDetector

# 伪自信判定阈值：自评置信度 - 实际得分 >= 该值且自评较高时记为伪自信
ILLUSORY_GAP_THRESHOLD = 0.5
ILLUSORY_SELF_CONF_MIN = 0.7
# 伪自信命中对 C 维度掌握概率的折扣：自评与表现脱节 -> C 下调。
# 无校准曲线（数据不足）时的回退值；曲线驱动路径见 _calibrated_discount
ILLUSORY_MASTERY_DISCOUNT = 0.15
# 曲线驱动折扣的夹取范围：下限防过激惩罚、上限防自评满分桶把 C 打穿
CALIBRATED_DISCOUNT_MIN = 0.05
CALIBRATED_DISCOUNT_MAX = 0.5

HISTORY_MAXLEN = 100


@dataclass
class Observation:
    """单次作答观测.

    Attributes:
        skill_id: 涉及的知识点 ID（用于 BKT / TC）
        problem_id: 题目 ID（用于 MIRT）
        score: partial credit 评分 0.0-1.0（1.0=完全对）
        bloom_level: 题目对应的 Bloom 层级
        self_confidence: 答题前用户自评置信度 0.0-1.0（伪自信检测输入）
        explanation_text: 用户解释文本（可选，用于 misconception 关键词检测）
        user_answer: 用户提交的原始答案
        correct_answer: 正确答案
        timestamp: 观测时间
    """

    skill_id: str
    problem_id: str
    score: float = 0.0
    bloom_level: BloomLevel = BloomLevel.APPLY
    self_confidence: Optional[float] = None
    explanation_text: str = ""
    user_answer: str = ""
    correct_answer: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def correct(self) -> bool:
        """partial credit 派生：score >= 0.6 判对（ECOS v0.54.0 语义）."""
        return self.score >= 0.6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "problem_id": self.problem_id,
            "score": float(self.score),
            "bloom_level": self.bloom_level.name,
            "self_confidence": self.self_confidence,
            "explanation_text": self.explanation_text,
            "user_answer": self.user_answer,
            "correct_answer": self.correct_answer,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BeliefEngineConfig:
    """BeliefEngine 聚合配置."""

    evolution_config: EvolutionConfig = field(default_factory=EvolutionConfig)
    mirt_config: MIRTConfig = field(default_factory=MIRTConfig)
    # 步长足够大让六层分布拉开可见差距：0.05 时全对也挤在 0.5-0.65，
    # 自测反馈"六层数值差异不大"（2026-08-22）
    bloom_update_step: float = 0.12
    trajectory_maxlen: int = 500


class BeliefEngine:
    """5D 信念引擎.

    主入口：
        engine = BeliefEngine()
        state = engine.create_initial_state("user_001")
        state = engine.update(state, observation)

    每次更新按顺序执行：
        1. 累积响应历史（MIRT 输入）
        2. BKT 更新（skill 粒度掌握概率）
        3. MIRT MAP 估计 -> 5D theta / se / mastery_prob / confidence
        4. Bloom 六层更新（partial credit 驱动）
        5. misconception 关键词检测 -> C 维度折扣
        6. TC 状态检测（liminal 三态机）
        7. 伪自信检测（自评 vs 实际）
        8. overall_confidence + 轨迹快照
    """

    def __init__(
        self,
        config: BeliefEngineConfig | None = None,
        misconception_library: PythonBasicsMisconceptionLibrary | None = None,
        misconception_tracker: MisconceptionTracker | None = None,
    ) -> None:
        self.config = config or BeliefEngineConfig()
        self.l1 = BKTEvolutionLayer(self.config.evolution_config)
        self.l2 = BiFactorMIRT5D(self.config.mirt_config)
        self.tc_detector = TCStateDetector()
        self.misconception_library = misconception_library or PythonBasicsMisconceptionLibrary()
        # P4：misconception 证据追踪（None = 无证据数据，命中置信度回退固定 0.6，
        # 与迁移前行为一致）。由 CLI 从 DB 加载后注入
        self.misconception_tracker = misconception_tracker
        # user_id -> 响应历史（MIRT 输入 + 详情回看）
        self._response_history: Dict[str, List[Dict[str, Any]]] = {}
        # 校准曲线（P2）：None = 无校准数据，伪自信折扣走固定回退值。
        # 由 CLI 从 responses 历史重算后注入（单用户派生视图，不落库）
        self.calibration_curves: Optional[List[CalibrationCurve]] = None
        # 间隔衰减视图（P3）：{skill: (peak, decayed, days)}，None = 未计算。
        # 由 CLI 从 responses 历史重算后注入（只读派生，不落库）；
        # next_suggestion / 复测提示消费它
        self.decay_view: Optional[Dict[str, Tuple[float, float, int]]] = None

    # ── 状态创建 ────────────────────────────────────────────────────

    def create_initial_state(self, user_id: str) -> BeliefState:
        """创建新用户的初始 BeliefState."""
        state = BeliefState(user_id=user_id)
        state.theta_mean = np.zeros(5)
        state.theta_cov = np.eye(5)
        state.bloom_profile.update_dominant()
        state.overall_confidence = 0.0
        state.last_updated = datetime.now()
        return state

    # ── 主更新入口 ──────────────────────────────────────────────────

    def update(self, state: BeliefState, observation: Observation) -> BeliefState:
        """每次新观测后调用，返回更新后的 BeliefState（原地更新）."""
        user_id = state.user_id
        score = float(np.clip(observation.score, 0.0, 1.0))
        correct = score >= 0.6  # partial credit 派生，ECOS v0.54.0 语义

        # Step 1: 响应历史累积
        history = self._response_history.setdefault(user_id, [])
        history.append({
            "problem_id": observation.problem_id,
            "correct": int(correct),
            "score": score,
            "bloom_level": observation.bloom_level.name,
            "self_confidence": observation.self_confidence,
            "user_answer": observation.user_answer,
            "correct_answer": observation.correct_answer,
            "timestamp": observation.timestamp.isoformat(),
        })
        if len(history) > HISTORY_MAXLEN:
            self._response_history[user_id] = history[-HISTORY_MAXLEN:]
            history = self._response_history[user_id]

        # Step 2: BKT 更新（知识点粒度）
        self.l1.update(observation.skill_id, correct)

        # Step 3: MIRT MAP 估计 -> 5D 各维度
        if len(history) >= 2:
            problem_ids = [h["problem_id"] for h in history]
            responses = np.array([h["score"] for h in history], dtype=float)
            theta_hat, theta_cov = self.l2.estimate_theta(responses, problem_ids)
            state.theta_mean = theta_hat.copy()
            state.theta_cov = theta_cov.copy()
            for i, dim_char in enumerate(DIM_CHARS):
                dim = getattr(state, dim_char)
                dim.theta = float(theta_hat[i])
                dim.se = float(np.sqrt(max(theta_cov[i, i], 1e-6)))
                dim.mastery_prob = float(1.0 / (1.0 + np.exp(-theta_hat[i])))
                dim.mastered = dim.mastery_prob >= 0.5
                dim.confidence = float(1.0 / (1.0 + dim.se))
                dim.last_updated = observation.timestamp

        # Step 4: Bloom 六层更新（partial credit 驱动）
        bloom_name = observation.bloom_level.name.lower()
        current_prob = float(getattr(state.bloom_profile, bloom_name))
        bloom_delta = (score - 0.5) * 2.0 * self.config.bloom_update_step
        new_prob = max(0.0, min(1.0, current_prob + bloom_delta))
        setattr(state.bloom_profile, bloom_name, new_prob)
        # 记录该层有观测：主导层级只从有数据的层里选，避免未练层停
        # 先验 0.5 却压过练过但失败的层（自测发现，2026-08-22）
        state.bloom_profile.covered_layers.add(observation.bloom_level)
        state.bloom_profile.update_dominant()
        state.bloom_profile.confidence = min(1.0, len(history) / 30.0)

        # Step 5: misconception 检测（关键词路径，非 LLM）-> C 维度折扣
        misc_hit = self._detect_misconception(observation)
        if misc_hit is not None:
            state.C.misconception_hits.append(misc_hit)
            discount = 1.0 - min(misc_hit.confidence * 0.3, 0.3)
            state.C.discount_factor = min(state.C.discount_factor * discount, 1.0)

        # Step 6: TC 状态检测
        updated_tc = self.tc_detector.detect(
            topic=observation.skill_id,
            correct=correct,
            bloom_level=observation.bloom_level,
            current_tc_state=state.C.tc_states.get(observation.skill_id),
            has_active_misc=misc_hit is not None,
        )
        state.C.tc_states[observation.skill_id] = updated_tc

        # Step 7: 伪自信检测（自评 vs 实际）
        if observation.self_confidence is not None:
            gap = float(observation.self_confidence) - score
            if gap >= ILLUSORY_GAP_THRESHOLD and observation.self_confidence >= ILLUSORY_SELF_CONF_MIN:
                state.C.illusory_confidence_hits.append(IllusoryConfidenceHit(
                    problem_id=observation.problem_id,
                    self_confidence=float(observation.self_confidence),
                    score=score,
                    gap=gap,
                    timestamp=observation.timestamp,
                ))
                state.C.illusory_confidence_flag = True
                # C 维度反馈：伪自信命中 = 自评与表现脱节，累计进持久折扣因子。
                # 折扣幅度曲线驱动（自评所在桶的历史真实答对率），
                # 数据不足回退固定值（方案 P2）
                discount = self._calibrated_discount(float(observation.self_confidence))
                state.C.discount_factor = min(
                    state.C.discount_factor * (1.0 - discount), 1.0)

        # C 维度掌握概率 = sigmoid(theta_C) × discount_factor（伪自信/misconception
        # 校准信号持久化）。必须在每次更新末尾重算：Step 3 的 MIRT 会整体重算
        # mastery_prob，若只在命中时直接改会被下一次 MIRT 覆盖。
        state.C.mastery_prob = float(np.clip(
            (1.0 / (1.0 + np.exp(-state.C.theta))) * state.C.discount_factor, 0.0, 1.0))
        state.C.mastered = state.C.mastery_prob >= 0.5

        # Step 8: overall_confidence + 轨迹快照
        state.overall_confidence = float(np.mean([
            state.K.confidence, state.P.confidence,
            state.S.confidence, state.C.confidence, state.X.confidence,
        ]))
        state.last_updated = observation.timestamp
        state.trajectory.append(state.snapshot())
        if len(state.trajectory.snapshots) > self.config.trajectory_maxlen:
            state.trajectory.snapshots = state.trajectory.snapshots[-self.config.trajectory_maxlen:]

        return state

    # ── 查询接口 ────────────────────────────────────────────────────

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        """获取响应历史（空列表兜底）."""
        return self._response_history.get(user_id, [])

    def set_history(self, user_id: str, history: List[Dict[str, Any]]) -> None:
        """恢复响应历史（DB restore 路径）."""
        self._response_history[user_id] = history

    def get_bkt_mastery(self, skill_id: str) -> float:
        """获取 BKT 当前掌握概率."""
        return self.l1.get_mastery(skill_id)

    def set_calibration(self, curves: List[CalibrationCurve] | None) -> None:
        """注入自评校准曲线（CLI 从 responses 历史重算后调用）."""
        self.calibration_curves = curves

    # ── 间隔衰减（P3）：历史重放峰值 + 无状态衰减视图 ────────────────

    def peak_mastery_from_history(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """从 responses 历史重放推导每个 skill 的历史峰值掌握概率（只读，不改 l1）.

        BKT 状态不持久化（恢复路径只喂 MIRT），"曾掌握"在重启后无从从 l1 读出
        --对每个 skill 取其作答序列，用独立的临时 BKTModel 逐条 update
        （学习更新，非时间衰减），记录过程中 P(L) 最大值；可从 DB 幂等重算。
        rows 需含 skill_id + correct；缺 skill_id 的行跳过。
        """
        models: Dict[str, BKTModel] = {}
        peaks: Dict[str, float] = {}
        for r in history:
            skill = r.get("skill_id")
            if not skill:
                continue
            if skill not in models:
                models[skill] = BKTModel(skill, self.l1.config.get_params(skill))
            p = models[skill].update(bool(r.get("correct")))
            if p > peaks.get(skill, 0.0):
                peaks[skill] = p
        return peaks

    def decayed_mastery_view(self, history: List[Dict[str, Any]],
                             now: datetime | None = None) -> Dict[str, Tuple[float, float, int]]:
        """无状态衰减视图：{skill: (peak, decayed, days_since_last)}，不改 l1 状态.

        decayed = peak · e^(-days/τ)，直接按公式算、不经 l1.apply_decay 的
        原地乘法--后者若每次会话按"距上次作答天数"调用会在已衰减值上再乘
        全量因子（复合衰减）。days 取该 skill 最近一条 created_at 距今天数；
        本地单用户 CLI，用本地时区 now（测试可注入固定值）。rows 需含
        skill_id / correct / created_at（峰值来自 peak_mastery_from_history）。
        返回值在方案 2-tuple 上扩展了 days（复测文案要显示天数）。
        """
        peaks = self.peak_mastery_from_history(history)
        last: Dict[str, datetime] = {}
        for r in history:
            skill = r.get("skill_id")
            ts = r.get("created_at")
            if not skill or not ts:
                continue
            try:
                t = datetime.fromisoformat(str(ts))
            except ValueError:
                continue
            if skill not in last or t > last[skill]:
                last[skill] = t
        now = now or datetime.now()
        view: Dict[str, Tuple[float, float, int]] = {}
        for skill, peak in peaks.items():
            t = last.get(skill)
            days = max((now - t).days, 0) if t is not None else 0
            if days > 0:
                tau = self.l1.config.get_decay_constant(skill)
                decayed = peak * float(np.exp(-days / tau))
            else:
                decayed = peak
            view[skill] = (peak, decayed, days)
        return view

    def get_theta(self, state: BeliefState) -> np.ndarray:
        """获取当前 5D θ."""
        return state.theta_vector()

    def reset_user(self, user_id: str) -> None:
        """重置某用户的累积历史."""
        self._response_history.pop(user_id, None)
        for skill in self.l1.all_skills():
            self.l1.reset_skill(skill)

    # ── 内部 ────────────────────────────────────────────────────────

    def _calibrated_discount(self, self_confidence: float) -> float:
        """伪自信折扣幅度：校准曲线驱动，数据不足回退固定值.

        expected = 自评所在桶的历史真实答对率（Laplace 平滑）；
        discount = 1 - expected，夹取 [0.05, 0.5]（CALIBRATED_* 注释）。
        曲线缺失或桶样本 < MIN_BUCKET_N 时返回 ILLUSORY_MASTERY_DISCOUNT
        （与迁移前行为一致，黄金回归基线因此不受影响）。
        """
        if self.calibration_curves:
            expected = CalibrationCurveComputer.expected_accuracy(
                self.calibration_curves, self_confidence)
            if expected is not None:
                return float(np.clip(1.0 - expected,
                                     CALIBRATED_DISCOUNT_MIN, CALIBRATED_DISCOUNT_MAX))
        return ILLUSORY_MASTERY_DISCOUNT

    def _detect_misconception(self, observation: Observation) -> Optional[MisconceptionHit]:
        """从解释文本关键词检测 misconception（无 LLM 依赖的轻量路径）."""
        text = (observation.explanation_text or "").strip()
        if not text:
            return None
        entry = self.misconception_library.detect_by_keywords(text)
        if entry is None:
            return None
        # P4：置信度证据驱动（反复持续的误解 > 0.6，被克服的回落）；
        # 无 tracker（未注入，如测试/黄金回归）回退固定 0.6
        confidence = 0.6
        if self.misconception_tracker is not None:
            confidence = self.misconception_tracker.confidence(entry.misc_id)
        return MisconceptionHit(
            misc_id=entry.misc_id,
            confidence=confidence,
            trigger_problem_id=observation.problem_id,
            evidence_text=text,
            timestamp=observation.timestamp,
            correction_strategy=entry.correction_strategy,
        )
