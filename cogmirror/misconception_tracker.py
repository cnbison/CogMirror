"""misconception 证据闭环（方案 P4，A2 迁移自 PersonalAGI learning/procedural）.

把 misconception 检测从固定置信度 0.6 升级为证据驱动权重：
- 每条 misconception 的成功/失败计数随对账结果更新；
- 权重 = Laplace 置信度 (s+1)/(s+f+2)；
- 反复持续的误解权重上升（更被重视），被克服的回落。

计数语义说明（与方案 5.4 注释的差异，按 5.1/5.6 可证伪标准取舍）：
success/failure 记的是**「该 misconception 检测作为预测」的对账结果**
（PredictionReconciler 模式：检测命中 = 预测「该学习者有此误解」）：

- record_success：检测被证实 -- 命中后同 skill 的下一条响应仍错或重触发，
  预测成立，证据支持"这个检测模式对该学习者可靠"；
- record_failure：检测被证伪 -- 命中后同 skill 答对且未重触发（已克服/误报）。

方案 5.4 注释按学习者视角标注 success/failure（答对=success），但那样
Laplace 置信度会让"反复失败"的误解权重 < 0.6，与 5.1/5.6 验收标准
（"3 次失败 -> 权重 > 0.6、C 折扣更深；被克服后回落"）方向相反；
两处冲突时取可证伪的验收语义。

无主动"过时"判定（无 TTL/时间衰减）：靠失败降档隐式淘汰（源模式同款），
quarantined() 仅作查询（conf < 0.3 且 s+f >= 3），产品路径暂不消费。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

# quarantine 判定阈值（源模式：conf < 0.3 且 s+f >= 3）
QUARANTINE_CONF_MAX = 0.3
QUARANTINE_MIN_EVIDENCE = 3


class MisconceptionTracker:
    """misconception 检测证据追踪器（单用户，内存态 + DB 持久化）."""

    def __init__(self) -> None:
        self._evidence: Dict[str, Dict[str, Any]] = {}

    # ── 持久化 ──────────────────────────────────────────────────────

    def load(self, rows: List[Dict[str, Any]]) -> None:
        """从 DB 恢复（db.load_misconception_evidence 的行格式）."""
        for r in rows:
            self._evidence[r["misc_id"]] = {
                "success": int(r.get("success_count", 0)),
                "failure": int(r.get("failure_count", 0)),
                "last_updated": r.get("last_updated", ""),
            }

    def dump(self) -> List[Dict[str, Any]]:
        """导出为 DB 行格式（db.save_misconception_evidence 的输入）."""
        return [
            {
                "misc_id": misc_id,
                "success_count": e["success"],
                "failure_count": e["failure"],
                "last_updated": e["last_updated"],
            }
            for misc_id, e in sorted(self._evidence.items())
        ]

    # ── 查询 ────────────────────────────────────────────────────────

    def confidence(self, misc_id: str) -> float:
        """Laplace 置信度 (s+1)/(s+f+2)；无证据 = 0.5（先验）."""
        e = self._evidence.get(misc_id)
        if e is None:
            return 0.5
        return (e["success"] + 1) / (e["success"] + e["failure"] + 2)

    def quarantined(self, misc_id: str) -> bool:
        """检测长期被证伪 -> 该关键词模式对这个学习者不可靠."""
        e = self._evidence.get(misc_id)
        if e is None:
            return False
        total = e["success"] + e["failure"]
        return total >= QUARANTINE_MIN_EVIDENCE and self.confidence(misc_id) < QUARANTINE_CONF_MAX

    def evidence(self, misc_id: str) -> Optional[Dict[str, Any]]:
        return self._evidence.get(misc_id)

    # ── 更新 ────────────────────────────────────────────────────────

    def record_success(self, misc_id: str, now: Optional[datetime] = None) -> None:
        """检测被证实（误解持续）-- 证据计数 +1."""
        self._bump(misc_id, "success", now)

    def record_failure(self, misc_id: str, now: Optional[datetime] = None) -> None:
        """检测被证伪（已克服/误报）-- 证据计数 +1."""
        self._bump(misc_id, "failure", now)

    def reconcile(self, history: List[Dict[str, Any]]) -> None:
        """对账：把 misconception 命中 join 到同 skill 的后续表现（零 LLM）.

        history = 本次会话新增的响应行（DB load_responses 格式），需含
        skill_id / misc_id / score，按时间升序。对每条带 misc_id 的行，
        找同 skill 的下一条响应：

        - 重触发同误解 或 score < 0.6 -> 检测被证实 -> record_success
        - score >= 0.6 且未重触发 -> 检测被证伪 -> record_failure

        后面没有同 skill 响应 -> 本次无证据，不更新。对账窗口限同一会话
        （方案 5.7：跨会话的下一条响应可能隔很久，语义不成立）--所以入参
        只传本次会话的行，不要传全量历史（否则旧命中会重复计数）。
        """
        for i, r in enumerate(history):
            misc_id = r.get("misc_id")
            if not misc_id:
                continue
            for r2 in history[i + 1:]:
                if r2.get("skill_id") != r.get("skill_id"):
                    continue
                retriggered = r2.get("misc_id") == misc_id
                score2 = float(r2.get("score") or 0.0)
                if retriggered or score2 < 0.6:
                    self.record_success(misc_id)
                else:
                    self.record_failure(misc_id)
                break

    def _bump(self, misc_id: str, key: str, now: Optional[datetime]) -> None:
        e = self._evidence.setdefault(
            misc_id, {"success": 0, "failure": 0, "last_updated": ""})
        e[key] += 1
        e["last_updated"] = (now or datetime.now()).isoformat()
