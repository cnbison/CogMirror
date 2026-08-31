"""SQLite 持久层.

表结构按 MIGRATION.md 第2节重新设计（相对 ECOS `ecos/persistence/db.py`）：
- 删除 consent_version 等监护人同意相关字段（新项目面向成年人）
- 替换为成人向标准数据合规字段：数据导出请求、删除请求时间戳
- student_id -> user_id（去掉"学生"隐含身份）
- 删除教师 Q 矩阵审核 / evidence_log / calibration_log / interventions
  等 ECOS 专有表，只保留 MVP 链路需要的 users / responses / belief_snapshots
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .belief_state import BeliefState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    data_export_requested_at TEXT,
    data_delete_requested_at TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    response_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    problem_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    score REAL NOT NULL,
    correct INTEGER NOT NULL,
    bloom_level TEXT NOT NULL,
    self_confidence REAL,
    illusory_flag INTEGER NOT NULL DEFAULT 0,
    user_answer TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_responses_user ON responses(user_id, created_at);

CREATE TABLE IF NOT EXISTS belief_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_user ON belief_snapshots(user_id, created_at);

CREATE TABLE IF NOT EXISTS misconception_evidence (
    misc_id TEXT PRIMARY KEY,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL
);
"""

DEFAULT_DB_PATH = Path("data/cogmirror.db")


class Database:
    """SQLite 持久层（单用户本地工具，无并发诉求）."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # P4：responses 加 misc_id 列（命中记录落库，对账原料）。既有库的表
        # 已存在，CREATE TABLE IF NOT EXISTS 不会加列 -> 启动检查 + ALTER（单
        # 用户本地库可行，方案 5.4 B 路）
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(responses)")}
        if "misc_id" not in cols:
            self._conn.execute("ALTER TABLE responses ADD COLUMN misc_id TEXT")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── users ───────────────────────────────────────────────────────

    def ensure_user(self, user_id: str) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO users (user_id, created_at, last_active_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET last_active_at = ?",
            (user_id, now, now, now),
        )
        self._conn.commit()

    def get_user(self, user_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def request_data_export(self, user_id: str) -> None:
        """数据合规：记录用户的导出请求（PRD 第9节）."""
        self._conn.execute(
            "UPDATE users SET data_export_requested_at = ? WHERE user_id = ?",
            (datetime.now().isoformat(), user_id),
        )
        self._conn.commit()

    def request_data_delete(self, user_id: str) -> None:
        """数据合规：记录用户的删除请求并立即清除该用户全部数据."""
        now = datetime.now().isoformat()
        self._conn.execute("DELETE FROM responses WHERE user_id = ?", (user_id,))
        self._conn.execute("DELETE FROM belief_snapshots WHERE user_id = ?", (user_id,))
        # 证据表无 user_id 列（单用户本地库），"删除全部数据"时一并清空
        self._conn.execute("DELETE FROM misconception_evidence")
        self._conn.execute(
            "UPDATE users SET data_delete_requested_at = ? WHERE user_id = ?",
            (now, user_id),
        )
        self._conn.commit()

    # ── responses ───────────────────────────────────────────────────

    def save_response(self, user_id: str, obs_dict: dict, illusory_flag: bool,
                      misc_id: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO responses (user_id, problem_id, skill_id, score, correct, "
            "bloom_level, self_confidence, illusory_flag, user_answer, misc_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                obs_dict["problem_id"],
                obs_dict["skill_id"],
                float(obs_dict["score"]),
                int(obs_dict["score"] >= 0.6),
                obs_dict["bloom_level"],
                obs_dict.get("self_confidence"),
                int(illusory_flag),
                obs_dict.get("user_answer", ""),
                misc_id,
                obs_dict["timestamp"],
            ),
        )
        self._conn.commit()

    def load_responses(self, user_id: str) -> list[dict]:
        """加载作答历史（BeliefEngine.set_history 的 DB restore 格式）."""
        rows = self._conn.execute(
            "SELECT * FROM responses WHERE user_id = ? ORDER BY response_id", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── misconception evidence（P4）──────────────────────────────────

    def save_misconception_evidence(self, rows: list[dict]) -> None:
        """写入证据行（tracker.dump() 格式），按 misc_id 幂等覆盖."""
        for r in rows:
            self._conn.execute(
                "INSERT INTO misconception_evidence (misc_id, success_count, "
                "failure_count, last_updated) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(misc_id) DO UPDATE SET success_count = ?, "
                "failure_count = ?, last_updated = ?",
                (r["misc_id"], r["success_count"], r["failure_count"], r["last_updated"],
                 r["success_count"], r["failure_count"], r["last_updated"]),
            )
        self._conn.commit()

    def load_misconception_evidence(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM misconception_evidence ORDER BY misc_id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── belief snapshots ────────────────────────────────────────────

    def save_state(self, state: BeliefState) -> None:
        self._conn.execute(
            "INSERT INTO belief_snapshots (user_id, state_json, created_at) VALUES (?, ?, ?)",
            (state.user_id, json.dumps(state.to_dict(), ensure_ascii=False),
             datetime.now().isoformat()),
        )
        self._conn.commit()

    def load_latest_state(self, user_id: str) -> Optional[BeliefState]:
        row = self._conn.execute(
            "SELECT state_json FROM belief_snapshots WHERE user_id = ? "
            "ORDER BY snapshot_id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return BeliefState.from_dict(json.loads(row["state_json"]))

    # ── 导出 ────────────────────────────────────────────────────────

    def export_user_data(self, user_id: str) -> dict[str, Any]:
        """导出该用户全部数据（PRD 第9节"可导出"承诺的实现）."""
        return {
            "user": self.get_user(user_id),
            "responses": self.load_responses(user_id),
            "misconception_evidence": self.load_misconception_evidence(),
        }
