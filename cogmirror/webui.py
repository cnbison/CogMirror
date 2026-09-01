"""本地 Web UI（学生端单页，stdlib 零新依赖）.

形态对齐 ECOS `web/student/`（无框架纯 JS 单页 + 静态伺服），服务层用
标准库 http.server（127.0.0.1 单用户本地，无并发诉求但加锁防错）。
产品化 UI（React/CodeMirror/ECharts + npm 构建链）见 SOMEDAY，触发
条件：真人验证正向信号。

API（全部 JSON，复用 CLI 的纯函数，判分/引擎逻辑零重复）：
    GET  /api/init?user=          恢复状态 + 欢迎信息（概览/上次卡点）
    GET  /api/quiz?user=&n=&topic=&level=&review=   取一组题（剥离答案）
    POST /api/grade               {user, problem_id, answer} -> 判分预览
    POST /api/commit              {user, explanation, self_confidence}
                                   -> 引擎更新 + 落库 + 逐题实时反馈
    POST /api/quiz/finish         {user} -> 组末对账 + 完整地图数据 + 建议
    GET  /api/map?user=           只看地图（无 delta，同 --map-only）

流程刻意镜像 cli.run_session：grade（纯判分）-> [答错追问解释] -> commit
（update + save_response）-> finish（reconcile + 证据落库），P4 的
explanation/misconception 链路与 CLI 完全同构。

运行：python -m cogmirror.webui [--user local_user] [--port 8300] [--open]
"""

from __future__ import annotations

import argparse
import copy
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .belief_engine import BeliefEngine, Observation
from .belief_state import BeliefState, BloomLevel
from .calibration import CalibrationCurveComputer
from .cli import (
    DIM_LABELS,
    BLOOM_LABELS,
    BLOOM_LAYER_LABELS,
    _calibration_line,
    _illusory_live_feedback,
    _liminal_live_feedback,
    _map_delta_lines,
    _refresh_decay_view,
    _retest_candidates,
    _tc_display_name,
    _tc_remaining_text,
    _topic_label,
    _welcome_progress,
    _wrong_problem_ids,
    map_interpretation,
    next_suggestion,
    practice_command,
    suggested_practice,
)
from .db import Database
from .misconception_tracker import MisconceptionTracker
from .questions import QuestionBank
from .session import last_session_struggles, multi_session_trend, trend_line

_STATIC_DIR = Path(__file__).parent / "web"
DEFAULT_PORT = 8300


class UserSession:
    """一个用户的答题会话（engine + state + 本组未提交的判分结果）.

    grade -> pending（保存答案与分数）；commit 消费 pending 构造 Observation
    走完整引擎更新与落库。prev_state 在本组开始时深拷贝，供地图 delta。
    """

    def __init__(self, user_id: str, db: Database) -> None:
        self.user_id = user_id
        self.db = db
        self.bank = QuestionBank()
        self.tracker = MisconceptionTracker()
        self.tracker.load(db.load_misconception_evidence())
        self.engine = BeliefEngine(misconception_tracker=self.tracker)
        self.engine.l2.register_items_bulk(self.bank.mirt_items())
        self.pending: Optional[Dict[str, Any]] = None
        self.prev_state: Optional[BeliefState] = None
        # 进行中的题组（服务端保存进度：浏览器刷新/关闭后可「继续答题」，
        # 不再从头出题——真机反馈）。quiz_pos = 下一道未答题的下标
        self.quiz_questions: List[Any] = []
        self.quiz_pos: int = 0
        self._n_rows_before = len(db.load_responses(user_id))
        state = db.load_latest_state(user_id)
        self._state_existed = state is not None
        self.state = state or self.engine.create_initial_state(user_id)
        self._restore()

    @property
    def is_new(self) -> bool:
        # 动态判断（服务器长驻，用户答题后就不是新用户了）
        return not (self._state_existed or self.state.trajectory.snapshots)

    def _restore(self) -> None:
        """从 DB 重建派生视图（镜像 cli.main 的恢复路径）."""
        history = [
            {"problem_id": r["problem_id"], "correct": r["correct"], "score": r["score"],
             "bloom_level": r["bloom_level"], "self_confidence": r["self_confidence"]}
            for r in self.db.load_responses(self.user_id)
        ]
        self.engine.set_history(self.user_id, history)
        self.engine.set_calibration(CalibrationCurveComputer().compute(history))
        _refresh_decay_view(self.engine, self.db, self.user_id)

    # ── 题组 ────────────────────────────────────────────────────────

    def select_questions(self, n: int, topic: str, level: Optional[BloomLevel],
                         review: bool, skip_answered: bool = False) -> List[Dict[str, Any]]:
        """选题并深拷贝快照 prev_state.

        skip_answered=True（web「开始答题」）：新题优先--已作答过的题排到
        组尾，只在剩余新题不足时才补位（真机反馈：每组都从题库第一道
        出，已答过的题反复出现）。练习轮/错题重练不传该标志（重做是巩固
        语义的一部分）。
        """
        selected = self.bank.all_questions()
        problem_ids = None
        if review:
            problem_ids = _wrong_problem_ids(self.db, self.user_id)
            wanted = set(problem_ids)
            selected = [q for q in selected if q.problem_id in wanted]
        if topic:
            selected = [q for q in selected if q.topic == topic]
        if level is not None:
            selected = [q for q in selected if q.bloom_level == level]
        if skip_answered and not review:
            answered = {r["problem_id"] for r in self.db.load_responses(self.user_id)}
            fresh = [q for q in selected if q.problem_id not in answered]
            redo = [q for q in selected if q.problem_id in answered]
            selected = fresh + redo
        # n <= 0 = 不限数量（错题重练模式前端传 0，重练全部错题）
        questions = selected[:n] if n > 0 else selected
        self.prev_state = copy.deepcopy(self.state) if questions else self.prev_state
        self.quiz_questions = list(questions)
        self.quiz_pos = 0
        return self._strip_questions(questions)

    def resume_questions(self) -> List[Dict[str, Any]]:
        """进行中题组的剩余题（无进行中题组返回空）."""
        if not self.quiz_questions or self.quiz_pos >= len(self.quiz_questions):
            return []
        return self._strip_questions(self.quiz_questions[self.quiz_pos:])

    @staticmethod
    def _strip_questions(questions) -> List[Dict[str, Any]]:
        # 题面剥离答案与讲解（判分在服务端，讲解答完才返回）
        return [
            {
                "problem_id": q.problem_id,
                "qtype": q.qtype,
                "topic": q.topic,
                "topic_label": _topic_label(q.topic),
                "bloom_level": q.bloom_level.name,
                "prompt": q.prompt,
                "options": list(q.options) if q.qtype == "choice" else [],
            }
            for q in questions
        ]

    # ── 单题：grade（纯判分） -> commit（更新 + 落库） ─────────────

    def grade(self, problem_id: str, answer: str) -> Dict[str, Any]:
        q = self.bank.get(problem_id)
        if q is None:
            raise KeyError(f"题目不存在: {problem_id}")
        score, details = self.bank.grade_answer(q, answer)
        self.pending = {"question": q, "answer": answer, "score": score,
                        "details": details}
        payload: Dict[str, Any] = {"problem_id": problem_id, "score": score,
                                   "correct": score >= 0.6, "details": details}
        if q.qtype == "choice" and q.option_explanations:
            try:
                chosen = int(answer.strip())
            except ValueError:
                chosen = -1
            if 0 <= chosen < len(q.option_explanations):
                payload["option_explanation"] = q.option_explanations[chosen]
        elif q.explanation:
            payload["key_point"] = q.explanation
        return payload

    def commit(self, explanation: str, self_confidence: Optional[float]) -> Dict[str, Any]:
        """消费 pending：构造 Observation 走引擎更新 + 落库（镜像 run_session 单题流程）."""
        if self.pending is None:
            raise RuntimeError("没有待提交的判分结果（先 POST /api/grade）")
        q = self.pending["question"]
        answer = self.pending["answer"]
        score = self.pending["score"]
        self.pending = None

        tc_before = self.state.C.tc_states.get(q.skill_id)
        prev_tc_status = tc_before.status if tc_before else None
        illusory_before = len(self.state.C.illusory_confidence_hits)
        misc_before = len(self.state.C.misconception_hits)

        obs = Observation(
            skill_id=q.skill_id, problem_id=q.problem_id, score=score,
            bloom_level=q.bloom_level, self_confidence=self_confidence,
            user_answer=answer, explanation_text=explanation,
        )
        self.state = self.engine.update(self.state, obs)
        # 本题是否新增命中：用命中数增量判断（引擎每次 update 最多追加一条）。
        # 不能用「最后一条命中的题号 == 本题」--重练曾命中的题且本次未命中时，
        # 旧行命中会被误标进新行（练习轮自测发现，CLI 同款 bug 一并修）
        misc_id = None
        if len(self.state.C.misconception_hits) > misc_before:
            misc_id = self.state.C.misconception_hits[-1].misc_id
        self.db.save_response(
            self.user_id, obs.to_dict(),
            illusory_flag=len(self.state.C.illusory_confidence_hits) > illusory_before,
            misc_id=misc_id)
        # 每题存快照（CLI 是每题组存一次）：浏览器随时可能关掉，逐题落库
        # 让状态与 responses 不脱节
        self.db.save_state(self.state)

        live: List[str] = []
        line = _liminal_live_feedback(self.engine, self.state, q.skill_id, score,
                                      q.bloom_level, prev_tc_status)
        if line:
            live.append(line)
        line = _illusory_live_feedback(self.state, illusory_before)
        if line:
            live.append(line)
        self.quiz_pos += 1  # 断点续答：已答一题（真机反馈，刷新后不再从头出题）
        return {"live_feedback": live, "misc_id": misc_id}

    # ── 组末：对账 + 地图 ─────────────────────────────────────────

    def finish(self) -> Dict[str, Any]:
        # 先出地图（此时本组行仍在 session_rows 里，delta/反思句基于本组），
        # 再对账并推进窗口
        payload = self.map_payload()
        self.quiz_questions = []
        self.quiz_pos = 0
        rows = self.db.load_responses(self.user_id)
        self.tracker.reconcile(rows[self._n_rows_before:])
        # 对账窗口 = 一个题组（web 会话边界），对账过的行不再参与下次
        # 对账（练习轮 finish 时防重复计数）
        self._n_rows_before = len(rows)
        self.db.save_misconception_evidence(self.tracker.dump())
        _refresh_decay_view(self.engine, self.db, self.user_id)
        return payload

    def map_payload(self) -> Dict[str, Any]:
        """认知地图的结构化数据（前端渲染；纯函数复用自 cli）."""
        state = self.state
        engine = self.engine
        payload: Dict[str, Any] = {}

        session_rows = None
        if self.prev_state is not None:
            rows = self.db.load_responses(self.user_id)
            session_rows = rows[self._n_rows_before:]
            if not session_rows:
                # 本次会话没有新作答（如 map-only / 已 finish 后再看地图）：
                # 无"与上次相比"基准，同 CLI --map-only 语义
                self.prev_state = None
        payload["interpretation"] = map_interpretation(
            engine, state, prev_state=self.prev_state, session_rows=session_rows)

        dims = []
        for dim, label in DIM_LABELS.items():
            d = getattr(state, dim)
            entry = {"dim": dim, "label": label, "mastery": None, "note": None}
            if dim == "X":
                entry["note"] = "MVP 未提供支架/提示机制，暂未测量"
            elif dim == "C":
                if state.C.illusory_confidence_hits:
                    entry["mastery"] = d.mastery_prob
                    entry["note"] = f"发现 {len(state.C.illusory_confidence_hits)} 处失准，见下方"
                elif any(h.get("self_confidence") is not None
                         for h in engine.get_history(self.user_id)):
                    entry["note"] = "未发现失准（自评与表现一致）"
                else:
                    entry["note"] = "暂无自评数据，暂未测量"
            else:
                entry["mastery"] = d.mastery_prob
            dims.append(entry)
        payload["dims"] = dims

        calibration = _calibration_line(engine)
        if calibration:
            payload["calibration"] = calibration

        if self.prev_state is not None:
            payload["delta_lines"] = _map_delta_lines(engine, state, self.prev_state)

        payload["bloom"] = [
            {"label": label,
             "value": getattr(state.bloom_profile, field),
             "covered": BloomLevel[field.upper()] in state.bloom_profile.covered_layers}
            for field, label in BLOOM_LABELS
        ]
        payload["bloom_dominant"] = (
            BLOOM_LAYER_LABELS[state.bloom_profile.dominant_layer]
            if state.bloom_profile.covered_layers else None)

        payload["illusory_hits"] = [
            {"problem_id": h.problem_id, "self_confidence": h.self_confidence,
             "score": h.score, "gap": h.gap}
            for h in state.C.illusory_confidence_hits
        ]

        payload["misc_hits"] = []
        for h in state.C.misconception_hits:
            entry = engine.misconception_library.get(h.misc_id)
            payload["misc_hits"].append({
                "problem_id": h.trigger_problem_id,
                "name": entry.name if entry else h.misc_id,
                "confidence": h.confidence,
                "evidence_text": h.evidence_text,
            })

        liminal = [(tid, tc) for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
        crossed = [tid for tid, tc in state.C.tc_states.items() if tc.status == "post_liminal"]
        payload["tc"] = {
            "liminal": [
                {"name": _tc_display_name(engine, tid), "progress": tc.progress,
                 "remaining": _tc_remaining_text(engine, tc)}
                for tid, tc in liminal
            ],
            "crossed": [_tc_display_name(engine, tid) for tid in crossed],
        }

        payload["retest"] = [
            {"skill": _topic_label(skill), "days": days, "peak": peak, "decayed": decayed}
            for skill, peak, decayed, days in _retest_candidates(engine)
        ]

        trend = multi_session_trend(self.db, self.user_id)
        line = trend_line(trend)
        if line:
            payload["trend"] = {
                "line": line,
                "n_sessions": next(iter(trend.values()))[2] if trend else 0,
            }

        payload["suggestion"] = next_suggestion(engine, state)
        payload["practice_command"] = practice_command(engine, state)
        target = suggested_practice(engine, state)
        if target is not None:
            topic, level = target
            payload["suggested_practice"] = {
                "topic": topic, "topic_label": _topic_label(topic),
                "level": f"L{level.value}" if level is not None else None,
                "n": 3,
            }
        return payload


class WebUI:
    """HTTP 层：路由 + 用户会话池（本地单用户，全局锁防并发错写）."""

    def __init__(self, db_path: str | Path = "data/cogmirror.db",
                 default_user: str = "local_user") -> None:
        self.db_path = str(db_path)
        self.default_user = default_user
        self._sessions: Dict[str, UserSession] = {}
        # RLock：api_* 方法持锁调用 session()（内部也取锁）
        self._lock = threading.RLock()
        self._db = Database(self.db_path)

    def close(self) -> None:
        self._db.close()

    def session(self, user_id: str) -> UserSession:
        with self._lock:
            s = self._sessions.get(user_id)
            if s is None:
                self._db.ensure_user(user_id)
                s = UserSession(user_id, self._db)
                self._sessions[user_id] = s
            return s

    # ── API 实现（与 HTTP 解析/序列化解耦，测试可直接调用） ───────

    def api_init(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            s = self.session(user_id)
            out: Dict[str, Any] = {"user": user_id, "is_new": s.is_new}
            if s.is_new:
                return out
            overview = _welcome_progress(s.engine, s.state)
            struggles = last_session_struggles(s.db, user_id)
            out["overview"] = overview
            out["struggles"] = [_topic_label(x) for x in struggles[:3]]
            out["n_responses"] = len(s.engine.get_history(user_id))
            # 零进度的组不显示「继续」（与「开始答题」等价，只会造成困惑）
            remaining = len(s.quiz_questions) - s.quiz_pos
            if remaining > 0 and s.quiz_pos > 0:
                out["quiz_in_progress"] = {
                    "remaining": remaining,
                    "total": len(s.quiz_questions),
                }
            return out

    def api_quiz(self, user_id: str, n: int, topic: str,
                 level: Optional[BloomLevel], review: bool,
                 skip_answered: bool = False) -> Dict[str, Any]:
        with self._lock:
            s = self.session(user_id)
            questions = s.select_questions(n, topic, level, review, skip_answered)
            return {"questions": questions, "count": len(questions)}

    def api_quiz_resume(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            s = self.session(user_id)
            questions = s.resume_questions()
            return {"questions": questions, "count": len(questions)}

    def api_grade(self, user_id: str, problem_id: str, answer: str) -> Dict[str, Any]:
        with self._lock:
            s = self.session(user_id)
            return s.grade(problem_id, answer)

    def api_commit(self, user_id: str, explanation: str,
                   self_confidence: Optional[float]) -> Dict[str, Any]:
        with self._lock:
            s = self.session(user_id)
            return s.commit(explanation, self_confidence)

    def api_finish(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            s = self.session(user_id)
            return s.finish()

    def api_map(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            s = self.session(user_id)
            return s.map_payload()


def _parse_level(raw: str) -> Optional[BloomLevel]:
    from .cli import _LEVEL_NAMES
    if not raw:
        return None
    return _LEVEL_NAMES.get(raw.strip().upper())


def _opt_float(raw) -> Optional[float]:
    if raw is None or raw == "":
        return None
    return float(raw)


def make_handler(webui: WebUI) -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 静默默认访问日志
            pass

        def _send_json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            body = path.read_bytes()
            ctype = "text/html; charset=utf-8" if path.suffix == ".html" else (
                "text/css; charset=utf-8" if path.suffix == ".css" else
                "application/javascript; charset=utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send_static(_STATIC_DIR / "index.html")
                return
            if parsed.path in ("/app.js", "/styles.css"):
                self._send_static(_STATIC_DIR / parsed.path.lstrip("/"))
                return
            if parsed.path == "/api/init":
                qs = parse_qs(parsed.query)
                user = qs.get("user", [webui.default_user])[0]
                self._send_json(webui.api_init(user))
                return
            if parsed.path == "/api/quiz":
                qs = parse_qs(parsed.query)
                user = qs.get("user", [webui.default_user])[0]
                n = int(qs.get("n", ["3"])[0])
                topic = qs.get("topic", [""])[0]
                level = _parse_level(qs.get("level", [""])[0])
                review = qs.get("review", [""])[0] in ("1", "true")
                skip_answered = qs.get("fresh", [""])[0] in ("1", "true")
                self._send_json(webui.api_quiz(user, n, topic, level, review, skip_answered))
                return
            if parsed.path == "/api/quiz/resume":
                qs = parse_qs(parsed.query)
                user = qs.get("user", [webui.default_user])[0]
                self._send_json(webui.api_quiz_resume(user))
                return
            if parsed.path == "/api/map":
                qs = parse_qs(parsed.query)
                user = qs.get("user", [webui.default_user])[0]
                self._send_json(webui.api_map(user))
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                user = body.get("user") or webui.default_user
                if parsed.path == "/api/grade":
                    self._send_json(webui.api_grade(
                        user, str(body.get("problem_id", "")), str(body.get("answer", ""))))
                elif parsed.path == "/api/commit":
                    self._send_json(webui.api_commit(
                        user, str(body.get("explanation", "")),
                        _opt_float(body.get("self_confidence"))))
                elif parsed.path == "/api/quiz/finish":
                    self._send_json(webui.api_finish(user))
                else:
                    self.send_error(404)
            except (KeyError, RuntimeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, 400)

    return Handler


def serve(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cogmirror-webui", description="CogMirror 本地 Web UI（学生端单页）")
    parser.add_argument("--user", default="local_user", help="用户 ID（本地单用户默认 local_user）")
    parser.add_argument("--db", default="data/cogmirror.db", help="SQLite 路径")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="端口（默认 8300）")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args(argv)

    webui = WebUI(args.db, default_user=args.user)
    # 单线程 HTTPServer（非 Threading）：请求在主线程处理，代码题判分的
    # signal.alarm 超时保护得以生效；单用户本地场景请求本就被锁串行化，
    # 多线程无收益（真机发现：请求线程里 signal.alarm 直接抛 ValueError）
    server = HTTPServer(("127.0.0.1", args.port), make_handler(webui))
    url = f"http://127.0.0.1:{args.port}"
    print(f"CogMirror Web UI 已启动：{url}（Ctrl-C 退出，数据在 {args.db}）")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        server.server_close()
        webui.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
