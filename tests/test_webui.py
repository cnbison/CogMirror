"""Web UI 测试（stdlib 后端 API + 前端静态伺服）.

核心逻辑直调（不起 HTTP）+ 一次真实 HTTP 冒烟。判分/引擎链路与 CLI
共享同一套代码，这里只测 web 特有行为：答案剥离、grade/commit 两段式、
组末对账、地图结构化数据、重练误标回归。
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from cogmirror.belief_state import BloomLevel
from cogmirror.db import Database
from cogmirror.webui import DEFAULT_PORT, WebUI, make_handler

_M8_TEXT = "为什么函数里改不了外面的x？"


@pytest.fixture
def web(tmp_path):
    w = WebUI(str(tmp_path / "web.db"), default_user="t1")
    yield w
    w.close()


def _run_quiz(w, user, answers_conf, topic="", level=None):
    """跑一组题：answers_conf = [(answer, self_conf, explanation), ...]."""
    data = w.api_quiz(user, len(answers_conf), topic, level, False)
    out = []
    for q, (answer, conf, explanation) in zip(data["questions"], answers_conf):
        graded = w.api_grade(user, q["problem_id"], answer)
        out.append(w.api_commit(user, explanation, conf))
    return data, out


# ── init / quiz ───────────────────────────────────────────────────


def test_init_new_and_returning(web):
    assert web.api_init("t1") == {"user": "t1", "is_new": True}
    _run_quiz(web, "t1", [("1", 0.8, "")])
    init = web.api_init("t1")
    assert init["is_new"] is False
    assert init["n_responses"] == 1


def test_init_struggles_surface_last_session(web):
    # 第一组答错（variables）-> 第二次进入浮现「上次卡住」
    _run_quiz(web, "t1", [("0", None, ""), ("wrong", None, "")])
    assert "变量赋值" in web.api_init("t1")["struggles"]


def test_quiz_strips_answers(web):
    data = web.api_quiz("t1", 3, "", None, False)
    assert data["count"] == 3
    raw = json.dumps(data)
    for leak in ("correct_answer", "option_explanations", "explanation"):
        assert leak not in raw, f"题面泄漏 {leak}"
    q = data["questions"][0]
    assert q["problem_id"] == "pv-l1-01"
    assert q["qtype"] == "choice" and len(q["options"]) == 4


def test_quiz_topic_level_filter(web):
    data = web.api_quiz("t1", 5, "python.loops", BloomLevel.APPLY, False)
    assert data["count"] > 0
    assert all(q["topic"] == "python.loops" for q in data["questions"])
    assert all(q["bloom_level"] == "APPLY" for q in data["questions"])


def test_quiz_review_mode_all_wrong_questions(web, tmp_path):
    _run_quiz(web, "t1", [("0", None, "")])  # pv-l1-01 答错
    data = web.api_quiz("t1", 0, "", None, True)
    assert [q["problem_id"] for q in data["questions"]] == ["pv-l1-01"]


# ── grade / commit 两段式 ─────────────────────────────────────────


def test_grade_then_commit_flow(web):
    g = web.api_grade("t1", "pv-l1-01", "1")
    assert g["score"] == 1.0 and g["correct"] is True
    assert "option_explanation" in g and "正确" in g["option_explanation"]
    c = web.api_commit("t1", "", 0.8)
    assert c == {"live_feedback": [], "misc_id": None}


def test_grade_code_partial_credit(web):
    g = web.api_grade("t1", "pv-l3-01", "def swap_values(a, b):\n    return (b, a)")
    assert g["score"] == 1.0
    g = web.api_grade("t1", "pv-l2-02", "[1, 2, 3]")
    assert g["score"] == 1.0


def test_grade_code_in_worker_thread():
    # 回归（web 真机发现）：代码题判分的 signal.alarm 只在主线程可用，
    # 请求线程里判分曾抛 "signal only works in main thread" -> 现在
    # 非主线程优雅降级（无超时保护但不崩）
    import threading
    from cogmirror.questions import QuestionBank
    bank = QuestionBank()
    results = {}

    def run():
        q = bank.get("pv-l3-01")
        results["out"] = bank.grade_answer(q, "def swap_values(a, b):\n    return (b, a)")

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=10)
    score, details = results["out"]
    assert score == 1.0, details


def test_commit_explanation_triggers_misc_and_live_feedback(web):
    g = web.api_grade("t1", "pv-l2-01", "0")
    assert g["score"] == 0.0
    c = web.api_commit("t1", _M8_TEXT, 0.9)
    assert c["misc_id"] == "M8"
    assert any("伪自信提示" in line for line in c["live_feedback"])


def test_commit_without_pending_raises(web):
    with pytest.raises(RuntimeError):
        web.api_commit("t1", "", None)


def test_commit_persists_per_question_snapshot(web):
    _run_quiz(web, "t1", [("1", 0.8, "")])
    db = Database(web.db_path)
    try:
        assert len(db.load_responses("t1")) == 1
        assert db.load_latest_state("t1") is not None  # 每题存快照（防关浏览器丢状态）
    finally:
        db.close()


# ── 重练误标回归（web 练习轮自测发现的 bug） ──────────────────────


def test_repractice_no_stale_hit_mislabel(web):
    # 第一组：pv-l2-01 答错 + M8 解释 -> 命中落库
    _run_quiz(web, "t1", [("1", 0.8, ""), ("0", 0.9, _M8_TEXT)])
    db = Database(web.db_path)
    try:
        rows = db.load_responses("t1")
        assert rows[1]["misc_id"] == "M8"
        assert rows[1]["illusory_flag"] == 1  # 自评 0.9 答错，真命中
        # 重练同一题：这次自评低 + 不解释 -> 旧行命中不得误标进新行
        web.api_grade("t1", "pv-l2-01", "1")
        web.api_commit("t1", "", 0.3)
        rows = db.load_responses("t1")
        assert rows[2]["misc_id"] is None
        assert rows[2]["illusory_flag"] == 0
    finally:
        db.close()


# ── finish：对账 + 地图 ───────────────────────────────────────────


def test_finish_reconciles_and_returns_map(web):
    # q1 命中 M8（variables）、q2 同 skill 答对未重触发 -> 检测被证伪 -> failure=1
    _run_quiz(web, "t1", [("0", 0.3, _M8_TEXT), ("2", 0.8, "")])
    payload = web.api_finish("t1")
    db = Database(web.db_path)
    try:
        ev = db.load_misconception_evidence()
        assert ev and ev[0]["misc_id"] == "M8"
        assert ev[0]["failure_count"] == 1 and ev[0]["success_count"] == 0
    finally:
        db.close()
    assert payload["misc_hits"][0]["name"] == "全局/局部作用域混淆"
    assert payload["suggestion"]
    assert "dims" in payload and "bloom" in payload and "tc" in payload


def test_finish_reconcile_no_double_count(web):
    # 两组各含一次 M8 命中 -> 对账窗口按题组推进，第二组 finish 不重复算第一组
    _run_quiz(web, "t1", [("0", 0.3, _M8_TEXT)])
    web.api_finish("t1")
    _run_quiz(web, "t1", [("0", 0.3, _M8_TEXT)])
    web.api_finish("t1")
    db = Database(web.db_path)
    try:
        ev = db.load_misconception_evidence()
        # 第一组命中无同 skill 后续 -> 无证据；第二组命中后无后续 -> 无证据
        assert ev == []
    finally:
        db.close()


def test_map_only_has_no_delta(web):
    _run_quiz(web, "t1", [("1", 0.8, "")])
    web.api_finish("t1")
    m = web.api_map("t1")
    assert "delta_lines" not in m  # map-only 同 CLI：无 prev_state 不出对比段


def test_map_after_quiz_has_delta_and_reflection(web):
    _run_quiz(web, "t1", [("1", 0.8, ""), ("2", 0.8, "")])
    payload = web.api_finish("t1")
    assert payload.get("delta_lines"), "答题后地图应带「与上次相比」"
    assert "本次" in payload["interpretation"] or "作答样本还少" in payload["interpretation"]


# ── HTTP 冒烟（真实服务器线程） ───────────────────────────────────


def test_http_smoke(web):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(web))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/api/init?user=t1") as r:
            assert json.loads(r.read())["user"] == "t1"
        with urllib.request.urlopen(f"{base}/") as r:
            body = r.read().decode()
            assert "CogMirror" in body and "app.js" in body
        for path in ("/app.js", "/styles.css"):
            with urllib.request.urlopen(f"{base}{path}") as r:
                assert r.status == 200 and len(r.read()) > 0
        req = urllib.request.Request(
            f"{base}/api/grade",
            data=json.dumps({"user": "t1", "problem_id": "pv-l1-01", "answer": "1"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["score"] == 1.0
        # 回归（真机 bug）：代码题在请求线程判分曾抛 signal 错误 -> 现在 200 + 正常判分
        req = urllib.request.Request(
            f"{base}/api/grade",
            data=json.dumps({"user": "t1", "problem_id": "pv-l3-01",
                             "answer": "def swap_values(a, b):\n    return (b, a)"}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["score"] == 1.0
        with pytest.raises(Exception):
            urllib.request.urlopen(f"{base}/nonexistent")
    finally:
        server.shutdown()
        server.server_close()


def test_default_port_constant():
    assert DEFAULT_PORT == 8300


# ── 断点续答（真机反馈：刷新后不应从头出题） ──────────────────────


def test_quiz_resume_after_partial_answers(web):
    # 3 题组答 1 题 -> init 报 in_progress；resume 返回剩余 2 题（从第 2 题起）
    data = web.api_quiz("t1", 3, "", None, False)
    q1, q2, q3 = data["questions"]
    web.api_grade("t1", q1["problem_id"], "1")
    web.api_commit("t1", "", 0.8)
    init = web.api_init("t1")
    assert init["quiz_in_progress"] == {"remaining": 2, "total": 3}
    resumed = web.api_quiz_resume("t1")
    assert [q["problem_id"] for q in resumed["questions"]] == [q2["problem_id"], q3["problem_id"]]
    # 判分后未 commit 的题也算未答（刷新发生在判分出错/提交前）
    web.api_grade("t1", q2["problem_id"], "1")
    resumed = web.api_quiz_resume("t1")
    assert [q["problem_id"] for q in resumed["questions"]] == [q2["problem_id"], q3["problem_id"]]


def test_quiz_resume_cleared_after_finish(web):
    _run_quiz(web, "t1", [("1", 0.8, "")])
    assert web.api_finish("t1")  # 题组结束清进度
    assert "quiz_in_progress" not in web.api_init("t1")
    assert web.api_quiz_resume("t1") == {"questions": [], "count": 0}


def test_quiz_new_group_overwrites_progress(web):
    data = web.api_quiz("t1", 3, "", None, False)
    q1 = data["questions"][0]
    web.api_grade("t1", q1["problem_id"], "1")
    web.api_commit("t1", "", 0.8)
    # 主动开新组（覆盖进行中的进度，prev_state 以新组开始为准）；
    # 新组先答 1 题后，续答进度应反映新组（2 题组 remaining=1）
    data2 = web.api_quiz("t1", 2, "python.loops", None, False)
    web.api_grade("t1", data2["questions"][0]["problem_id"], "1")
    web.api_commit("t1", "", 0.8)
    assert web.api_init("t1")["quiz_in_progress"] == {"remaining": 1, "total": 2}


# ── 新题优先（真机反馈：每组都从题库第一道出，已答题反复出现） ────


def test_fresh_quiz_skips_answered(web):
    # 已答 pv-l1-01 -> fresh 新组从 pv-l2-01 起
    _run_quiz(web, "t1", [("1", 0.8, "")])
    data = web.api_quiz("t1", 2, "", None, False, skip_answered=True)
    assert [q["problem_id"] for q in data["questions"]] == ["pv-l2-01", "pv-l2-02"]


def test_fresh_quiz_fills_with_answered_when_exhausted(web):
    # 全部 51 题已答 -> 新题不够，已答题补位（题组不空）
    answers = [("1", 0.8, "")] * 51
    _run_quiz(web, "t1", answers)
    data = web.api_quiz("t1", 3, "", None, False, skip_answered=True)
    assert data["count"] == 3


def test_practice_round_not_skip_answered(web):
    # 练习轮（无 fresh 标志）：重做是巩固语义的一部分，仍从该筛选的首题出
    _run_quiz(web, "t1", [("1", 0.8, "")])
    data = web.api_quiz("t1", 3, "python.variables", None, False, skip_answered=False)
    assert data["questions"][0]["problem_id"] == "pv-l1-01"


def test_zero_progress_group_no_resume_button(web):
    # 零进度的组（刚开未答）不显示「继续」——与开始答题等价，只会造成困惑
    web.api_quiz("t1", 10, "", None, False)
    assert "quiz_in_progress" not in web.api_init("t1")


# ── 语法错误可修正重交（真机反馈：少一个冒号直接判 0 不合理） ────


def test_syntax_error_retryable_payload(web):
    # 语法错误 -> payload 带 syntax_error（行号 + 源码行），供前端就地修正重交
    g = web.api_grade("t1", "pv-l3-01", "def swap_values(a, b)\n    return (b, a)")
    assert g["score"] == 0.0
    assert "第 1 行" in g["syntax_error"]["message"]
    assert g["syntax_error"]["line"] == "def swap_values(a, b)"
    # 正常判分不带该标记
    g = web.api_grade("t1", "pv-l3-01", "def swap_values(a, b):\n    return (b, a)")
    assert "syntax_error" not in g and g["score"] == 1.0


def test_syntax_error_retry_then_commit(web):
    # 修正重交：grade 是纯判分不落库 -> 语法错误重交后 commit 只记录最终作答
    web.api_grade("t1", "pv-l3-01", "def swap_values(a, b)\n    return (b, a)")  # 笔误
    web.api_grade("t1", "pv-l3-01", "def swap_values(a, b):\n    return (b, a)")  # 修正
    c = web.api_commit("t1", "", 0.8)
    assert c["misc_id"] is None
    db = Database(web.db_path)
    try:
        rows = db.load_responses("t1")
        assert len(rows) == 1 and rows[0]["score"] == 1.0
    finally:
        db.close()


def test_syntax_error_giveup_commits_zero(web):
    # 放弃保底：语法错误放弃 -> 计 0 分提交
    web.api_grade("t1", "pv-l3-01", "def swap_values(a, b)\n    return (b, a)")
    web.api_commit("t1", "修不出来", 0.5)
    db = Database(web.db_path)
    try:
        rows = db.load_responses("t1")
        assert rows[0]["score"] == 0.0
    finally:
        db.close()
