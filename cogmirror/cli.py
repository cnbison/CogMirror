"""Phase 0 最小链路 CLI：做题 -> 5D 状态更新 -> 认知地图展示.

不追求界面美观，追求链路真实可用（ROADMAP Phase 0）。
运行：python -m cogmirror.cli  或安装后 cogmirror
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

from .belief_engine import BeliefEngine, Observation
from .belief_state import BeliefState, BloomLevel
from .calibration import CalibrationCurveComputer, compute_ece
from .db import Database, DEFAULT_DB_PATH
from .misconception_tracker import MisconceptionTracker
from .questions import QuestionBank
from .session import last_session_struggles, multi_session_trend, trend_line

DIM_LABELS = {
    "K": "知识（概念记得住吗）",
    "P": "程序技能（能写出来吗）",
    "S": "策略（会诊断、会分析吗）",
    "C": "置信度（自我评估准不准）",
    "X": "外部支架（借助提示的程度）",
}

BLOOM_LABELS = [
    ("remember", "L1 记忆"),
    ("understand", "L2 理解"),
    ("apply", "L3 应用"),
    ("analyze", "L4 分析"),
    ("evaluate", "L5 评价"),
    ("create", "L6 创造"),
]

# 主导层级显示用中文层名（英文枚举名如 APPLY 对非开发者不友好）
BLOOM_LAYER_LABELS = {BloomLevel[field.upper()]: label for field, label in BLOOM_LABELS}

# topic 中文短名（一句话建议用）。题库 5 个 topic 静态不变，与
# content/threshold_concepts.py / tc.DEFAULT_TC_LIBRARY 的 key 对应。
TOPIC_LABELS = {
    "python.variables": "变量赋值",
    "python.loops": "循环",
    "python.functions": "函数",
    "python.recursion": "递归",
    "python.scope": "作用域",
}


def _topic_label(skill_id: str) -> str:
    """topic 中文短名（建议文案用），未知 id 原样返回."""
    return TOPIC_LABELS.get(skill_id, skill_id)


# --level 参数接受 L1-L6 或 Bloom 层名（大小写不敏感）
_LEVEL_NAMES = {
    "L1": BloomLevel.REMEMBER, "REMEMBER": BloomLevel.REMEMBER,
    "L2": BloomLevel.UNDERSTAND, "UNDERSTAND": BloomLevel.UNDERSTAND,
    "L3": BloomLevel.APPLY, "APPLY": BloomLevel.APPLY,
    "L4": BloomLevel.ANALYZE, "ANALYZE": BloomLevel.ANALYZE,
    "L5": BloomLevel.EVALUATE, "EVALUATE": BloomLevel.EVALUATE,
    "L6": BloomLevel.CREATE, "CREATE": BloomLevel.CREATE,
}


def _parse_level(raw: str) -> BloomLevel:
    try:
        return _LEVEL_NAMES[raw.strip().upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"无效层级: {raw}（接受 L1-L6 或 Bloom 层名，如 L3 / APPLY）")


def _use_color() -> bool:
    """仅交互终端用 ANSI 配色；管道/测试（非 tty）保持纯文本."""
    return sys.stdout.isatty()


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    bar = "█" * filled + "░" * (width - filled)
    num = f"{value:.0%}"
    if not _use_color():
        return f"{bar} {num}"
    # 掌握概率档位配色：>=80% 绿（良好）/ >=60% 黄（中间）/ 其它 红（薄弱）
    code = "32" if value >= 0.8 else ("33" if value >= 0.6 else "31")
    return f"\033[{code}m{bar} {num}\033[0m"


def _tc_display_name(engine: BeliefEngine, tid: str) -> str:
    """TC 显示名单一来源：状态机库（避免第二套库文案漂移）."""
    return engine.tc_detector.tc_library.get(tid, {}).get("name", tid)


def _tc_remaining_text(engine: BeliefEngine, tc) -> str:
    """liminal 态下距跨越的剩余次数文案（把"跨越进度"翻译成可行动步骤）."""
    streak = sum(1 for s in tc.liminal_signals if s == "post_liminal_candidate")
    remaining = max(0, engine.tc_detector.post_liminal_streak - streak)
    return "即将跨越" if remaining <= 0 else f"再答对 {remaining} 次 L3+ 题即跨越"


def _liminal_live_feedback(engine: BeliefEngine, state: BeliefState,
                           skill_id: str, score: float,
                           bloom_level: BloomLevel, prev_status: str | None) -> str:
    """逐题临界概念反馈：liminal 相关的瞬间报一行进度/跨越/回落，其余静默.

    在 run_session 每题 update 后调用，把"临界概念"从地图标注变成做题当下
    能感受到的进度（prev_status 为该题 update 前的 TC 状态，用于识别刚跨越）。
    """
    tc = state.C.tc_states.get(skill_id)
    if tc is None:
        return ""
    name = _tc_display_name(engine, skill_id)
    if prev_status != "post_liminal" and tc.status == "post_liminal":
        return f"「{name}」已跨越！恭喜，这个概念你已经真正掌握。"
    if tc.status == "liminal":
        if score >= 0.6:
            # 只有 L3+ 答对才推进 liminal；L1/L2 答对不推进，静默避免误报进度
            if bloom_level.value >= BloomLevel.APPLY.value:
                return f"「{name}」跨越进度 {tc.progress:.0%}——{_tc_remaining_text(engine, tc)}"
            return ""
        return (f"「{name}」这次答错，跨越进度回落到 {tc.progress:.0%}"
                f"——这不是退步，重来一组即可。")
    return ""


def _illusory_live_feedback(state: BeliefState, illusory_before: int) -> str:
    """伪自信逐题提示：本题新命中伪自信时当题点出（自评 vs 实际落差），否则空串.

    引擎 update 每次最多追加一条伪自信命中，用命中数是否增加判断本题是否新命中。
    """
    hits = state.C.illusory_confidence_hits
    if len(hits) <= illusory_before:
        return ""
    h = hits[-1]
    return (f"伪自信提示：你自评 {h.self_confidence:.0%}，但这题实际得分 {h.score:.0%}"
            f"——落差有点大，『感觉会』可能掩盖了『其实还没会』，地图会把这个点标出来。")


def _welcome_progress(engine: BeliefEngine, state: BeliefState) -> str:
    """返回用户的进度概览一行（上次主导层级 + 临界概念跨越进度），无则空串."""
    parts = []
    if state.bloom_profile.covered_layers:
        parts.append(f"上次主导层级：{BLOOM_LAYER_LABELS[state.bloom_profile.dominant_layer]}")
    liminal = [(tid, tc) for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    if liminal:
        detail = "；".join(
            f"「{_tc_display_name(engine, tid)}」{_tc_remaining_text(engine, tc)}"
            for tid, tc in liminal)
        parts.append(f"{len(liminal)} 个临界概念跨越中（{detail}）")
    return "；".join(parts)


def _map_delta_lines(engine: BeliefEngine, state: BeliefState,
                     prev_state: BeliefState) -> list[str]:
    """「与上次相比」段的行：K/P/S 维度变化 + C 双侧可测时变化 + 主导层级 + 新跨越临界概念."""
    lines = []
    for dim, name in (("K", "知识"), ("P", "程序技能"), ("S", "策略")):
        cur = getattr(state, dim).mastery_prob
        prev = getattr(prev_state, dim).mastery_prob
        d = cur - prev
        if abs(d) < 0.005:
            continue
        sign = "+" if d > 0 else ""
        lines.append(f"{dim} {name}: {sign}{d:.0%}（{prev:.0%} → {cur:.0%}）")
    # C 只有两侧都有实测（命中过伪自信）才给数值对比；X MVP 不测，跳过
    if state.C.illusory_confidence_hits and prev_state.C.illusory_confidence_hits:
        d = state.C.mastery_prob - prev_state.C.mastery_prob
        if abs(d) >= 0.005:
            sign = "+" if d > 0 else ""
            lines.append(f"C 置信度: {sign}{d:.0%}（{prev_state.C.mastery_prob:.0%} → {state.C.mastery_prob:.0%}）")
    cur_dom = state.bloom_profile
    prev_dom = prev_state.bloom_profile
    if cur_dom.covered_layers:
        if not prev_dom.covered_layers:
            lines.append(f"主导层级首次确定：{BLOOM_LAYER_LABELS[cur_dom.dominant_layer]}")
        elif cur_dom.dominant_layer != prev_dom.dominant_layer:
            lines.append(f"主导层级：{BLOOM_LAYER_LABELS[prev_dom.dominant_layer]} → "
                         f"{BLOOM_LAYER_LABELS[cur_dom.dominant_layer]}")
    prev_crossed = {tid for tid, tc in prev_state.C.tc_states.items() if tc.status == "post_liminal"}
    cur_crossed = {tid for tid, tc in state.C.tc_states.items() if tc.status == "post_liminal"}
    new_crossed = sorted(cur_crossed - prev_crossed)
    if new_crossed:
        names = "、".join(_tc_display_name(engine, tid) for tid in new_crossed)
        lines.append(f"新跨越的临界概念：{names}")
    return lines


def _print_welcome() -> None:
    """首次运行的上手说明：让第一次用的人不靠文档也能跑完一组题."""
    print()
    print("第一次用？很简单：")
    print("  1) 每道题先自评把握（0-100，直接回车跳过），再作答")
    print("  2) 选择题输选项编号；填空题直接输答案；写码题写代码，写完单独一行输入 END")
    print("  3) 答完本组会画出你的认知地图（5D 状态 / Bloom 六层 / 伪自信 / 临界概念 / 一句话建议）")
    print("  4) 中途想退出用 Ctrl-C；已答的题都会保存，下次接着看地图")
    print("  查看全部命令：cogmirror --help；导出/删除你的数据：cogmirror --export / --delete")
    print()


def ask_self_confidence() -> float | None:
    while True:
        # 末尾 \n：跳过自评（直接回车）时下一行提示另起一行，避免与选项粘连
        raw = input("答题前自评：你有多大把握答对？（0-100，直接回车跳过）\n").strip()
        if raw == "":
            return None
        try:
            v = int(raw)
        except ValueError:
            print("请输入 0-100 的整数")
            continue
        if 0 <= v <= 100:
            return v / 100.0
        print("请输入 0-100 的整数")


def ask_explanation() -> str:
    """答错后的可选追问：一句话解释「为什么这么答」，回车跳过（跳过是常态）.

    输入流结束（EOF）也视为跳过：追问在判分之后、下一题之前，若在这里因
    EOF 抛出会提前中断整组答题（自测发现交互流测试即此场景）。
    """
    try:
        return input("为什么这么答？用一句话说说你的理由（直接回车跳过）\n").strip()
    except EOFError:
        return ""


def read_answer(question) -> str:
    if question.qtype == "choice":
        for i, opt in enumerate(question.options):
            print(f"  {i}. {opt}")
        n = len(question.options)
        while True:
            raw = input(f"输入选项编号（0-{n - 1}）: ").strip()
            if raw.isdigit() and int(raw) in range(n):
                return raw
            print(f"请输入 0-{n - 1} 之间的选项编号。")
    if question.qtype == "fill":
        return input("输入你的答案（直接回车视为未答）: ")
    print("（写代码：定义题目要求的函数，写完单独一行输入 END 结束）")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _wrong_problem_ids(db: Database, user_id: str) -> list[str]:
    """错题集：每个 problem_id 最近一次得分 < 0.6 的题.

    load_responses 按 response_id 升序返回，顺序覆盖即可取到每题最新一次得分。
    """
    latest: dict[str, float] = {}
    for r in db.load_responses(user_id):
        latest[r["problem_id"]] = r["score"]
    return [pid for pid, score in latest.items() if score < 0.6]


def run_session(engine: BeliefEngine, bank: QuestionBank, state: BeliefState,
                db: Database, n_questions: int, topic: str = "",
                level: BloomLevel | None = None,
                problem_ids: list[str] | None = None) -> BeliefState:
    selected = bank.all_questions()
    if problem_ids is not None:
        wanted = set(problem_ids)
        selected = [q for q in selected if q.problem_id in wanted]
    if topic:
        selected = [q for q in selected if q.topic == topic]
    if level is not None:
        selected = [q for q in selected if q.bloom_level == level]
    questions = selected[:n_questions]
    if not questions:
        print("\n当前筛选条件下没有题目，直接展示认知地图。\n")
        return state
    print(f"\n本组共 {len(questions)} 道题。\n")
    for i, q in enumerate(questions, 1):
        print(f"── 题 {i}/{len(questions)} [{q.problem_id}] "
              f"({q.topic} / {q.bloom_level.name}) ──")
        print(q.prompt)
        try:
            self_conf = ask_self_confidence()
            answer = read_answer(q)
        except EOFError:
            # 输入流结束（Ctrl-D / 脚本输入耗尽）——提前结束，已答的题都保存过，
            # 不再追问，直接进地图；否则会吐一屏 traceback（自测发现）
            print("\n（输入已结束，提前结束本次答题——已答的题都已保存）")
            break
        score, details = bank.grade_answer(q, answer)
        print(f"得分: {score:.2f}" + ("（部分正确）" if 0.0 < score < 1.0 else ""))
        # P4 第 0 步 A 路：答错后可选追问解释（判分后、讲解前--先让学习者
        # 说出自己的思路，再看正确答案）。解释文本进 Observation，是
        # misconception 关键词检测在产品路径唯一的输入来源
        explanation = ask_explanation() if score < 0.6 else ""
        if details:
            for d in details[:3]:
                if "error" in d:
                    print(f"  {d['error']}")
                else:
                    print(f"  用例 {d.get('args')}: 期望 {d.get('expected')}, "
                          f"得到 {d.get('got')} {'✓' if d.get('passed') else '✗'}")
        if q.qtype == "choice" and q.option_explanations:
            # 选择题逐选项讲解：点出所选选项为何对/错（错选正是要澄清的误解）
            try:
                chosen = int(answer.strip())
            except ValueError:
                chosen = -1
            if 0 <= chosen < len(q.option_explanations):
                print(f"  讲解: {q.option_explanations[chosen]}")
        elif q.explanation:
            print(f"  要点: {q.explanation}")
        tc_before = state.C.tc_states.get(q.skill_id)
        prev_tc_status = tc_before.status if tc_before else None
        illusory_before = len(state.C.illusory_confidence_hits)
        misc_before = len(state.C.misconception_hits)
        obs = Observation(
            skill_id=q.skill_id, problem_id=q.problem_id, score=score,
            bloom_level=q.bloom_level, self_confidence=self_conf,
            user_answer=answer, explanation_text=explanation,
        )
        state = engine.update(state, obs)
        # P4 第 0 步 B 路：命中记录落库（对账原料）。本题是否新增命中用
        # 命中数增量判断（引擎每次 update 最多追加一条）--不能用「最后一条
        # 命中的题号 == 本题」：重练曾命中的题且本次未命中时，旧行命中会被
        # 误标进新行（web 练习轮自测发现）
        misc_id = None
        if len(state.C.misconception_hits) > misc_before:
            misc_id = state.C.misconception_hits[-1].misc_id
        db.save_response(state.user_id, obs.to_dict(),
                         illusory_flag=len(state.C.illusory_confidence_hits) > illusory_before,
                         misc_id=misc_id)
        live = _liminal_live_feedback(engine, state, q.skill_id, score,
                                      q.bloom_level, prev_tc_status)
        if live:
            print(f"  {live}")
        illusory_line = _illusory_live_feedback(state, illusory_before)
        if illusory_line:
            print(f"  {illusory_line}")
        print()
    db.save_state(state)
    return state


def _calibration_line(engine: BeliefEngine) -> str:
    """地图「自评校准度（ECE）」行；无自评数据返回空串（不显示）.

    样本 <5 时诚实标注数据不足（避免拿先验数值下结论，同 C 维度的克制原则）。
    """
    curves = engine.calibration_curves
    if not curves:
        return ""
    total_n = sum(c.n for c in curves)
    if total_n < 5:
        return f"自评校准度（ECE）：数据不足（{total_n} 次自评，需 ≥5）"
    return f"自评校准度（ECE）: {compute_ece(curves):.2f}（越接近 0 自评越准）"


def print_map(engine: BeliefEngine, state: BeliefState,
              covered_bloom: set[BloomLevel] | None = None,
              prev_state: BeliefState | None = None,
              trend: dict | None = None,
              session_rows: list[dict] | None = None) -> None:
    print("\n" + "═" * 56)
    print("你的认知地图")
    print("═" * 56)
    print("（怎么看：每行条形 = 该维度的掌握概率，越接近 100% 越稳；作答越多，数值越准）")

    print("\n[整体解读]")
    print("  " + map_interpretation(engine, state, prev_state=prev_state,
                                   session_rows=session_rows))

    print("\n[5 维状态]（掌握概率）")
    history = engine.get_history(state.user_id)
    n_selfconf = sum(1 for h in history if h.get("self_confidence") is not None)
    for dim, label in DIM_LABELS.items():
        d = getattr(state, dim)
        if dim == "X":
            # MVP 无支架/提示机制，X 无观测来源，诚实标注而不是给一个先验假数值
            print(f"  {dim} {label:<16} （MVP 未提供支架/提示机制，暂未测量）")
            continue
        if dim == "C":
            if state.C.illusory_confidence_hits:
                print(f"  {dim} {label:<16} {_bar(d.mastery_prob)}"
                      f"（发现 {len(state.C.illusory_confidence_hits)} 处失准，见下方）")
            elif n_selfconf > 0:
                # 无失准证据时系统并未"测出 0.55"——不显示误导性先验数值
                # （自测反馈：数值 0.55 被读成"中等自信"，实际语义是校准信息不足）
                print(f"  {dim} {label:<16} 未发现失准（{n_selfconf} 次自评与表现一致）")
            else:
                print(f"  {dim} {label:<16} 暂无自评数据，暂未测量")
            continue
        print(f"  {dim} {label:<16} {_bar(d.mastery_prob)}")

    calibration = _calibration_line(engine)
    if calibration:
        print(f"  {calibration}")

    if prev_state is not None:
        delta_lines = _map_delta_lines(engine, state, prev_state)
        if delta_lines:
            print("\n[与上次相比]")
            for line in delta_lines:
                print("  " + line)

    covered_bloom = covered_bloom or set(BloomLevel)
    print("\n[Bloom 六层分布]（各层掌握概率）")
    for field, label in BLOOM_LABELS:
        if BloomLevel[field.upper()] not in covered_bloom:
            print(f"  {label}  （暂无对应层级题目，暂未测量）")
            continue
        print(f"  {label}  {_bar(getattr(state.bloom_profile, field))}")
    if state.bloom_profile.covered_layers:
        print(f"  当前主导层级: {BLOOM_LAYER_LABELS[state.bloom_profile.dominant_layer]}")
    else:
        print("  当前主导层级: 暂未测量（尚无作答数据）")

    if state.C.illusory_confidence_hits:
        print("\n[伪自信点] 自评很高但实际表现不佳的题：")
        # 全量列出（与上方 C 维度的命中数一致）；伪自信命中需要自评 ≥0.7
        # 且落差 ≥0.5，正常学习路径里不会多到刷屏
        for h in state.C.illusory_confidence_hits:
            print(f"  题 {h.problem_id}: 自评 {h.self_confidence:.0%}，"
                  f"实际得分 {h.score:.0%}（落差 {h.gap:.0%}）")
        print("  这些地方『感觉会』可能掩盖了『其实还没会』，建议重做并讲出理由。")
    else:
        print("\n[伪自信点] 本次无（自评与表现基本校准，或未采集自评）")

    if state.C.misconception_hits:
        print("\n[误解点] 解释文本中检测到的典型误解：")
        # 全量列出（关键词命中需解释文本，正常路径里不会多到刷屏）
        for h in state.C.misconception_hits:
            entry = engine.misconception_library.get(h.misc_id)
            name = entry.name if entry else h.misc_id
            print(f"  题 {h.trigger_problem_id}: {name}（置信度 {h.confidence:.0%}）"
                  f"「{h.evidence_text}」")
        print("  这些解释透露的概念误解值得留意，重做相关题时换个思路。")
    else:
        print("\n[误解点] 本次无（答错后可选的「为什么这么答」追问是检测输入，"
              "跳过则无数据）")

    liminal = [(tid, tc) for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    crossed = [tid for tid, tc in state.C.tc_states.items() if tc.status == "post_liminal"]
    if liminal:
        print("\n[临界概念] 正在跨越中（这不是退步，是学习的正常中间态）：")
        for tid, tc in liminal:
            print(f"  {_tc_display_name(engine, tid)}（跨越进度 {tc.progress:.0%}，"
                  f"{_tc_remaining_text(engine, tc)}）")
    if crossed:
        print("\n[临界概念] 已跨越（恭喜，这些概念你已经真正掌握）：")
        for tid in crossed:
            print(f"  {_tc_display_name(engine, tid)}")
    if not liminal and not crossed:
        print("\n[临界概念] 当前无正在跨越中的概念")

    retest = _retest_candidates(engine)
    if retest:
        print("\n[复习提示] 曾掌握、正在遗忘（隔太久不练会掉）：")
        for skill, peak, decayed, days in retest:
            print(f"  「{_topic_label(skill)}」{days} 天未练，"
                  f"掌握概率从 {peak:.0%} 掉到 {decayed:.0%}")

    line = trend_line(trend or {})
    if line:
        n_sess = next(iter(trend.values()))[2] if trend else 0
        print(f"\n[近几次趋势]（{n_sess} 次会话末对比）")
        print(f"  {line}")

    print("\n[一句话建议]")
    print("  " + next_suggestion(engine, state))
    cmd = practice_command(engine, state)
    if cmd:
        print(f"  立刻练习：{cmd}")
    print()


def _dim_attribution(dim: str, delta_line: str, session_rows: list[dict]) -> str:
    """反思句的「为什么」从句：把维度变化归因到本次会话的具体作答证据.

    归因按 Bloom 层切分（MVP 题库各维度信号的主要来源）：K <- L1/L2 题、
    P <- L3 题、S <- L4+ 题；无对应层作答（如 MIRT 先验微调产生的 delta）
    时返回空串--不臆造证据。
    """
    layer_names = {"K": "记忆/理解层", "P": "应用层", "S": "分析及以上层"}
    bloom_fields = {"K": ("REMEMBER", "UNDERSTAND"), "P": ("APPLY",), "S": ("ANALYZE",)}
    levels = bloom_fields.get(dim)
    if not levels:
        return ""
    rows = [r for r in session_rows if str(r.get("bloom_level", "")).upper() in levels]
    if not rows:
        return ""
    n_correct = sum(1 for r in rows if (r.get("score") or 0.0) >= 0.6)
    n_wrong = len(rows) - n_correct
    # 归因只在方向一致时给出：维度上升归因答对、下降归因答错。方向相反
    # （如 MIRT 全历史重估让 K 在本会话答对时仍下降）-> 无从归因，不臆造
    if "+" in delta_line:
        n, what = n_correct, "题答对"
    else:
        n, what = n_wrong, "题答错"
    if n == 0:
        return ""
    return f"{n} 道{layer_names[dim]}{what}"


def _session_reflection(engine: BeliefEngine, state: BeliefState,
                        prev_state: BeliefState | None,
                        session_rows: list[dict] | None) -> str:
    """B2 反思句：「本次会话变了什么 + 为什么」，1 句、证据可回溯.

    无 prev_state（如 --map-only / 黄金回归 runner）或无 delta 或本次无
    作答 -> 空串（不声称有变化，方案 6.6 DISPROVEN 点）。下一步不在本句
    重复，由解读段既有收尾句引用「一句话建议」承担。
    """
    if prev_state is None or not session_rows:
        return ""
    delta_lines = _map_delta_lines(engine, state, prev_state)
    if not delta_lines:
        return ""
    # 只取维度数值变化行做归因（主导层级/新跨越行自带语义，不归因）
    dim_lines = [l for l in delta_lines if l[:2] in ("K ", "P ", "S ")][:2]
    parts = []
    for line in dim_lines:
        clause = f"本次{line}"
        why = _dim_attribution(line[0], line, session_rows)
        if why:
            clause += f"--{why}"
        parts.append(clause)
    if not parts:
        # 无维度数值变化（只有层级/跨越类 delta）-> 只报变化不归因
        parts = [f"本次{delta_lines[0]}"]
    return "；".join(parts) + "。"


def map_interpretation(engine: BeliefEngine, state: BeliefState,
                       prev_state: BeliefState | None = None,
                       session_rows: list[dict] | None = None) -> str:
    """整体解读段：把地图各分节的信号综合成一段「你现在处于什么状态、为什么、下一步」。

    只做综合与证据回溯，不重复下方分节已有的说明文案；样本少时如实说明信号不稳定。
    """
    history = engine.get_history(state.user_id)
    if not history:
        return ("你还没有作答记录。完成一组题后，系统才能画出有依据的认知地图"
                "（当前显示的是先验默认值）。")

    n = len(history)
    n_correct = sum(1 for h in history if (h.get("score") or 0.0) >= 0.6)
    n_partial = sum(1 for h in history if 0.0 < (h.get("score") or 0.0) < 0.6)
    n_wrong = sum(1 for h in history if (h.get("score") or 0.0) == 0.0)
    n_illusory = len(state.C.illusory_confidence_hits)

    rate = n_correct / n
    if rate >= 0.8:
        tone = "整体掌握良好"
    elif rate >= 0.6:
        tone = "整体掌握处于中间水平，有扎实的部分，也有还没稳的部分"
    else:
        tone = "整体还比较薄弱，不少题还没有真正掌握"

    dist = f"{n} 道题里 {n_correct} 道完整答对"
    if n_partial:
        dist += f"、{n_partial} 道部分正确"
    if n_wrong:
        dist += f"、{n_wrong} 道答错"

    clauses = [f"你目前{tone}：{dist}。"]
    # MIRT 生效（≥2 次作答）后才谈维度差，避免拿先验值下结论。
    # 相对差距还要落在「确实掌握」的水平（掌握线 0.5 之上留余量）才下结论：
    # 全错学习者 P 0.40 > K 0.24 若照常触发，会把 0/5 读成「程序技能更扎实」（真机发现）。
    if n >= 2:
        kp_gap = state.K.mastery_prob - state.P.mastery_prob
        if kp_gap >= 0.15 and state.K.mastery_prob >= 0.6:
            ev = f"（{n_partial} 道题只对了一半）" if n_partial else ""
            clauses.append(
                f"你的「知识记忆」明显强于「程序技能」{ev}——你知道概念，"
                f"但把概念写成一运行就对的代码，这条能力还在建立中。")
        elif kp_gap <= -0.15 and state.P.mastery_prob >= 0.6:
            clauses.append(
                f"你的「程序技能」比「知识记忆」更扎实——动手写代码的能力已经超过背概念。")
    if n_illusory:
        clauses.append(
            f"最该留意的是 {n_illusory} 处伪自信：自评很高但实际没做对。"
            f"做错说明「还没会」，伪自信说明「以为会了其实没会」，后者更值得放慢检查。")
    liminal = [tid for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    if liminal:
        names = "、".join(_tc_display_name(engine, tid) for tid in liminal)
        clauses.append(f"「{names}」正在跨越中，这是学习的正常中间态，不是退步。")
    if n < 5:
        clauses.append(f"作答样本还少（{n} 题），上面的判断会随练习增多而更准。")
    reflection = _session_reflection(engine, state, prev_state, session_rows)
    if reflection:
        clauses.append(reflection)
    clauses.append("具体到下一步，见下方「一句话建议」。")
    return " ".join(clauses)


# 复测分支阈值（方案 P3 4.4，保守双条件）：曾掌握（峰值≥0.7）且
# 显著衰减（衰减后 <0.55 或 跌幅 ≥0.15）
RETEST_PEAK_MIN = 0.7
RETEST_DECAYED_MAX = 0.55
RETEST_DROP_MIN = 0.15


def _retest_candidates(engine: BeliefEngine) -> list[tuple[str, float, float, int]]:
    """命中复测条件的 skill 列表 [(skill, peak, decayed, days)].

    按衰减后掌握概率升序（最遗忘的排最前）；无视图/无候选返回空列表。
    连续练习（days=0 -> decayed==peak）天然不命中。
    """
    view = engine.decay_view
    if not view:
        return []
    out = []
    for skill, (peak, decayed, days) in view.items():
        if peak < RETEST_PEAK_MIN:
            continue
        if decayed < RETEST_DECAYED_MAX or (peak - decayed) >= RETEST_DROP_MIN:
            out.append((skill, peak, decayed, days))
    return sorted(out, key=lambda c: c[2])


def next_suggestion(engine: BeliefEngine, state: BeliefState) -> str:
    # 优先：正在 liminal 的概念 -> 巩固应用题（合意困难：不引入新概念）
    liminal = [tid for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    if liminal:
        tid = liminal[0]
        remaining = _tc_remaining_text(engine, state.C.tc_states[tid])
        return (f"「{_topic_label(tid)}」正在跨越中——{remaining}。"
                f"建议接下来做 3 道「{_topic_label(tid)}」的应用题（L3），不建议现在学新概念。")
    # 其次：曾掌握但正在遗忘的 topic -> 复测（间隔效应，何时做）
    for skill, peak, decayed, days in _retest_candidates(engine):
        return (f"「{_topic_label(skill)}」上次 {days} 天前练过，掌握概率从 {peak:.0%} "
                f"掉到 {decayed:.0%}--建议先做 3 道复测题，趁遗忘加深前巩固。")
    # 再次：练过的 topic 里 BKT 最弱的
    practiced = engine.l1.all_skills()
    if practiced:
        weakest = min(practiced, key=lambda s: engine.get_bkt_mastery(s))
        if engine.get_bkt_mastery(weakest) < 0.7:
            return (f"「{_topic_label(weakest)}」是当前最弱的一项"
                    f"（掌握概率 {engine.get_bkt_mastery(weakest):.0%}），建议再做几道对应的基础题巩固。")
        return "已练概念掌握良好，下次可以开启新 topic（比如还没练过的那个）。"
    return "先完成一组题，系统才能给出有依据的建议。"


def suggested_practice(engine: BeliefEngine, state: BeliefState) -> tuple[str, BloomLevel | None] | None:
    """返回建议练习目标 (topic, level)；无可执行建议时返回 None.

    level 为 None 表示不限层级（基础题巩固/复测）。与 next_suggestion 决策一致：
    优先正在 liminal 的概念（巩固应用题 L3），其次曾掌握且显著衰减的 topic
    （复测），否则练过且 BKT 最弱的 topic。
    """
    liminal = [tid for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    if liminal:
        return (liminal[0], BloomLevel.APPLY)
    for skill, peak, decayed, days in _retest_candidates(engine):
        return (skill, None)
    practiced = engine.l1.all_skills()
    if practiced:
        weakest = min(practiced, key=lambda s: engine.get_bkt_mastery(s))
        if engine.get_bkt_mastery(weakest) < 0.7:
            return (weakest, None)
    return None


def practice_command(engine: BeliefEngine, state: BeliefState) -> str:
    """建议对应的可执行命令；无建议时返回空串（供地图「立刻练习」行）."""
    target = suggested_practice(engine, state)
    if target is None:
        return ""
    topic, level = target
    level_part = f" --level L{level.value}" if level is not None else ""
    return f"cogmirror --topic {topic}{level_part} --questions 3"


def _refresh_decay_view(engine: BeliefEngine, db: Database, user_id: str) -> None:
    """重算间隔衰减视图（会话开始/每轮答题后调用，纯只读，l1 语义不变）.

    答题后刷新避免陈旧提示：刚练过的 topic days 归零、不再命中复测条件。
    """
    rows = db.load_responses(user_id)
    engine.decay_view = engine.decayed_mastery_view(rows) if rows else None


def _run_data_command(args) -> int:
    """--export / --delete：成人向合规的"可导出、可删除"从 CLI 直达.

    数据层的 request/export/delete 已具备，这里补上 CLI 入口（PRD 第9节承诺）。
    """
    db = Database(args.db)
    try:
        if db.get_user(args.user) is None:
            print(f"用户 {args.user} 在 {args.db} 中不存在，无需处理。")
            return 0
        if args.export:
            db.request_data_export(args.user)
            data = db.export_user_data(args.user)
            print(json.dumps(data, ensure_ascii=False, indent=2))
            print(f"\n已记录用户 {args.user} 的导出请求。")
            return 0
        try:
            confirm = input(f"确认删除用户 {args.user} 的全部作答与认知数据？"
                            f"此操作不可恢复，输入 DELETE 确认：").strip()
        except EOFError:
            print("已取消（未收到确认输入），未删除任何数据。")
            return 0
        if confirm != "DELETE":
            print("已取消，未删除任何数据。")
            return 0
        db.request_data_delete(args.user)
        print(f"已删除 {args.user} 的全部数据（删除请求已记录在用户档案）。")
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cogmirror", description="编程学习认知教练（Phase 0 CLI）")
    parser.add_argument("--user", default="local_user", help="用户 ID（本地单用户默认 local_user）")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite 路径")
    parser.add_argument("--questions", type=int, default=10, help="本次答题数量")
    parser.add_argument("--topic", default="", help="只练指定 topic（如 python.loops）")
    parser.add_argument("--level", type=_parse_level, default=None,
                        help="只练指定 Bloom 层级（如 L3 或 APPLY）")
    parser.add_argument("--map-only", action="store_true", help="只看认知地图，不答题")
    parser.add_argument("--review", action="store_true",
                        help="重练全部错题（最近一次得分 < 60% 的题）")
    parser.add_argument("--export", action="store_true",
                        help="导出该用户全部作答数据为 JSON（成人向合规：可导出）")
    parser.add_argument("--delete", action="store_true",
                        help="删除该用户全部作答与认知数据（成人向合规：可删除，需确认）")
    args = parser.parse_args(argv)

    if args.export or args.delete:
        return _run_data_command(args)

    bank = QuestionBank()
    db = Database(args.db)
    db.ensure_user(args.user)
    # P4：misconception 证据闭环--从 DB 恢复证据计数，注入引擎（命中置信度
    # 数据驱动）；会话结束对账后写回
    tracker = MisconceptionTracker()
    tracker.load(db.load_misconception_evidence())
    engine = BeliefEngine(misconception_tracker=tracker)
    engine.l2.register_items_bulk(bank.mirt_items())
    # 对账只看本次会话新增的响应（窗口限同一会话，方案 5.7）
    n_rows_before = len(db.load_responses(args.user))

    state = db.load_latest_state(args.user)
    if state is None:
        state = engine.create_initial_state(args.user)
        print(f"新用户 {args.user}，创建初始状态。")
        _print_welcome()
    else:
        # 从作答历史恢复引擎内部状态（MIRT 输入）
        history = [
            {"problem_id": r["problem_id"], "correct": r["correct"], "score": r["score"],
             "bloom_level": r["bloom_level"], "self_confidence": r["self_confidence"]}
            for r in db.load_responses(args.user)
        ]
        engine.set_history(args.user, history)
        # P2 校准曲线：responses 是派生视图（无新表），每次加载重算，
        # 伪自信折扣由曲线驱动（数据不足时引擎内回退固定值）
        engine.set_calibration(CalibrationCurveComputer().compute(history))
        # P3 间隔衰减视图：从 responses 历史重放峰值 + 无状态衰减（只读派生），
        # 复测建议与 [复习提示] 消费它
        _refresh_decay_view(engine, db, args.user)
        print(f"欢迎回来，{args.user}（已完成 {len(history)} 次作答）。")
        overview = _welcome_progress(engine, state)
        if overview:
            print(f"  进度概览：{overview}")
        # P5 B1：上次会话卡点主动浮现（跨会话"相关历史召回"的本地退化版）
        struggles = last_session_struggles(db, args.user)
        if struggles:
            names = "、".join(_topic_label(s) for s in struggles[:3])
            print(f"  上次卡住：{names}"
                  + ("等" if len(struggles) > 3 else ""))

    review_ids = None
    if args.review:
        review_ids = _wrong_problem_ids(db, args.user)
        if review_ids:
            print(f"错题重练：找到 {len(review_ids)} 道最近一次得分不足 60% 的题。")
        else:
            print("当前没有需要重练的错题（最近一次得分都 ≥ 60%）。")

    prev_state = None
    if not (args.map_only or review_ids == []):
        # engine.update 原地修改 state，必须深拷贝一份作为「与上次相比」基准
        prev_state = copy.deepcopy(state)
        state = run_session(engine, bank, state, db,
                            len(review_ids) if review_ids else args.questions,
                            topic=args.topic, level=args.level,
                            problem_ids=review_ids)
        _refresh_decay_view(engine, db, args.user)

    covered_bloom = {q.bloom_level for q in bank.all_questions()}
    all_rows = db.load_responses(args.user)
    print_map(engine, state, covered_bloom, prev_state=prev_state,
              trend=multi_session_trend(db, args.user),
              session_rows=all_rows[n_rows_before:])

    # 按建议直达练习：地图末尾问是否继续练建议的题组，直到无建议或用户拒绝。
    # 一轮练习后重渲染地图（liminal 概念可能因此跨过），状态经 db 持久化。
    while True:
        target = suggested_practice(engine, state)
        if target is None:
            break
        topic, level = target
        retest_topics = {c[0] for c in _retest_candidates(engine)}
        if level is not None:
            level_txt = "的 L3 题"
        elif topic in retest_topics:
            level_txt = "的复测题"
        else:
            level_txt = "的基础题"
        tc = state.C.tc_states.get(topic)
        extra = f"（{_tc_remaining_text(engine, tc)}）" if tc and tc.status == "liminal" else ""
        try:
            ans = input(f"按建议现在练 3 道「{_topic_label(topic)}」{level_txt}吗？{extra}[y/N] ").strip().lower()
        except EOFError:
            break  # 非交互/输入耗尽 -> 视为拒绝
        if ans != "y":
            break
        prev_round = copy.deepcopy(state)
        state = run_session(engine, bank, state, db, 3, topic=topic, level=level)
        _refresh_decay_view(engine, db, args.user)
        all_rows = db.load_responses(args.user)
        print_map(engine, state, covered_bloom, prev_state=prev_round,
                  trend=multi_session_trend(db, args.user),
                  session_rows=all_rows[n_rows_before:])

    # P4 收尾：把本次会话的 misconception 命中与同 skill 后续表现对账，
    # 证据计数落库（下次会话恢复，命中置信度随之变化）
    rows = db.load_responses(args.user)
    tracker.reconcile(rows[n_rows_before:])
    db.save_misconception_evidence(tracker.dump())

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
