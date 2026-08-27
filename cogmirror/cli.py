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
from .db import Database, DEFAULT_DB_PATH
from .questions import QuestionBank

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
        if details:
            for d in details[:3]:
                if "error" in d:
                    print(f"  {d['error']}")
                else:
                    print(f"  用例 {d.get('args')}: 期望 {d.get('expected')}, "
                          f"得到 {d.get('got')} {'✓' if d.get('passed') else '✗'}")
        if q.explanation:
            print(f"  要点: {q.explanation}")
        tc_before = state.C.tc_states.get(q.skill_id)
        prev_tc_status = tc_before.status if tc_before else None
        illusory_before = len(state.C.illusory_confidence_hits)
        obs = Observation(
            skill_id=q.skill_id, problem_id=q.problem_id, score=score,
            bloom_level=q.bloom_level, self_confidence=self_conf,
            user_answer=answer, explanation_text="",
        )
        state = engine.update(state, obs)
        db.save_response(state.user_id, obs.to_dict(),
                         illusory_flag=bool(state.C.illusory_confidence_hits
                                            and state.C.illusory_confidence_hits[-1].problem_id == q.problem_id))
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


def print_map(engine: BeliefEngine, state: BeliefState,
              covered_bloom: set[BloomLevel] | None = None,
              prev_state: BeliefState | None = None) -> None:
    print("\n" + "═" * 56)
    print("你的认知地图")
    print("═" * 56)
    print("（怎么看：每行条形 = 该维度的掌握概率，越接近 100% 越稳；作答越多，数值越准）")

    print("\n[整体解读]")
    print("  " + map_interpretation(engine, state))

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

    print("\n[一句话建议]")
    print("  " + next_suggestion(engine, state))
    cmd = practice_command(engine, state)
    if cmd:
        print(f"  立刻练习：{cmd}")
    print()


def map_interpretation(engine: BeliefEngine, state: BeliefState) -> str:
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
    clauses.append("具体到下一步，见下方「一句话建议」。")
    return " ".join(clauses)


def next_suggestion(engine: BeliefEngine, state: BeliefState) -> str:
    # 优先：正在 liminal 的概念 -> 巩固应用题（合意困难：不引入新概念）
    liminal = [tid for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    if liminal:
        tid = liminal[0]
        remaining = _tc_remaining_text(engine, state.C.tc_states[tid])
        return (f"「{_topic_label(tid)}」正在跨越中——{remaining}。"
                f"建议接下来做 3 道「{_topic_label(tid)}」的应用题（L3），不建议现在学新概念。")
    # 其次：练过的 topic 里 BKT 最弱的
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

    level 为 None 表示不限层级（基础题巩固）。与 next_suggestion 决策一致：
    优先正在 liminal 的概念（巩固应用题 L3），否则练过且 BKT 最弱的 topic。
    """
    liminal = [tid for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    if liminal:
        return (liminal[0], BloomLevel.APPLY)
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
    engine = BeliefEngine()
    engine.l2.register_items_bulk(bank.mirt_items())
    db = Database(args.db)
    db.ensure_user(args.user)

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
        print(f"欢迎回来，{args.user}（已完成 {len(history)} 次作答）。")
        overview = _welcome_progress(engine, state)
        if overview:
            print(f"  进度概览：{overview}")

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

    covered_bloom = {q.bloom_level for q in bank.all_questions()}
    print_map(engine, state, covered_bloom, prev_state=prev_state)

    # 按建议直达练习：地图末尾问是否继续练建议的题组，直到无建议或用户拒绝。
    # 一轮练习后重渲染地图（liminal 概念可能因此跨过），状态经 db 持久化。
    while True:
        target = suggested_practice(engine, state)
        if target is None:
            break
        topic, level = target
        level_txt = "的 L3 题" if level is not None else "的基础题"
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
        print_map(engine, state, covered_bloom, prev_state=prev_round)

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
