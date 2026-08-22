"""Phase 0 最小链路 CLI：做题 -> 5D 状态更新 -> 认知地图展示.

不追求界面美观，追求链路真实可用（ROADMAP Phase 0）。
运行：python -m cogmirror.cli  或安装后 cogmirror
"""

from __future__ import annotations

import argparse
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


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled) + f" {value:.2f}"


def _tc_display_name(engine: BeliefEngine, tid: str) -> str:
    """TC 显示名单一来源：状态机库（避免第二套库文案漂移）."""
    return engine.tc_detector.tc_library.get(tid, {}).get("name", tid)


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
        return input("输入选项编号: ")
    if question.qtype == "fill":
        return input("输入你的答案: ")
    print("（输入代码，单独一行输入 END 结束）")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def run_session(engine: BeliefEngine, bank: QuestionBank, state: BeliefState,
                db: Database, n_questions: int) -> BeliefState:
    questions = bank.all_questions()[:n_questions]
    print(f"\n本组共 {len(questions)} 道题。\n")
    for i, q in enumerate(questions, 1):
        print(f"── 题 {i}/{len(questions)} [{q.problem_id}] "
              f"({q.topic} / {q.bloom_level.name}) ──")
        print(q.prompt)
        self_conf = ask_self_confidence()
        answer = read_answer(q)
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
        obs = Observation(
            skill_id=q.skill_id, problem_id=q.problem_id, score=score,
            bloom_level=q.bloom_level, self_confidence=self_conf,
            user_answer=answer, explanation_text="",
        )
        state = engine.update(state, obs)
        db.save_response(state.user_id, obs.to_dict(),
                         illusory_flag=bool(state.C.illusory_confidence_hits
                                            and state.C.illusory_confidence_hits[-1].problem_id == q.problem_id))
        print()
    db.save_state(state)
    return state


def print_map(engine: BeliefEngine, state: BeliefState,
              covered_bloom: set[BloomLevel] | None = None) -> None:
    print("\n" + "═" * 56)
    print("你的认知地图")
    print("═" * 56)

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

    covered_bloom = covered_bloom or set(BloomLevel)
    print("\n[Bloom 六层分布]")
    for field, label in BLOOM_LABELS:
        if BloomLevel[field.upper()] not in covered_bloom:
            print(f"  {label}  （暂无对应层级题目，暂未测量）")
            continue
        print(f"  {label}  {_bar(getattr(state.bloom_profile, field))}")
    if state.bloom_profile.covered_layers:
        print(f"  当前主导层级: {state.bloom_profile.dominant_layer.name}")
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
    if liminal:
        print("\n[临界概念] 正在跨越中（这不是退步，是学习的正常中间态）：")
        for tid, tc in liminal:
            print(f"  {_tc_display_name(engine, tid)}（跨越进度 {tc.progress:.0%}）")
    else:
        print("\n[临界概念] 当前无正在跨越中的概念")

    print("\n[一句话建议]")
    print("  " + next_suggestion(engine, state))
    print()


def next_suggestion(engine: BeliefEngine, state: BeliefState) -> str:
    # 优先：正在 liminal 的概念 -> 巩固应用题（合意困难：不引入新概念）
    liminal = [tid for tid, tc in state.C.tc_states.items() if tc.status == "liminal"]
    if liminal:
        tid = liminal[0]
        return f"建议接下来做 3 道「{_topic_label(tid)}」的应用题（L3），不建议现在学新概念。"
    # 其次：练过的 topic 里 BKT 最弱的
    practiced = engine.l1.all_skills()
    if practiced:
        weakest = min(practiced, key=lambda s: engine.get_bkt_mastery(s))
        if engine.get_bkt_mastery(weakest) < 0.7:
            return f"「{_topic_label(weakest)}」目前掌握概率 {engine.get_bkt_mastery(weakest):.0%}，建议再做几道对应的基础题巩固。"
        return "已练概念掌握良好，下次可以开启新 topic（比如还没练过的那个）。"
    return "先完成一组题，系统才能给出有依据的建议。"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cogmirror", description="编程学习认知教练（Phase 0 CLI）")
    parser.add_argument("--user", default="local_user", help="用户 ID（本地单用户默认 local_user）")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite 路径")
    parser.add_argument("--questions", type=int, default=10, help="本次答题数量")
    parser.add_argument("--map-only", action="store_true", help="只看认知地图，不答题")
    args = parser.parse_args(argv)

    bank = QuestionBank()
    engine = BeliefEngine()
    engine.l2.register_items_bulk(bank.mirt_items())
    db = Database(args.db)
    db.ensure_user(args.user)

    state = db.load_latest_state(args.user)
    if state is None:
        state = engine.create_initial_state(args.user)
        print(f"新用户 {args.user}，创建初始状态。")
    else:
        # 从作答历史恢复引擎内部状态（MIRT 输入）
        history = [
            {"problem_id": r["problem_id"], "correct": r["correct"], "score": r["score"],
             "bloom_level": r["bloom_level"], "self_confidence": r["self_confidence"]}
            for r in db.load_responses(args.user)
        ]
        engine.set_history(args.user, history)
        print(f"欢迎回来，{args.user}（已完成 {len(history)} 次作答）。")

    if not args.map_only:
        state = run_session(engine, bank, state, db, args.questions)

    covered_bloom = {q.bloom_level for q in bank.all_questions()}
    print_map(engine, state, covered_bloom)
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
