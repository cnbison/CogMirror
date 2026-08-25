"""Phase 0.5 spike CLI.

用法：
  python -m spike diagnose --user X --topics loops scope
      [--model MODEL] [--jsonl DIR] [--max-rounds 30] [--ground-truth]
  python -m spike compare --jsonl DIR --user X
  python -m spike smoke [--topics loops variables]

smoke 用 FakeLLM 全链路跑通，无需任何环境变量 / API key（可复现）。
diagnose 走真实 AnthropicClient，需要 ANTHROPIC_API_KEY。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from cogmirror.questions import QuestionBank

from .compare import compare_n1, render_comparison
from .dialogue import DialogueEngine
from .graph import build_graph, DimensionId, Graph, TOPIC_SHORT_TO_ID
from .llm import AnthropicClient, LLMConfig, SpikeConfigError
from .protocol import (
    GRAPH_VERSION,
    GroundTruthAnchors,
    SessionRecord,
    build_bank_anchors,
    load_sessions,
    save_session,
)
from .scorer import score_session

DEFAULT_SESSION_DIR = "spike/data/sessions/"

TOPIC_LABELS = {
    "python.variables": "变量赋值",
    "python.loops": "循环",
    "python.functions": "函数",
    "python.recursion": "递归",
    "python.scope": "作用域",
}


def _resolve_topics(topics_arg: list[str]) -> list[str]:
    """短名 -> 完整 topic id；未知短名报错."""
    ids: list[str] = []
    for t in topics_arg:
        tid = TOPIC_SHORT_TO_ID.get(t)
        if tid is None:
            raise SystemExit(f"未知 topic 短名: {t}（可选: "
                             + ", ".join(sorted(TOPIC_SHORT_TO_ID)) + "）")
        ids.append(tid)
    return ids


def _read_bank_answer(question) -> str:
    if question.qtype == "choice":
        for i, opt in enumerate(question.options):
            print(f"  {i}. {opt}")
        return input("输入选项编号: ").strip()
    if question.qtype == "fill":
        return input("输入你的答案: ").strip()
    print("（输入代码，单独一行输入 END 结束）")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _collect_bank_anchors(bank: QuestionBank, topics: list[str]) -> GroundTruthAnchors:
    """题库测验（确定性判分）作为 ground truth 锚点."""
    print("\n== 第一步：题库锚点测验（确定性判分，作为独立 ground truth）==")
    print("（先做一组静态题，建立你的能力锚点，再进入对话诊断）")
    answers: dict[str, str] = {}
    for q in bank.all_questions():
        if q.topic not in topics:
            continue
        print(f"\n[{q.problem_id}] ({q.topic} / {q.bloom_level.name})")
        print(q.prompt)
        answers[q.problem_id] = _read_bank_answer(q)
    return build_bank_anchors(bank, answers)


def _print_diagnose_summary(rec: SessionRecord) -> None:
    print("\n" + "═" * 56)
    print("对话诊断摘要")
    print("═" * 56)
    print(f"用户: {rec.user_id}   模型: {rec.model}   轮次: {len(rec.transcript)}")
    print(f"P 代码执行锚点: {len(rec.exec_results)} 个")
    if rec.estimate:
        dims = "  ".join(
            f"{d.value}={rec.estimate.five_d.get(d, float('nan')):.2f}"
            for d in DimensionId)
        print(f"[5D 估计] {dims}")
        if rec.estimate.insufficient:
            print(f"[证据不足] {'、'.join(rec.estimate.insufficient)}（暂未测量）")
        else:
            print("[证据不足] 无（各维度均有对话证据）")
    else:
        print("[5D 估计] 评分失败，无估计值（诚实标注）")


def cmd_diagnose(args: argparse.Namespace) -> int:
    graph = build_graph()
    bank = QuestionBank()
    topics = _resolve_topics(args.topics)

    try:
        llm = AnthropicClient(LLMConfig(model=args.model) if args.model else LLMConfig())
    except SpikeConfigError as e:
        print(f"无法启动诊断: {e}")
        print("提示：设置 ANTHROPIC_API_KEY 环境变量后再运行 diagnose；"
              "冒烟测试用 python -m spike smoke 无需任何环境变量。")
        return 1

    ground_truth = _collect_bank_anchors(bank, topics) if args.ground_truth else None

    engine = DialogueEngine(llm, graph, bank, topics, max_rounds=args.max_rounds)
    state = engine.run(args.user)

    estimate = score_session(llm, graph, state.transcript, state.exec_results)
    rec = SessionRecord(
        user_id=args.user, date=datetime.now().isoformat(),
        graph_version=GRAPH_VERSION, model=llm.config.model,
        ground_truth=ground_truth, transcript=state.transcript,
        exec_results=state.exec_results, estimate=estimate,
    )
    path = save_session(Path(args.jsonl) / f"{args.user}.jsonl", rec)
    llm.close()

    _print_diagnose_summary(rec)
    print(f"\n会话已保存: {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    records, errors = load_sessions(args.jsonl)
    if errors:
        for e in errors:
            print(f"[警告] 跳过坏行: {e}")
    if not records:
        print(f"目录 {args.jsonl} 没有可用的会话记录")
        return 1
    target = [r for r in records if r.user_id == args.user] or records
    for rec in target:
        print(f"\n用户 {rec.user_id}（{rec.date}，模型 {rec.model}）")
        if rec.ground_truth is None:
            print("  无 ground truth 锚点，无法比对（用 --ground-truth 重新诊断）")
            continue
        report = compare_n1(rec)
        print(render_comparison(report))
    return 0


# ── smoke：FakeLLM 全链路，无需环境变量 ─────────────────────────────────


def _make_smoke_responder() -> tuple:
    """确定性 fake 应答器.

    性质：
    - 第一次面试官调用返回非法 anchor（触发重试路径）
    - 对 variables-L4（X 预测验证）返回"你有多大把握"的追问（伪自信轮的提问）
    - 评分器返回固定的确定性 JSON
    返回 (responder, meta)，meta 记录非法 anchor 发射次数供测试断言。
    """
    meta = {"invalid_emitted": 0}

    def responder(system: str, user: str) -> str:
        # 评分器 system 含 rubric 标记「评卷判据」；面试官 system 没有——据此分流
        if "评卷判据" not in system:
            m = re.search(r"本轮要探测的节点 id：(\S+)", user)
            node_id = m.group(1) if m else "invalid"
            if meta["invalid_emitted"] == 0:
                meta["invalid_emitted"] += 1
                return json.dumps({"anchor": "not-a-real-node",
                                   "question": "（非法 anchor 的提问，应被重试）"})
            if node_id == "variables-L4-S4-X":
                return json.dumps({"anchor": node_id,
                                   "question": "你有多大把握说对？请先预测结果，再说依据。"})
            return json.dumps({"anchor": node_id, "question": "请回答：根据你的理解解释一下。"})
        # 评分器
        return json.dumps({
            "five_d": {"K": 0.65, "P": 0.40, "S": 0.55, "C": 0.55, "X": 0.45},
            "bloom": {"REMEMBER": 0.70, "UNDERSTAND": 0.65, "APPLY": 0.40, "ANALYZE": 0.55},
            "solo": {"loops": 3.0, "variables": 3.0},
            "overall": 0.55,
            "evidence_notes": {
                "smoke_fake": "variables-L4 用户高把握但预测错误（伪自信轮）"},
            "insufficient": [],
        })

    return responder, meta


def _make_smoke_ask():
    """脚本化用户作答（确定性）."""
    answers = {
        "loops-L1-S1-K": "range(5) 是 0 到 4，左闭右开，最后一个不是 5。",
        "loops-L2-S2-C": "range(1,5) 是 1,2,3,4；range(5) 是 0..4。stop 不包含。",
        # 伪自信：很确定但代码是错的（range(n) 求和差一位）
        "loops-L3-S3-P": "def sum_to(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total",
        "loops-L4-S4-S": "死循环是因为循环体里没更新 i，我会先初始化再 i += 1。",
        "variables-L1-S1-K": "变量是给值起的名字，x = 5 让 x 指向 5。",
        "variables-L2-S3-C": "x = x + 1 是先算右边再存回左边，不是数学等式。",
        "variables-L3-S3-P": "def swap_values(a, b):\n    return (b, a)",
        # 伪自信轮：高把握但预测错误
        "variables-L4-S4-X": "我很有把握：结果是 [1, 2]。赋值就是拷贝，b 改不到 a。",
    }

    def ask(node):
        return answers[node.node_id]

    return ask


def _make_smoke_bank_answers() -> dict[str, str]:
    """题库锚点测验的脚本作答（确定性，与对话同一个人设）. """
    return {
        "pv-l1-01": "1",   # 对
        "pv-l2-01": "3",   # 错（伪自信：高把握但答错）
        "pv-l3-01": "def swap_values(a, b):\n    return (b, a)",  # 对
        "pv-l4-01": "0",   # 错
        "pl-l1-01": "0,1,2,3,4",  # 对
        "pl-l2-01": "1",   # 对
        "pl-l3-01": "def sum_to(n):\n    return sum(range(n))",   # 错
        "pl-l4-01": "1",   # 对
    }


def cmd_smoke(args: argparse.Namespace) -> int:
    from .llm import FakeLLM

    graph = build_graph()
    bank = QuestionBank()
    topics = _resolve_topics(args.topics)

    responder, meta = _make_smoke_responder()
    llm = FakeLLM(responder)
    engine = DialogueEngine(llm, graph, bank, topics, max_rounds=args.max_rounds)
    state = engine.run(args.user, ask=_make_smoke_ask())

    estimate = score_session(llm, graph, state.transcript, state.exec_results)
    ground_truth = build_bank_anchors(bank, _make_smoke_bank_answers())
    rec = SessionRecord(
        user_id=args.user, date=datetime.now().isoformat(),
        graph_version=GRAPH_VERSION, model="fake-smoke",
        ground_truth=ground_truth, transcript=state.transcript,
        exec_results=state.exec_results, estimate=estimate,
    )
    out_path = Path(args.jsonl) / "smoke_report.jsonl"
    save_session(out_path, rec)

    print("══ 冒烟测试：FakeLLM 全链路（无需 API key，确定性可复现）══")
    print(f"覆盖节点: {len(state.covered_nodes)} 个（topics: {', '.join(topics)}）")
    print(f"面试官 LLM 调用: {len(llm.calls)} 次，其中非法 anchor 重试: "
          f"{meta['invalid_emitted']} 次")
    print(f"P 代码执行锚点: {len(state.exec_results)} 个")
    report = compare_n1(rec)
    print(render_comparison(report))
    print(f"\n会话已保存: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spike", description="Phase 0.5 对话式认知诊断 spike")
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="跑一轮对话诊断（真实 LLM，需 API key）")
    p_diag.add_argument("--user", required=True)
    p_diag.add_argument("--topics", nargs="+", required=True)
    p_diag.add_argument("--model", default=None)
    p_diag.add_argument("--jsonl", default=DEFAULT_SESSION_DIR)
    p_diag.add_argument("--max-rounds", type=int, default=30)
    p_diag.add_argument("--ground-truth", action="store_true",
                        help="诊断前先做题库测验，建立 ground truth 锚点")
    p_diag.set_defaults(func=cmd_diagnose)

    p_cmp = sub.add_parser("compare", help="比对已有会话")
    p_cmp.add_argument("--jsonl", default=DEFAULT_SESSION_DIR)
    p_cmp.add_argument("--user", default=None, help="只比对指定用户；缺省比对全部")
    p_cmp.set_defaults(func=cmd_compare)

    p_smoke = sub.add_parser("smoke", help="FakeLLM 全链路冒烟（无需 API key）")
    p_smoke.add_argument("--user", default="smoke-user")
    p_smoke.add_argument("--topics", nargs="+", default=["loops", "variables"])
    p_smoke.add_argument("--jsonl", default="spike/data/")
    p_smoke.add_argument("--max-rounds", type=int, default=30)
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
