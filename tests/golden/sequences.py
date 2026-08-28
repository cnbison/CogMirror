"""黄金序列定义（P1 回归评测基建）.

每条序列 = 一个"合成学习者"的可复现作答历史：按顺序 replay 进 BeliefEngine，
断言引擎输出落在 expect 窗口内（docs/implementation-plan.md 第 2.4 节）。

约定：
- steps 只声明 (problem_id, score, self_confidence?, explanation_text?)；
  skill_id / bloom_level / qtype 一律从 QuestionBank 按 problem_id 查出，
  保证与产品路径同一份题目定义（不复制第二套元数据造成漂移）。
- Observation.timestamp 由 runner 固定为 BASE_TIME + i 分钟（Observation 默认
  datetime.now()，不固定则涉时间的断言不可复现）。
- expect 窗口来自首版基线生成时的实测值 + 人工合理性判断（见 baseline.json
  生成记录）；窗口收紧 = 更敏感，放宽 = 更迟钝，调整须在 PR 里说明。

覆盖：5 个 topic × 主要行为模式（全对 / 全错 / 伪自信密集 / liminal 跨越中 /
部分正确混合 / misconception 文本触发）。
"""

from __future__ import annotations

SEQUENCES: list[dict] = [
    {
        "name": "all_correct_loops",
        "steps": [
            {"problem_id": "pl-l2-01", "score": 1.0, "self_confidence": 0.9},
            {"problem_id": "pl-l3-01", "score": 1.0, "self_confidence": 0.9},
            {"problem_id": "pl-l3-02", "score": 1.0, "self_confidence": 0.85},
            {"problem_id": "pl-l3-03", "score": 1.0, "self_confidence": 0.9},
            {"problem_id": "pl-l3-04", "score": 1.0, "self_confidence": 0.9},
            {"problem_id": "pl-l4-01", "score": 1.0, "self_confidence": 0.85},
            {"problem_id": "pl-l5-01", "score": 1.0, "self_confidence": 0.9},
            {"problem_id": "pl-l6-01", "score": 1.0, "self_confidence": 0.9},
        ],
        "expect": {
            # 8 连对 + 6 次 L3+ 正确 -> loops 跨过临界概念（不可逆）
            "tc": {"python.loops": "post_liminal"},
            "bkt": {"python.loops": (0.85, 1.0)},
            "mastery": {"P": (0.6, 0.95)},
            "dominant_bloom": "APPLY",
            "suggestion_contains": ["新 topic"],
            "practice": None,
        },
    },
    {
        "name": "all_wrong_variables",
        "steps": [
            {"problem_id": "pv-l1-01", "score": 0.0, "self_confidence": 0.3},
            {"problem_id": "pv-l2-01", "score": 0.0, "self_confidence": 0.2},
            {"problem_id": "pv-l2-02", "score": 0.0, "self_confidence": 0.3},
            {"problem_id": "pv-l3-01", "score": 0.0, "self_confidence": 0.25},
            {"problem_id": "pv-l3-02", "score": 0.0, "self_confidence": 0.3},
            {"problem_id": "pv-l4-01", "score": 0.0, "self_confidence": 0.2},
        ],
        "expect": {
            # 自评低 + 全错 -> 无伪自信；临界概念停在 pre_liminal
            "tc": {"python.variables": "pre_liminal"},
            "bkt": {"python.variables": (0.0, 0.15)},
            "mastery": {"K": (0.0, 0.35), "P": (0.0, 0.45)},
            "dominant_bloom": "REMEMBER",
            "illusory_hits": 0,
            "suggestion_contains": ["变量赋值", "最弱"],
            "practice": ("python.variables", None),
        },
    },
    {
        "name": "overconfident_loop_learner",
        "steps": [
            {"problem_id": "pl-l1-01", "score": 0.0, "self_confidence": 0.9},
            {"problem_id": "pl-l2-01", "score": 0.0, "self_confidence": 0.95},
            {"problem_id": "pl-l3-01", "score": 0.1, "self_confidence": 0.9},
            {"problem_id": "pl-l3-02", "score": 0.0, "self_confidence": 0.85},
            {"problem_id": "pl-l4-01", "score": 0.0, "self_confidence": 0.9},
        ],
        "expect": {
            # 5 次自评 >=0.7 且落差 >=0.5 -> 全部命中伪自信；
            # discount = 0.85^5 ≈ 0.4437（ILLUSORY_MASTERY_DISCOUNT=0.15 的敏感点）
            "illusory_hits": 5,
            "discount_factor": (0.40, 0.49),
            "mastery": {"C": (0.1, 0.32)},
            "tc": {"python.loops": "pre_liminal"},
            "mastery": {"K": (0.0, 0.4), "P": (0.0, 0.4)},
            "interpretation_contains": ["伪自信"],
            "suggestion_contains": ["循环"],
        },
    },
    {
        "name": "liminal_in_progress_functions",
        "steps": [
            {"problem_id": "pf-l3-01", "score": 1.0, "self_confidence": 0.8},
            {"problem_id": "pf-l3-02", "score": 1.0, "self_confidence": 0.75},
            {"problem_id": "pf-l3-03", "score": 1.0, "self_confidence": 0.8},
            {"problem_id": "pf-l3-04", "score": 1.0, "self_confidence": 0.8},
            {"problem_id": "pf-l4-01", "score": 1.0, "self_confidence": 0.85},
        ],
        "expect": {
            # 3 次 L3+ 正确进 liminal，再 2 次不够 3 连 -> 停在 liminal（再答对 1 次）
            "tc": {"python.functions": "liminal"},
            "bkt": {"python.functions": (0.8, 1.0)},
            "dominant_bloom": "APPLY",
            "suggestion_contains": ["函数", "跨越"],
            "practice": ("python.functions", "APPLY"),
        },
    },
    {
        "name": "mixed_partial_credit",
        "steps": [
            {"problem_id": "pv-l2-01", "score": 1.0, "self_confidence": 0.8},
            {"problem_id": "pl-l3-01", "score": 0.4, "self_confidence": 0.6},
            {"problem_id": "pf-l2-01", "score": 1.0, "self_confidence": 0.7},
            {"problem_id": "ps-l3-01", "score": 0.9, "self_confidence": 0.5},
            {"problem_id": "pr-l2-01", "score": 0.9, "self_confidence": 0.8},
            {"problem_id": "pv-l3-01", "score": 0.5, "self_confidence": 0.7},
            {"problem_id": "pl-l4-01", "score": 0.0, "self_confidence": 0.4},
        ],
        "expect": {
            # 5 个 topic 各有作答；partial credit <0.6 记错 -> loops 0/2 最弱
            "tc": {"python.loops": "pre_liminal", "python.scope": "pre_liminal"},
            "bkt": {"python.loops": (0.0, 0.2), "python.variables": (0.1, 0.3)},
            "dominant_bloom": "UNDERSTAND",
            "illusory_hits": 0,
            "suggestion_contains": ["循环", "最弱"],
            "practice": ("python.loops", None),
        },
    },
    {
        "name": "misconception_scope_text",
        "steps": [
            {"problem_id": "ps-l2-01", "score": 0.0, "self_confidence": 0.6,
             "explanation_text": "为什么函数里改不了外面的x？"},
            {"problem_id": "ps-l3-01", "score": 0.0, "self_confidence": 0.5,
             "explanation_text": "全局变量和局部变量有什么区别？"},
            {"problem_id": "ps-l4-01", "score": 0.3, "self_confidence": 0.4,
             "explanation_text": ""},
        ],
        "expect": {
            # 解释文本命中 M8（全局/局部作用域混淆）两次：
            # discount = (1 - min(0.6*0.3, 0.3))^2 = 0.82^2 ≈ 0.6724
            "misc_hits": 2,
            "discount_factor": (0.65, 0.70),
            "tc": {"python.scope": "pre_liminal"},
            "suggestion_contains": ["作用域"],
        },
    },
    {
        "name": "calibration_boundary_learner",
        "steps": [
            # 三个伪自信检测边界，锁定 ILLUSORY_GAP_THRESHOLD / ILLUSORY_SELF_CONF_MIN：
            # 1) gap 0.45 < 0.5 -> 不命中（阈值若降为 0.4 会命中 -> 回归 FAIL）
            {"problem_id": "pl-l1-01", "score": 0.5, "self_confidence": 0.95},
            # 2) gap 0.65 但自评 0.65 < 0.7 -> 不命中（自评下限若降为 0.6 会命中）
            {"problem_id": "pl-l2-01", "score": 0.0, "self_confidence": 0.65},
            # 3) gap 0.55 且自评 0.75 >= 0.7 -> 命中（唯一一处）
            {"problem_id": "pl-l3-01", "score": 0.2, "self_confidence": 0.75},
            {"problem_id": "pl-l4-01", "score": 0.0, "self_confidence": 0.3},
        ],
        "expect": {
            "illusory_hits": 1,
            "discount_factor": (0.84, 0.86),
            "tc": {"python.loops": "pre_liminal"},
            "suggestion_contains": ["循环"],
        },
    },
]


def sequence_by_name(name: str) -> dict:
    for seq in SEQUENCES:
        if seq["name"] == name:
            return seq
    raise KeyError(f"未知黄金序列: {name}")
