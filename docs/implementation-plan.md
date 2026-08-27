# CogMirror 功能迁移详细实施方案（PersonalAGI → CogMirror）

> **上游文档**：`mac-mini-and-cogmirror-transfer.md` 第 2 部分（位于 PersonalAGI 仓库根目录，跨仓库故无相对链接）
> **日期**：2026-08-27
> **性质**：详细实施方案（PLAN），非已实施代码。所有"现状/源模式"均基于两仓库源码核实（READ），工作量/信心为估计（INFERRED，明确标注）。
> **修订记录**：2026-08-27 首版；同日经 PersonalAGI / CogMirror 双仓库源码交叉复核后修订（P3 补两个前置缺口与无状态视图设计、P4 补第 0 步证据采集入口、多处源模式细节勘误）。
> **适用约束**：本方案完全遵守 CogMirror 治理——GOVERNANCE 规则 4（不抽象化）、CLAUDE.md 硬规则（无 LLM、仅 numpy+scipy、判分确定性可复现）、两条线汇报（工程 vs 验证永不合并）。

---

## 0. 总览

从 PersonalAGI 抽取 **4 个算法移植（A1-A4）+ 2 个模式转移（B1-B2）**，全部服务于 CogMirror 现有 P0 主链路（做题 → 5D 更新 → 认知地图），不引入任何新外部依赖、不引入 LLM。

**阶段划分与依赖顺序（A3 基线先行，是关键决策）**：

| Phase | 内容 | 前置 | 工作量(估) | 价值 | 涉及新文件 |
|---|---|---|---|---|---|
| P1 | A3 回归评测基建（先建黄金基线） | 无 | 4-6h | 高（守护后续所有引擎改动） | `tests/golden/*`、`tests/test_golden_regression.py` |
| P2 | A1 自评置信度校准曲线 | P1 | 3-4h | 最高（升级 P0 伪自信核心） | `cogmirror/calibration.py` |
| P3 | A4 间隔衰减接线（复活 apply_decay） | P1 | 2-3h | 中高 | （改 2 文件，无新文件） |
| P4 | A2 misconception 闭环参数学习（含第 0 步证据采集入口） | P1 | 6-10h | 中高 | `cogmirror/misconception_tracker.py` + DB 新表/新列 |
| P5 | B1+B2 纵向档案 + 会话反思段 | P1 | 4-6h | 中 | `cogmirror/session.py`（可选） |
| **合计** | | | **≈20-30h**（2026-08-27 复核后上调：P3/P4 各补前置缺口工作量） | | |

**为什么 A3 先行**：P2/P3/P4 都改动引擎行为（discount、suggestion、misconception 权重）。若不先固化"当前行为为黄金基线"，无法判断这些改动是改进还是破坏。这直接对应 PersonalAGI `eval/gauntlet_regression.py` 的核心纪律——**固定黄金集，每次变更跑全量，FAIL 且历史曾有 PASS = 回归**。基线先行让后续每个改动都能被自动裁判。

---

## 1. 通用不变量（Guardrails，任何 Phase 不得违反）

1. **无 LLM**：产品路径不引入 LLM/OpenAI SDK；spike（`spike/`）保持独立，互不触碰。
2. **依赖不扩**：`pyproject.toml` 依赖维持 `numpy + scipy`（+dev: pytest）。新增纯标准库/纯 numpy 实现。
3. **不重命名 C/X**：CLI 仍显示 C=置信度、X=外部支架（PRD 8b 的"代码在 Phase 0.5 前不重命名"决定），本方案不在认知模型层改名。
4. **不引入调度/消息/向量基建**：A4 是同步计算（会话开始时算一次），不需要 APScheduler/CronTrigger/Telegram/Qdrant。
5. **DB 变更向后兼容**：新表一律 `CREATE TABLE IF NOT EXISTS`（沿用 `db.py:_SCHEMA` 模式）；不 alter 既有 3 表。
6. **测试纪律**：每个 Phase 新增测试，提交前 `pytest` 全绿（基线 = 迁移前全绿，README 2026-08-27 记录 259 项）；测试命名/风格沿用 `tests/test_*.py` + `make_obs` 这类 helper 约定。
7. **双线汇报**：README 的"工程线"记录"实现了什么"，"验证线"只记录"验证证实了什么"；本方案所有内容属于**工程线**，不改变验证线任何既有结论。
8. **改动有自测记录**：开发者自测数据标注"自测数据，不计入正式验证样本"（GOVERNANCE 规则 3）。

---

## 2. Phase 1 — A3 回归评测基建（先建基线）

### 2.1 目标
把 GOVERNANCE 规则 1-2（"完成必须可复现""验证需独立方法"）**自动化进 CI**：建立一组"合成学习者"黄金序列，任何引擎改动必须保持输出在容差内；否则标记回归。

### 2.2 源模式（PersonalAGI，READ）
`src/genesis/eval/gauntlet.py` + `gauntlet_regression.py` 的可移植核心：
- 固定黄金集，每次变更跑全量；
- 客观 scorer 逐 case 判 pass；
- aggregate = passed/(passed+failed)，skip 排除分母；
- 回归判定 = **本次 FAIL 且历史 25 条内曾有 PASS**；冷启动（从未 PASS）不报；恢复（PASS）撤回告警；
- **永不自动回滚，只告警 + 存幂等提案**。

### 2.3 现状（CogMirror，READ）
259 项单元测试（`tests/`，README 2026-08-27），覆盖引擎/CLI/判分/策略，但**无端到端行为回归**——BKT/MIRT/折扣任一改动，没有任何机制保证"认知地图输出自洽"（Phase 0 关卡"5D 数值合理"只是人工检查）。

### 2.4 设计

**文件与结构**：
- `tests/golden/sequences.py` — 黄金序列定义（Python，确定性）。每条：
  ```python
  GoldenSequence = dict
  # {
  #   "name": "overconfident_loop_learner",
  #   "steps": [  # 依次 replay 的观测
  #     {"problem_id": "pl-l1-01", "self_confidence": 0.9, "answer": "0"},
  #     ...
  #   ],
  #   "expect": {
  #     "mastery": {"K": (0.4, 0.6)},   # (min, max) 容差窗口
  #     "dominant_bloom": "APPLY",
  #     "tc": {"python.loops": "liminal"},
  #     "illusory_hits": 2,
  #     "suggestion_contains": ["循环", "L3"],
  #   },
  # }
  ```
- `tests/golden/baseline.json` — 当前引擎跑每条序列的**输出快照**（首次生成后提交为基线）。
- `tests/test_golden_regression.py` — runner：
  - 对每条序列：构造 `BeliefEngine` + `QuestionBank` + 初始状态，replay steps（用 `Observation` + `engine.update`，复用 `belief_engine` 公开接口，**不调用 CLI 交互**）；
  - 断言 `expect` 窗口（客观 scorer）；
  - 断言 baseline：本次 FAIL 且 baseline 中该 case 为 PASS → 标回归（输出告警，不 fail 整个 run？——**决定：首版 fail**，因为工程纪律 > 平滑，标注清楚）；
  - 冷启动排除：baseline 未收录的 case 不报回归；
  - `pytest -m regression` 标记（需在 `pyproject.toml [tool.pytest.ini_options]` 注册 markers，否则 pytest 8 会告警）。

**可复现性三细节（2026-08-27 复核补）**：
- 每条黄金序列的 `Observation.timestamp` 必须**显式固定**（`Observation` 默认 `datetime.now()`，涉时间窗/衰减的断言会不可复现）；
- baseline 断言与 `expect` 窗口一样**带容差**（scipy/BLAS 版本间可能有 1e-9 级微漂移，全等断言会碎）；
- baseline.json 存**数值摘要**（mastery/theta/建议 topic 等），不存整份 state 序列化，避免无关字段差异造成假回归。

**与 CLI 的耦合注意**：地图文案（suggestion/解读）在 `cli.py` 里。回归 runner 直接测 `next_suggestion()`、`map_interpretation()`、`suggested_practice()`（纯函数，可单测），避免驱动整个 `main()` 的输入流。

### 2.5 测试计划
- `tests/test_golden_regression.py`：≥5 条合成序列（全对/全错/伪自信密集/liminal 跨越/部分正确混合）；
- 自检用例：故意改动一个阈值（如 `ILLUSORY_GAP_THRESHOLD`），回归必须抓到差异（**这是验收的核心证伪测试**）。

### 2.6 验收标准（可证伪）
- `pytest -m regression` 在当前代码上全绿并生成 baseline；
- **DISPROVEN IF**：故意改动引擎后回归未报 FAIL；
- 基线更新必须带文档化 diff（改了什么、为什么），不允许静默覆盖。

### 2.7 风险与回滚
- 风险：黄金序列选取得太窄 → 回归覆盖面不足（缓解：5 条覆盖 5 个 topic × 主要行为模式）。
- 回滚：删除 `tests/golden/` + `test_golden_regression.py` 即可，不动产品代码。

### 2.8 决策关卡
- 通过：回归能抓到 seeded 破坏 → 进入 P2。
- 不通过：先修 runner 本身，不推进任何引擎改动。

---

## 3. Phase 2 — A1 自评置信度校准曲线

### 3.1 目标
把 P0 核心功能"伪自信检测"从**固定阈值 + 固定折扣**升级为**数据驱动的校准曲线**：按自评置信度分桶，学习每桶的真实答对率，用曲线斜率决定折扣。这是全方案**价值最高的单笔迁移**（直接升级核心卖点之一，且是 PRD 8b 中 X 维度的 L3 证据来源）。

### 3.2 源模式（PersonalAGI，READ）
`src/genesis/calibration/`（纯算法，零 LLM）：
- `bucket_confidence(conf)`（types.py）— clamp 到 `[0, 0.999]`，`int(c*10)/10` 生成 0.1 宽桶；
- `CalibrationCurveComputer.compute(domain)`（curves.py）— 取 `(confidence_bucket, correct)` 配对 → 分桶 → 每桶 `actual_rate = Σcorrect/n` → `predicted = 桶中点` → `correction_factor = actual_rate / predicted`（predicted=0 时取 1.0）；
- `compute_ece / compute_mce`（metrics.py）— 加权绝对误差、最大桶误差；
- 多维结构：**每 domain 一张曲线表**（CogMirror 的 domain = 全局 / 未来 per-skill）。
- 研究 agent 明确建议：PersonalAGI 的 calibration **无平滑**，但 CogMirror n 很小（单用户、桶稀疏），**必须加 Laplace 平滑**（Laplace 来自 `learning/procedural`：`(success+1)/(success+failure+2)`）——这是对源模式的必要本地化。

### 3.3 现状（CogMirror，READ）
`belief_engine.py:38-41`：`ILLUSORY_GAP_THRESHOLD=0.5`、`ILLUSORY_SELF_CONF_MIN=0.7`、`ILLUSORY_MASTERY_DISCOUNT=0.15`（固定）。Step 7（`belief_engine.py:219-233`）命中后用固定 0.15 折扣 `C.discount_factor`。`responses` 表已存 `self_confidence` + `score`，**曲线可直接从 responses 重算，无需新表**。

### 3.4 设计

**新文件 `cogmirror/calibration.py`**（纯 numpy，无新依赖）：
```python
def bucket_confidence(conf: float) -> str      # 移植 PersonalAGI types.py
@dataclass
class CalibrationCurve:
    bucket: str; n: int; correct: int
    predicted: float; actual_rate: float       # Laplace: (correct+1)/(n+2)
    correction_factor: float                   # actual_rate/predicted，predicted==0→1.0
class CalibrationCurveComputer:
    def compute(records: list[dict], bucket_width=0.1) -> list[CalibrationCurve]
        # records = [{self_confidence, score}]；返回按桶排序曲线
    def expected_accuracy(curves, claimed_conf, min_n=5) -> float | None
        # 查桶；桶样本 < min_n 返回 None（数据不足，退回固定折扣）
def compute_ece(curves) -> float                # 加权 |actual-predicted|
```

**改动 `cogmirror/belief_engine.py`**：
- `BeliefEngine` 增加 `set_calibration(curves)` 入口（由 CLI 在加载历史后注入，从 `responses` 重算）；
- Step 7 命中伪自信时：`expected = expected_accuracy(curves, self_confidence)`；`expected is not None` → `discount = 1.0 - expected`（clamp 到 `[0.05, 0.5]` 防过激），否则退回固定 0.15；
- 检测阈值（0.5/0.7）**保持不动**（这是"命中"定义，不是"折扣幅度"；改它需要更多证据）。

**改动 `cogmirror/cli.py`**（小）：`main()` 加载历史后 `engine.set_calibration(...)`；地图加一行可选指标 `自评校准度（ECE）`，样本 <5 时诚实标注"数据不足"。

### 3.5 数据模型
无 schema 变更——曲线是 `responses` 表的派生视图。**理由**：单用户 n 小，重算成本可忽略，避免维护第二份聚合表与响应表的一致性（PersonalAGI 用独立表是因为多 domain + 高频写；CogMirror 不满足该前提，硬搬独立表反而是过度设计）。

### 3.6 测试计划
`tests/test_calibration.py`：
- 桶划分/Laplace 数学（n=1 → 2/3 等边界）；
- correction_factor 语义（自评高于实绩 → factor<1）；
- `expected_accuracy` 的 min_n 回退；
- 合成"过度自信学习者"序列：曲线驱动折扣使 `C.mastery_prob` 下降**大于**固定折扣 → 校准信息更敏感；
- ECE 单调性（合成欠校准 → ECE 大）。

### 3.7 验收标准（可证伪）
- 曲线驱动在样本 ≥5 时生效、样本不足时优雅回退固定折扣；
- 259 项既有 + 新增测试全绿；
- **DISPROVEN IF**：合成过度自信学习者身上，曲线驱动导致 C 维度比固定折扣更不敏感（校准变差）→ 设计方向错，回退。

### 3.8 风险与回滚
- 风险：单用户数据少，桶稀疏 → 多数桶走回退路径（缓解：min_n 设 5 + Laplace 兜底；这是"数据不足诚实回退"，非故障）。
- 回滚：删 `calibration.py` + 还原 `belief_engine.py` Step 7 两处 + 还原 cli 两行。

### 3.9 决策关卡
- 通过：合成序列上 ECE 改善 & 全绿 → 进入 P3。
- 不通过：诊断是"桶太稀疏"还是"算法语义错"，对症修，不硬推。

---

## 4. Phase 3 — A4 间隔衰减接线（复活死代码 apply_decay）

### 4.1 目标
把"一句话建议"从**做什么**升级到**何时做**：识别"曾经掌握、正在遗忘"的 skill，给出复习时机。**同时**把 `bkt.py:137 apply_decay` 从死代码接入产品路径（这是"built ≠ wired"的活实例）。

### 4.2 源模式（PersonalAGI，READ）
`scheduler/user_jobs.py` + `campaigns/` 的核心模式是"**基于状态的再触达时机**"（campaign 状态机 + 内部指针 + 幂等认领 + 程序化 precheck 门）。但 CogMirror 是本地同步 CLI，**不移植** APScheduler/CronTrigger 调度基建与 Telegram 分发渠道（Telegram 是出站渠道而非调度依赖）——只取"何时该再触达"的判定逻辑，退化为会话开始的一次同步计算。注：上游文档 A4 把 scheduler/campaigns 列为正面迁移源，本方案降级为"只取判定逻辑"是**由 CogMirror 产品形态（本地 CLI、单用户）决定的取舍**，非引用可移植性评级。

### 4.3 现状（CogMirror，READ，2026-08-27 交叉复核）
- `bkt.py:137 apply_decay(skill_id, days_since_last)` + `get_decay_constant(skill_id)`（默认 30 天）已实现；**产品路径零调用者**（仅 `test_engines.py:39` 引用）。
- `cli.py:next_suggestion` 顺序：liminal 优先 → BKT 最弱 → 全好开新 topic。无"遗忘/复测"概念。
- `responses` 表有 `created_at`，可算 skill 最近作答间隔。
- **前置缺口 1（P3 必须先解决）**：BKT 状态（`engine.l1` 的 skill 模型）**不持久化**——恢复路径只 `engine.set_history`（`cli.py:581`，仅喂 MIRT），`l1` 每会话从零开始。因此"peak_mastery ≥ 0.7（曾掌握）"在重启后**无从计算**，不能直接读当前 `get_bkt_mastery`。
- **前置缺口 2（双重衰减陷阱）**：`apply_decay` 是**原地乘法**（`bkt.py:146`：`model.p_mastered *= e^(-days/τ)`）。若每次会话开始都对同一 skill 按"距上次作答天数"调用，会在已衰减值上再乘全量因子 = 复合指数衰减。历史重放也不能走 `l1.update`（那是学习更新，不是时间衰减）。

### 4.4 设计

**先解决两个前置缺口**：峰值来源（缺口 1）用**独立临时 BKTModel 历史重放**推导；衰减计算（缺口 2）改为**无状态视图**（直接按公式算，不调 `l1` 的原地 `apply_decay`，规避复合衰减）。

**改动 `cogmirror/belief_engine.py`**：新增方法
```python
def peak_mastery_from_history(self, history: list[dict]) -> dict[str, float]:
    """从 responses 重放推导每个 skill 的历史峰值掌握概率（只读，不改 l1）：
    对每个 skill 取其作答序列，用独立的临时 BKTModel 逐条 update（学习更新，
    非时间衰减），记录过程中 P(L) 最大值。可从 DB 幂等重算（覆盖缺口 1）。"""

def decayed_mastery_view(self, history: list[dict]) -> dict[str, tuple[float, float]]:
    """无状态衰减视图：返回 {skill_id: (peak, decayed)}，不改 l1 状态。
    decayed = peak · e^(-days_since_last/τ)（直接算公式，不经 l1.apply_decay
    的原地乘法，规避缺口 2 的复合衰减）；days_since_last 用该 skill 最近一条
    response.created_at 距今天数。峰值来自 peak_mastery_from_history。"""
```

**改动 `cogmirror/cli.py`**：
- `main()` 加载历史后调用 `engine.decayed_mastery_view(history)`（纯只读，`l1` 会话内学习更新语义不变）；
- `next_suggestion` 增加**复测分支**（在 liminal 之后、最弱 BKT 之前）：
  - 条件：`peak_mastery ≥ 0.7`（曾掌握，来自历史重放视图而非当前 `l1`）且 `decayed_mastery < 0.55` 或相对峰值跌幅 ≥0.15；
  - 文案带"何时"："「循环」上次 42 天前练过，掌握概率从 82% 掉到 50%——建议先做 3 道复测题，趁遗忘前巩固。"
- 地图新增一行 `[复习提示]`：列出所有命中复测条件的 skill + 天数。
- `suggested_practice` / `practice_command` 同步支持复测分支（返回该 topic + level=None）。

**时区决定**：CogMirror 是本地单用户 CLI，`datetime.now()` 本地时区即可；在代码注释里明确（区别于 PersonalAGI 的 UTC 规则——那是多时区服务器要求，不适用）。

### 4.5 数据模型
无 schema 变更（时间来自 `responses.created_at`）。

### 4.6 测试计划
- `tests/test_engines.py` 增补：`peak_mastery_from_history` 重放正确性（历史峰值 ≠ 当前会话值，覆盖缺口 1）；`decayed_mastery_view` 的间隔计算 + 衰减数学（30 天 → e^-1）+ **幂等性**（同一历史连续调用两次结果相同，锁定缺口 2）；
- `tests/test_cli.py` 增补：模拟"8 天前掌握、42 天未练"历史 → `next_suggestion` 返回复测文案、`[复习提示]` 出现、`suggested_practice` 返回正确 (topic, None)。

### 4.7 验收标准（可证伪）
- 衰减语义接入产品路径（`decayed_mastery_view` 被 `cli.py` 调用；`bkt.py:apply_decay` 原地接口保留，但产品路径衰减一律走无状态视图）；
- 复测分支只在"曾掌握 + 显著衰减"时触发，**不干扰**正常 liminal/最弱分支；
- **DISPROVEN IF**：新序列（连续练习无间隔）出现误报"复习提示"（应为空）。

### 4.8 风险与回滚
- 风险：衰减阈值过激导致建议反复横跳（缓解：复测需峰值≥0.7 + 跌幅≥0.15 双条件，保守）。
- 回滚：还原 `belief_engine.py` 方法 + `cli.py` 三处。

### 4.9 决策关卡
- 通过：无间隔学习者不误报 + 有间隔学习者正确提示 → 进入 P4。

---

## 5. Phase 4 — A2 misconception 闭环参数学习

### 5.1 目标
把 misconception 检测从**静态库 + 固定置信度 0.6**升级为**证据驱动权重**：每条 misconception 的成功/失败计数随证据更新，权重用 Laplace 置信度，反复失败的 misconception 权重上升（更被重视）、被克服的下降。这是"给导师模型做校准"——直接喂核心闭环。

### 5.2 源模式（PersonalAGI，READ）
`learning/procedural/` 的可移植核心（研究 agent 明确点名）：
- **Laplace 置信度**：`confidence = (success+1)/(success+failure+2)`；
- **生命周期只靠证据**：失败积累 → 降档/隔离（降档为复合条件：`failure_count ≥ success+3` **且** `total_hits ≥ 3`；`conf < 0.3` 且 `s+f ≥ 3` → quarantine）；**无主动"过时"判定**（无 TTL/时间衰减，仅手动 `deprecated` 软删），靠失败降档隐式淘汰；
- **结果对账**：`PredictionReconciler`（位于 `calibration/reconciler.py`）把预测 join 到 outcome——CogMirror 版 = 把"某次 misconception 命中"join 到"后续同 skill 表现"判成功/失败。
- 不可移植部分（不搬）：LLM 分类、embedding 去重、LLM 提取。

### 5.3 现状（CogMirror，READ）
- `content/misconceptions.py`：8 条（M1-M8），静态 `MisconceptionEntry`；
- `belief_engine.py:280-295`：`_detect_misconception` 命中返回 `MisconceptionHit(confidence=0.6)` 固定；
- Step 5（`belief_engine.py:202-207`）：命中 → `C.discount_factor *= (1 - min(0.6*0.3, 0.3))`。
- **前置缺口 A（P4 第 0 步，2026-08-27 交叉复核发现）**：**misconception 检测在产品路径从未触发**--`cli.py:304` 构造 Observation 时恒传 `explanation_text=""`，而 `_detect_misconception`（`belief_engine.py:282-284`）对空文本直接返回 None。即 Step 5 的"命中 -> 折扣"链路在真实使用中是**死的**（仅测试直接构造带 explanation_text 的 Observation 才触发），P4 的证据闭环没有天然数据来源。
- **前置缺口 B（对账原料缺失）**：`responses` 表虽存 `user_answer` + `skill_id`，但 (a) 检测输入是**解释文本**（`explanation_text`）而非 `user_answer`，两者不是一回事；(b) responses 表**未存 misconception 命中记录**（无 misc_id 字段，`MisconceptionHit` 只活在内存 state 里），reconcile 无从 join。
- 结论：**P4 第 0 步必须先建证据采集入口**，否则闭环无米下锅（原估 4-6h 未计此项）。

### 5.4 设计

**第 0 步（新增，解决缺口 A/B）：证据采集入口**
- **A 路（采集解释文本）**：选择题答错后追加一个可选追问--「用一句话说说你为什么选这个？」（直接回车跳过）；fill/code 答错后同样追问。答案填入 `Observation.explanation_text`（改 `cli.py` 构造处，替代恒空串）。跳过是常态、解释是例外--不强制，避免打扰。
- **B 路（落库命中记录，解决对账原料）**：`responses` 表加一列 `misc_id TEXT`（`CREATE TABLE IF NOT EXISTS` 不可改既有表结构--用 `ALTER TABLE ... ADD COLUMN` + "列不存在才加"的启动检查，单用户本地库可行）；`save_response` 把当次 `MisconceptionHit.misc_id` 写入。reconcile 从此列 join，不再依赖内存 state。
- 判分/检测逻辑（关键词库 + `belief_engine` Step 5）不动，只补输入与落库。

**新文件 `cogmirror/misconception_tracker.py`**：
```python
class MisconceptionTracker:
    def __init__(self): self._evidence: dict[str, dict]  # {misc_id: {success, failure, last_updated}}
    def load(self, rows: list[dict]) -> None            # 从 DB 恢复
    def confidence(self, misc_id: str) -> float          # (s+1)/(s+f+2)
    def record_success(self, misc_id); record_failure(self, misc_id)
    def reconcile(self, history: list[dict]) -> None
        # 对账（移植 PredictionReconciler 模式，零 LLM）：
        # 遍历历史，对每次 detected misconception（有 misc 证据的题）：
        #   找同 skill 的下一条响应：score≥0.6 且未重触发 → success
        #   否则（重触发 或 score<0.6）→ failure
    def quarantined(self, misc_id) -> bool              # conf<0.3 且 s+f≥3
```
**DB 新表**（`db.py` `_SCHEMA` 追加，`IF NOT EXISTS`）：
```sql
CREATE TABLE IF NOT EXISTS misconception_evidence (
    misc_id TEXT PRIMARY KEY,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL
);
```
`db.py` 增 `save_misconception_evidence` / `load_misconception_evidence`。

**改动 `cogmirror/belief_engine.py`**：
- 构造器接收 `MisconceptionTracker`（CLI 从 DB 加载后注入）；
- Step 5：`MisconceptionHit.confidence = tracker.confidence(misc_id)`（替代固定 0.6）；
- 会话结束（CLI 收尾）调 `tracker.reconcile(history)` + `db.save_misconception_evidence`。

### 5.5 测试计划
- `tests/test_cli.py` 增补（第 0 步）：答错后出现追问提示、回车跳过不阻塞、解释文本进入 Observation 并能触发检测（覆盖缺口 A）；`misc_id` 落库 + 从 DB 恢复后 reconcile 可用（覆盖缺口 B）。
- `tests/test_misconception_tracker.py`：Laplace 数学、reconcile 的成功/失败分支、quarantine 阈值、DB 往返。
- `tests/test_belief_engine.py` 增补：固定 0.6 → 证据权重后，C 折扣随证据变化。

### 5.6 验收标准（可证伪）
- 反复触发同一 misconception（如 3 次失败）→ 权重 > 0.6、C 折扣更深；被克服后权重回落；
- **DISPROVEN IF**：reconcile 对"命中后首条同 skill 正确响应"判成 failure（对账语义错）；
- 259 项既有测试全绿（`_detect_misconception` 返回结构不变，仅 confidence 来源变）。
- **第 0 步验收**：真机答错一题并输入含关键词的解释 -> 地图 C 维度出现 misconception 命中（P4 前该链路从未在产品路径触发过）。

### 5.7 风险与回滚
- 风险：对账窗口定义不当（同 skill 下一条响应可能跨很长的真实时间）→ 用"同一会话内"或"间隔 ≤N 天"限定（首版：同一会话内）。
- 回滚：删 tracker + 还原 Step 5 + 删表语句（表可留，数据无意义）；第 0 步独立回滚 = 还原追问与落库改动，检测链路回到"待输入"状态即可。

### 5.8 决策关卡
- 通过：证据驱动的权重变化符合直觉 & 全绿 → 进入 P5。
- 不通过：先查对账窗口，不硬推。

---

## 6. Phase 5 — B1+B2 纵向档案 + 会话反思段

### 6.1 目标
- **B1**：让认知地图从"本次快照"变成"纵向连续"——下次回来时主动浮现"上次卡在哪、这几轮趋势如何"（借 PersonalAGI 主动召回/4 层记忆的**模式**，不搬基建）。
- **B2**：把"整体解读段"升级为"本次会话变化 + 为什么 + 单一下一步"的反思段（借 PersonalAGI `reflection/` 的"证据锚定洞察"模式）。

### 6.2 源模式（PersonalAGI，READ）
- `memory/` 4 层（L1 essentials 会话开始注入 / L2 语义召回每 prompt 触发 / L3 深检索 / L4 知识管线）：CogMirror 借的是"相关历史主动浮现"的**模式**（L1/L2 的效果），实现退化为会话开始时从已有 SQLite 聚合上次卡点 + 跨会话趋势，**不引入 Qdrant/向量**（注：源仓库 L1/L2 职责不同，此处合并表述为目标效果而非逐层对应）；
- `reflection/`：定时 + 证据锚定的洞察抽取（CogMirror 版：确定性启发式，零 LLM）。

### 6.3 现状（CogMirror，READ）
- `main()` 已用 `load_latest_state` 恢复上次状态，`_welcome_progress` 显示上次主导层级 + liminal 进度；`_map_delta_lines` 已做"上次会话末 vs 本次末"的 K/P/S 对比（**跨会话 delta 已部分存在**——README 2026-08-27 亦已列「与上次相比」对比段为近期功能）；
- 缺口：无"跨多会话趋势"、无"上次卡点"显式召回、解读段无"本次会话变了什么"的专项复盘。

### 6.4 设计

**B1 — `cogmirror/session.py`（新，纯函数）**：
```python
def last_session_struggles(db, user_id) -> list[str]
    # 上次会话（按 belief_snapshots 最后快照时间窗口切分）答错/部分正确的 skill
def multi_session_trend(db, user_id, n=3) -> dict[str, float]
    # 从 belief_snapshots 按时间聚合最近 n 个会话末的 K/P/S mastery 均值 → 趋势
```
**CLI 改动**：`main()` 欢迎行扩展——`上次卡住：循环(range 边界)、作用域`；地图新增 `[近几次趋势]` 段（仅 ≥2 会话且样本足时显示，否则诚实标注"数据不足"）。**无需新表**：从 `responses`（卡点）+ `belief_snapshots`（趋势）聚合即可（避免维护 session 边界表的理由同 P2——单用户聚合成本可忽略）。

**B2 — 扩展 `map_interpretation`（`cli.py:407`）**：在现有综合段后追加"本次会话"反思句：
- 本次变化：从 `_map_delta_lines` 结果取前 2 项（"本次 K +12%、P -5%"）；
- 为什么：绑定具体证据（"K 上升来自 4 道 L1-L2 全对；P 回落因 2 道写码题只对一半"）；
- 下一步：引用 `next_suggestion` 的动作 + 一句理由（"因为「循环」处于 liminal 跨越中，合意困难原则建议继续巩固而非学新概念"）。
- 保持确定性：全部从句式模板 + 既有数据生成，**零 LLM**。

### 6.5 测试计划
- `tests/test_session.py`（或并入 `test_cli.py`）：`last_session_struggles` / `multi_session_trend` 的聚合正确性；B2 反思句在"有 delta"“无 delta”“样本不足”三态下的输出。

### 6.6 验收标准（可证伪）
- 第 2 次会话起，地图出现趋势/卡点信息；样本不足时诚实标注不臆造；
- 反思句每个断言都能回溯到具体 responses/snapshots 证据（**可复核**，这是"证据匹配声明范围"的体现）；
- **DISPROVEN IF**：反思句在无 delta 时仍声称"本次有变化"。

### 6.7 风险与回滚
- 风险：文案膨胀削弱"一句话"克制感（缓解：趋势/反思各限 1-2 行；PRD"一句话建议"不动）。
- 回滚：删 `session.py` + 还原 cli 两处 + 还原 `map_interpretation`。

### 6.8 决策关卡
- 通过：纵向信息可复核、文案克制、全绿 → 全部 Phase 完成。

---

## 7. 跨 Phase 验证计划与治理

- **每 Phase 三步验证**：① 单元测试全绿；② 黄金回归（P1 起）无意外 FAIL；③ 开发者自测跑真实 CLI 一遍，记录"自测数据，不计入正式验证样本"。
- **治理规则映射**：
  - 规则 1（不虚标）→ P1 回归 = 自动化的"完成可复现"；
  - 规则 2（双指标）→ 单元测试 + 黄金回归双通道；
  - 规则 6（数据事故污染范围）→ P4 的 DB 新表不影响既有表；若 reconcile 写错污染了 `C.discount_factor` 语义，按规则 6 列出受影响结论（本阶段无正式验证结论，风险低，但机制上沿用）；
  - 规则 7（潜变量/观测分层）→ A1 明确"置信度是 L3 证据，不是 5D 维度"，校准曲线只作用于 `C.discount_factor`（观测层），不重命名 C/X（PRD 8b 决定）。

---

## 8. 明确不做（SOMEDAY 登记建议）

以下来自分析文档 C 类 + 本方案设计中被否决的选项，按 GOVERNANCE 规则 4 写入 SOMEDAY（附触发条件，不做）：

| 事项 | 触发条件（满足才重新讨论） |
|---|---|
| per-skill 校准曲线（全局曲线先上） | 单用户 response ≥300 且分 topic 桶样本 ≥20 |
| 对话式 LLM 进产品判分链路 | 先有 P2 校准 + P1 回归作安全轨；且 Phase 0.5 spike 多人验证通过 |
| 独立 session_summaries 表 | 聚合查询出现可测的性能问题（当前规模不会） |
| Telegram/每日一题分发 | Phase 2 正向信号 + 用户明确要求 |
| 向量记忆 / Qdrant | 学习者数量进入多用户阶段 |

---

## 9. 置信度与工作量汇总

| 项 | 置信度 | 理由 / 剩余风险 |
|---|---|---|
| P1 回归基建契合治理 | 90% | 直接操作化规则 1-2；与现有 259 项测试体系正交；剩余风险=黄金序列覆盖面 |
| P2 校准算法正确性 | 85% | 源算法纯算法可搬（研究 agent 确认）；Laplace 平滑为必要本地化；剩余风险=单用户桶稀疏 |
| P3 复测分支价值 | 75% | 死代码确认；BKT 不持久化 + 原地衰减陷阱两前置缺口已补设计（无状态视图），剩余风险 = 历史重放峰值语义（重放学 BKT ≠ 真实历史状态） |
| P4 misconception 对账语义 | 65% | 检测链路在产品路径从未触发（缺口 A）+ 命中记录未落库（缺口 B），已补第 0 步（答错追问 + misc_id 落库）；reconcile 窗口仍不确定，首版限同一会话。置信度低于原评估的原因：采集交互新增，真实用户是否输入解释未知（可能长期零证据 -> 闭环空转） |
| P5 纵向/反思段 | 80% | 复用已有 delta 机制，改动小；剩余风险=文案克制 |
| 总工作量 | - | 20-30h（INFERRED，2026-08-27 交叉复核后上调：P3 双缺口 + P4 第 0 步采集入口） |

**下一步**（本方案经确认后）：从 P1 开始实施。P1 完成并过关卡后，P2/P3/P4 均可并行或串行推进（三者都只依赖 P1 基线）。

---

## 10. 文档归属说明

本方案位于 CogMirror 仓库 `docs/implementation-plan.md`（2026-08-27 从 PersonalAGI 仓库根目录移入，供在 CogMirror 内执行）。上游分析文档 `mac-mini-and-cogmirror-transfer.md` 保留在 PersonalAGI 仓库根目录（跨仓库，无相对链接）。本文件已在 README「文档清单与阅读顺序」登记。
