# 项目文档入口

这是新项目（工作代号 **CogMirror**，编程学习认知教练）的起始文档集，基于对ECOS项目的深度复盘产出。建好新项目目录后，把这几份文档放进repo根目录，按下面的顺序阅读和执行。

## 当前状态（两条线分开汇报，见 GOVERNANCE 规则5）

- **工程线**：Phase 0 静态题库最小链路可运行——做题（静态题库 + 确定性 partial credit 判分）→ 5D 信念更新（K/P/S/C/X）→ 命令行认知地图（5 维状态 / Bloom 六层 / 伪自信 / 临界概念 / 一句话建议）。认知地图三大核心元素（伪自信 / 临界概念 / 一句话建议）已全部做成"做题当下的实时信号"。题库 51 题（5 个 topic × Bloom L1-L6 全覆盖，2026-08-27 补齐 L5/L6），**356 项自动化测试通过**。近期功能：29 道选择题逐选项讲解、错题重练 `--review`、认知地图「与上次相比」对比段、返回用户进度概览、**P1 黄金回归基建**（7 条合成学习者序列 + 数值基线快照，引擎行为变更自动判回归，2026-08-28，见 [docs/implementation-plan.md](./docs/implementation-plan.md) 第 2 节）、**P2 自评校准曲线**（伪自信折扣从固定 0.15 升级为按自评分桶的历史真实答对率驱动 + 地图「自评校准度（ECE）」行，2026-08-31，见方案第 3 节）、**P3 间隔衰减复测**（从作答历史重放掌握峰值 + 无状态遗忘曲线，「曾掌握、正在遗忘」的 topic 给复测建议与 [复习提示]，2026-08-31，见方案第 4 节）、**P4 misconception 证据闭环**（答错后可选追问「为什么这么答」补齐检测输入（P4 前该链路在产品路径从未触发），命中落库 + 会话末与同 skill 后续表现对账，命中置信度从固定 0.6 升级为 Laplace 证据驱动，地图新增 [误解点] 段，2026-08-31，见方案第 5 节）、**P5 纵向档案 + 会话反思段**（快照时间聚类切会话，返回用户欢迎行浮现「上次卡住」，地图新增 [近几次趋势] 段（≥2 会话末 K/P/S 首末对比），整体解读段追加「本次变化 + 为什么」证据锚定反思句，2026-08-31，见方案第 6 节）、**Web UI 学生端**（2026-09-01：`python -m cogmirror.webui` 起本地服务，浏览器答题——代码题 textarea 随意改行、答错追问解释、认知地图 HTML 渲染 + 建议练习循环，零新依赖（stdlib http.server + 无框架单页，形态对齐 ECOS `web/student/`；产品化 React UI 见 SOMEDAY 第 9 项））。自测记录见 [SELF-TEST-CHECKLIST.md](./SELF-TEST-CHECKLIST.md)
- **验证线**：暂无正式真实用户验证结论，当前全部为开发者自测（数据不计入正式样本）。Phase 0.5 对话式诊断 spike 已收口（2026-08-26，三轮 n=1 方向一致：对话能力估计 ≈ 题库锚点或略低，无系统性高估）——**P0 判分仍以确定性静态题库为核心，对话式诊断作增强/技术预研**。真人测试**推迟**：当前 CLI 还不够好用、UI 不够美观，等项目更完善后再交给身边人试

## 快速开始

```bash
pip install -e ".[dev]"
pytest                      # 跑测试（当前 356 项）
pytest -m regression        # 只跑黄金回归（引擎行为变更必跑；基线更新用 python -m tests.golden.update_baseline，须带文档化 diff）
cogmirror                   # 交互式答题 + 认知地图（或 python -m cogmirror.cli）
python -m cogmirror.webui --open   # 浏览器答题（本地 Web UI，代码题可随意改行）
```

第一次运行会打印上手说明。常用命令：

| 命令 | 作用 |
| --- | --- |
| `cogmirror` | 做一组题（默认 10 道）+ 认知地图 |
| `python -m cogmirror.webui`（或 `cogmirror-web`，或**双击 `start-webui.command`**） | 浏览器答题 + 认知地图（数据与 CLI 共用同一 SQLite，`--open` 自动开浏览器） |
| 欢迎页「切换用户」 | 给别人试用：让 TA 用自己的名字开新用户，答题数据完全隔离、互不污染（对应 URL `?user=名字`） |
| `cogmirror --questions 3` | 本次做 3 道题 |
| `cogmirror --topic python.loops --level L3` | 只练指定 topic / 层级（L1-L6 或层名） |
| `cogmirror --map-only` | 只看地图不答题 |
| `cogmirror --review` | 重练全部错题（最近一次得分 < 60% 的题） |
| `cogmirror --export` | 导出自己的全部数据为 JSON（合规：可导出） |
| `cogmirror --delete` | 删除自己的全部数据（合规：可删除，需输入 DELETE 确认） |
| `cogmirror --help` | 全部命令与参数 |

### 作答输入说明

- **答题前自评**：每题先问"你有多大把握答对"（0-100，直接回车跳过）
- **选择题**：输入选项编号（0-N）；输错会重问，不会默默判错
- **填空题**：直接输入答案（回车视为未答）
- **写码题**：输入完整函数定义，写完单独一行输入 `END` 结束
- **中途退出**：Ctrl-C / Ctrl-D，已答的题都会保存，下次接着看地图

## 文档清单与阅读顺序

1. **[PRD.md](./PRD.md)** —— 先读这个。产品定位、目标用户、MVP范围、明确的Out of Scope、成功指标。这是"做什么、不做什么"的唯一真相来源。
2. **[MIGRATION.md](./MIGRATION.md)** —— 从ECOS代码仓具体要迁移哪些文件、怎么改造、哪些坚决不带走，附带执行清单。
3. **[ROADMAP.md](./ROADMAP.md)** —— 分阶段计划（Phase 0搭建→Phase 1自测→Phase 2真实用户验证→Phase 3视结果而定），每个阶段有明确的决策关卡，不是无脑往前走的时间表。
4. **[GOVERNANCE.md](./GOVERNANCE.md)** —— 6条从ECOS真实教训提炼的硬规则（不虚标、双指标验证、测试者不能是开发者本人、抽象化要有日历约束等）。建议同步进AI协作工具（如Claude Code）的项目级指令文件里，让每次协作都自动遵守。
5. **[SOMEDAY.md](./SOMEDAY.md)** —— 暂缓事项清单（跨领域认知诊断、双Agent重新引入、插件化等），每项都写明了触发条件，不满足条件前不重新讨论。
6. **[REFERENCES-dialogue-assessment.md](./REFERENCES-dialogue-assessment.md)** —— Conversation-Based Assessment / 对话知识追踪相关文献整理，支撑Phase 0.5的技术方案设计。
7. **[docs/implementation-plan.md](./docs/implementation-plan.md)** —— PersonalAGI 功能迁移详细实施方案（P1-P5 执行蓝图，含 A1-A4/B1-B2 六个特性的文件级设计、测试计划、验收标准与决策关卡）。执行时按 Phase 推进，每个 Phase 过关卡后再进入下一个。
8. **[docs/CogMirror-项目完整流程文档.md](./docs/CogMirror-项目完整流程文档.md)** -- 全项目技术与业务链路的通读总结（理论依据 / 题库与判分 / 八步引擎 / 地图机制 / 数据合规 / ECOS 迁移决策），目标是不看代码也能理解「做了什么、为什么这么做、每一步怎么算出来的」。

## 一句话共识（避免开工后又飘回ECOS的老路）

**这是一个单一垂直、面向成年自学者、验证优先于抽象的独立产品，不是ECOS的延伸——直到Phase 2给出明确的正向信号之前，任何"做得更通用一点"的想法都先记进`SOMEDAY.md`，不要动手做。**
