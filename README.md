# CogMirror（工作代号）

一个帮成年自学者在学 Python 的过程中，随时看清自己真实认知状态（哪里真会、哪里是伪自信、卡在哪个概念）的 AI 学习教练。从 ECOS 项目选择性迁移并收缩重启，见 [MIGRATION.md](./MIGRATION.md)。

## 当前状态（两条线分开汇报，见 GOVERNANCE 规则5）

- **工程线**：Phase 0 最小链路已可运行--做题（静态题库 + 确定性 partial credit 判分）-> 5D 信念更新（K/P/S/C/X）-> 命令行认知地图（5 维状态 / Bloom 六层 / 伪自信标注 / 临界概念标注 / 一句话建议），81 项自动化测试通过。
- **验证线**：无。尚未经任何真实用户验证（含开发者自测也未开始），不声称任何"有效"结论。

## 快速开始

```bash
pip install -e ".[dev]"
pytest                      # 跑测试
cogmirror                   # 交互式答题 + 认知地图（或 python -m cogmirror.cli）
cogmirror --map-only        # 只看地图不答题
```

## 代码结构

```
cogmirror/
  belief_state.py   5D 状态数据结构（迁移自 ECOS，简化）
  belief_engine.py  信念引擎：BKT + 5D MIRT + Bloom + TC + 伪自信检测
  bkt.py            BKT 知识点掌握度更新（直接迁移）
  mirt.py           5D MIRT MAP 估计（直接迁移）
  tc.py             临界概念（liminal）状态机（直接迁移）
  questions.py      静态 Python 题库 + 确定性判分（新写）
  content/          ECOS 已验证内容库：Bloom 目标 / 错误模式 / 临界概念
  policy/           LinUCB / Thompson / A/B 框架（P1 预留，未接入主流程）
  db.py             SQLite 持久层（成人向合规字段，重新设计）
  cli.py            Phase 0 命令行界面
docs/theory/        理论参考（自 ECOS 迁移，头部有局限标注）
```

## 文档阅读顺序

1. **[PRD.md](./PRD.md)** -- 做什么、不做什么的唯一真相来源
2. **[MIGRATION.md](./MIGRATION.md)** -- 从 ECOS 迁移什么、怎么改造、什么坚决不带
3. **[ROADMAP.md](./ROADMAP.md)** -- Phase 0 搭建 -> 1 自测 -> 2 真实用户验证 -> 3 视结果
4. **[GOVERNANCE.md](./GOVERNANCE.md)** -- 6 条硬规则（防虚标、双指标验证等）
5. **[LESSONS-FROM-ECOS.md](./LESSONS-FROM-ECOS.md)** -- 迁移背后的教训背景

## 一句话共识

**这是一个单一垂直、面向成年自学者、验证优先于抽象的独立产品，不是ECOS的延伸--直到Phase 2给出明确的正向信号之前，任何"做得更通用一点"的想法都先记进[SOMEDAY.md](./SOMEDAY.md)，不要动手做。**
