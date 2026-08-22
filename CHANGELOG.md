# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与[语义化版本](https://semver.org/lang/zh-CN/)。
版本号自 v0.1.0 重新开始（不继承 ECOS 的 v0.9x）。条目用中文，风格与 git 提交信息一致（范围前缀 + 描述）。

## [Unreleased]

### 修复

- 代码判分递归误判：`exec` 拆开 globals/locals 导致用户函数的 `__globals__` 落在独立 dict，正确的递归代码被判 `NameError`（自测发现）；改为同一 dict，`make_counter` 专用判分同修
- C 维度伪自信折扣持久化到 discount_factor：此前命中伪自信后数值会被下一次 MIRT 重算覆盖，现在每次更新末尾按 `sigmoid(θ_C) × discount_factor` 折算；顺带修复 misconception 折扣双重应用的问题
- 认知地图诚实标注：X 维度标注「MVP 未提供支架/提示机制，暂未测量」，Bloom L5/L6 标注「暂无对应层级题目」（题库当前最高 L4），不再显示先验假数值
- TC 显示名统一为单一来源 `engine.tc_detector.tc_library`，删除 content 库的隐式 key 拼装，同步两库文案避免漂移
- 修正 liminal → post_liminal 判定：需连续 3 次 L3+ 正确才跨越（此前规则未真正生效）

## [v0.1.0] - 2026-08-22

Phase 0 从 ECOS 选择性迁移并搭建的最小可运行链路。

### 新增

- CLI 最小链路：做题 → 5D 状态更新 → 认知地图展示（`python -m cogmirror.cli`）
- 静态 Python 题库（5 个 topic × Bloom L1-L4）+ 确定性 partial credit 判分（choice/fill/code，本地跑测试用例）
- 5D 信念引擎：MIRT 载荷估计 + BKT 掌握概率 + Bloom 六层 + TC 三态机（pre/liminal/post）+ 伪自信检测
- SQLite 持久层：用户 / 作答记录 / 状态快照，支持数据导出与删除（成人向合规，无监护人字段）
- 自动化测试：pytest 全绿

### 说明

- 不迁移 ECOS 的 dual_agent 互校、POMDP 策略、Multi-Domain 内核（见 `LESSONS-FROM-ECOS.md`）
- MVP 无 LLM 依赖：静态题库 + 确定性判分为用户确认的方案
