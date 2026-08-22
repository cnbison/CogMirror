# MIGRATION：从ECOS迁移到新项目的具体方案

> **性质**：本文档基于对`ecos-main`代码仓的实际通读产出，路径均为在原始zip包`ecos-main/`下的相对路径。执行前请以你本地实际的ECOS代码仓为准核对路径是否一致。
> **原则**：迁移不是复制粘贴，是"选择性移植+改造"。每一项都要问一遍："新产品（成年人、单一学科Python、开放对话可能是新增场景）真的需要这个吗？"

---

## 0. 迁移前的心态提醒

ECOS项目最大的教训之一，是"能力已经建好，就忍不住想用上"（8月的Kernel通用化回潮）。迁移时最大的风险不是漏搬了什么，而是**多搬了什么**。拿不准的模块，默认先不搬，等真实需求出现了再回来取。

---

## 1. 直接迁移（代码几乎不用改，学科对象没变）

| 源路径 | 用途 | 迁移方式 |
|---|---|---|
| `ecos/bloom/subject_libraries/python_basics.py` | Python学科的Bloom六层认知目标库 | 直接复制 |
| `ecos/cta/content/python_basics_misconceptions.py` | Python常见错误模式库 | 直接复制 |
| `ecos/cta/content/threshold_concepts.py`（Python相关部分） | 临界概念（liminal）状态机，用于识别"变量""循环嵌套"这类跨越中的概念 | 复制后**审查一遍，去掉非Python学科的条目** |
| partial credit判分逻辑（`ecos/cta/observation_engine.py`中该部分） | 代码逻辑对但有小瑕疵不判全错的评分机制 | 复制，需要单独测试验证在脱离原有dual_agent流程后依然正确工作 |
| `ecos/bloom/__init__.py` | Bloom六层框架的核心数据结构 | 直接复制 |

## 2. 改造后迁移（思路对，代码要瘦身或改数据模型）

| 源路径 | 改造要点 |
|---|---|
| `ecos/cta/belief_engine.py` + `belief_state.py` | 保留5D信念更新的贝叶斯框架，**去掉**与教师Q矩阵审核、家长同意流程耦合的部分。检查是否有对`domain/education.py`的隐式依赖，解耦 |
| `ecos/lca/`中的LinUCB / Thompson Sampling部分 | 只保留这两种轻量策略，**不要**一并迁移POMDP+PBVI（`ecos/lca/l4_optimization/pomdp.py`不迁移，见第4节） |
| `ecos/evaluation/policy_ab_test.py` | A/B对比框架思路可用（P1阶段做策略对比时），MVP阶段可以先不启用，代码留着但不接入主流程 |
| `ecos/persistence/db.py` | 表结构重新设计：**删除**`consent_version`等监护人同意相关字段，**替换为**成人向的标准数据合规字段（数据导出请求、删除请求时间戳等）。student_id这类命名建议改为user_id，去掉"学生"这个隐含身份 |
| 教师端证据链下钻UI（`web/`中对应部分） | 界面逻辑（层级下钻、证据可追溯）值得参考，但**要重新设计成给用户自己看的版本**——教师端UI是"审查他人"的视角，新产品需要的是"审视自己"的视角，交互和文案基调都不同，不建议直接复用组件 |

## 3. 只搬文档、不搬代码（理论参考，需要重新实现或明确标注局限）

| 源路径 | 说明 |
|---|---|
| `research/30-shared-cognitive-tools/theoretical-foundations/01-cta-mathematical-foundations.md` | MIRT/CD-CAT理论基础，带走当参考文档，**但要在新项目文档里明确标注**：这套数学框架的前提是结构化题目-属性矩阵，只对"做题"场景成立，不能直接套用到开放对话场景（见PRD第8节的已知技术难题） |
| `research/30-shared-cognitive-tools/theoretical-foundations/02-lca-instructional-foundations.md` | 认知负荷理论、Bjork合意困难四件套、认知学徒制——教学法理论参考，用于设计"一句话建议"的呈现逻辑（比如间隔效应可以用来决定什么时候提醒复习） |

## 4. 不迁移，留在ECOS原仓库

| 源路径 | 不迁移的理由 |
|---|---|
| `ecos/dual_agent/`整个模块 | H3假设（双Agent互校降低幻觉）统计上显著反向（ECOS H3验证报告，p<0.0001），未证实成立的核心机制不该成为新产品地基 |
| `ecos/lca/l4_optimization/pomdp.py` | 数据规模不支撑，MVP阶段用LinUCB/Thompson Sampling足够 |
| `ecos/domain/education.py`、`science.py`、`career.py` | Multi-Domain通用内核抽象，新产品现阶段只服务Python一个学科，不需要这层 |
| `ecos/plugins/` | 插件化SDK，B端插件化是验证过C端价值之后才该考虑的事（见PRD 5.3 Out of Scope） |
| `research/00-overview/`中K12整体架构提案、竞品对比表（Khanmigo/Squirrel AI对比） | 定位完全不同的历史文档，不带过去 |
| `discussions/`中K12特定决策记录（家长同意流程设计、教师Q矩阵审核机制、Multi-Domain架构提案讨论等） | 历史包袱，不是资产。**例外**：治理类讨论（防虚标规则的由来、H3验证方法论教训）建议提炼成一份精简的"经验教训"文档带过去，见下一节 |
| README里的"认知操作系统""6-12年陪伴"等叙事表述 | 品牌定位完全不同，新产品需要独立的、诚实反映当前验证阶段的表述 |

## 5. 需要新写的"经验教训"摘要文档

建议在新项目里创建一份`LESSONS-FROM-ECOS.md`，浓缩以下几条ECOS阶段真金白银换来的教训，作为团队（哪怕只有你一人+AI协作）的共同记忆，避免在新项目里重蹈：

1. **"验证通过"必须有至少两种独立指标交叉印证**——ECOS的H3假设两次被发现是指标选错导致的假阳性，"0拐点"结论曾是三个bug叠加的假象
2. **不要让测试者=开发者本人**——ECOS的lbc001就是开发者自己，导致"3个测试用户"的真实代表性远低于表面看起来的数字，新项目第一批测试用户必须是真实的、非团队成员的外部人
3. **抽象化工作要有日历硬约束**——ECOS在6月已经自我诊断出"框架先于应用"的风险，8月还是花了一周做Kernel通用化。新项目里，任何"要不要把XX做成可插拔通用能力"的冲动，默认延后到至少完成一轮真实用户验证之后
4. **版本号/完成度标注要对应真实验证程度，不是工程完成度**——ECOS曾出现"标100%完成但核心功能从未真正工作"的情况（LearningDNA，累计4次虚标）

---

## 6. 执行清单（建议按顺序做）

- [ ] 新建项目目录，初始化git仓库
- [ ] 从ECOS复制第1类（直接迁移）文件，跑一遍原有测试，确认脱离ECOS主框架后依然可用
- [ ] 改造第2类（belief_engine、persistence schema等），针对新数据模型写新测试
- [ ] 撰写`LESSONS-FROM-ECOS.md`
- [ ] 参考第3类文档，明确写下MVP阶段的数学建模局限说明
- [ ] 确认第4类内容**没有**被无意带过来（检查import依赖、检查README/文案是否残留K12叙事）
- [ ] 按[ROADMAP.md](./ROADMAP.md)开始Phase 0
