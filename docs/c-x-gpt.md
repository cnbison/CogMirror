# C/X 定义差异：外部讨论记录

> **本文档性质**：2026-08-24 与外部大模型的讨论记录（原始问答），不是项目决策本身。
> **采纳结论**（已定稿，见 [PRD.md](../PRD.md) 8b 与 [GOVERNANCE.md](../GOVERNANCE.md) 规则7）：
> - 采纳"潜变量 vs 观测证据"三层分层（L1 5D / L2 结构化投影 / L3 证据）与五问框架
> - 5D 语义基准 = C 概念联结、X 元认知；Confidence / External Support 降级为 L3 证据变量，不删除
> - **修正本文档未覆盖的一处**：C=概念联结 的观测证据静态题库采不到，只有对话模式（Phase 0.5）能采到 → 语义基准现在定，但**代码不重命名、C 在 Phase 0.5 前保持未测量**，重写 C/X 推断须等 spike 数据（防"抽象先于验证"）
> - 以下为原文，保留作决策依据与溯源。

---

# 问题：
项目里有一个疑惑的地方，帮我分析和建议。这里有一个关于5D定义的差异，C/X 定义（新 PRD的概念联结/元认知 vs 代码现状的置信度/支架），为什么会产生这个差异，哪个更符合认知架构所需要的参数？

# 回答：
我看了你上传的 CogMirror 当前代码、PRD 和理论文档后，我认为这个问题**非常关键，而且不是简单的“命名不一致”问题**。它实际上暴露出了 CogMirror 当前认知架构里一个更深层的设计问题：

> **PRD 已经开始在定义“认知结构本身”，而代码目前还停留在定义“如何从现有观测信号计算一个数值”。**
>
> 所以 C/X 的差异，本质上是 **latent cognitive construct（潜在认知构念）** 和 **observable measurement/proxy（可观测测量代理）** 混在了一起。

而如果你的目标是把 CogMirror 做成真正的“认知架构”，我明确建议：

**以 PRD 的 C=概念联结、X=元认知为最终 5D 语义方向；把当前代码中的“置信度”和“外部支架”降级为观测变量/证据变量，而不是 5D 的核心认知维度。**

但还不能简单地把代码里的两个字段直接改名。下面我详细拆开。

---

# 一、先把这个差异说清楚

目前实际上存在两套 5D：

| 维度 | 新 PRD / 理论定义                   | 当前代码定义                    | 本质       |
| -- | ------------------------------ | ------------------------- | -------- |
| K  | Knowledge 知识                   | Knowledge                 | 认知能力     |
| P  | Procedure 程序技能                 | Procedure                 | 认知/技能能力  |
| S  | Strategy 策略                    | Strategy                  | 认知能力     |
| C  | **Conceptual Connection 概念联结** | **Confidence 置信度**        | 两者不是一回事  |
| X  | **Metacognition 元认知**          | **External Support 外部支架** | 两者也不是一回事 |

而且这个问题在你的代码里非常明确：

```python
C = "C"  # Confidence（认知置信度，含伪自信折扣）
X = "X"  # External Support（外部支架）
```

但理论文档已经开始使用另一套语义：

```text
C（Confidence / 概念联结）
X（External / 元认知）
```

甚至 PRD 已经明确写：

> C（概念联结）对应图式/结构性知识
> X（元认知）对应 Flavell 的经典元认知概念

所以我认为：

**现在不是 PRD 写错了，而是代码没有完成从 ECOS 旧语义到 CogMirror 新认知模型的迁移。**

---

# 二、为什么会产生这个差异？

我认为主要经历了三个阶段。

## 第一阶段：ECOS 更关注“能不能判断学生状态”

ECOS 最初的问题其实是：

> 我怎么根据学生做题行为，估计他的学习状态？

所以当时特别容易产生这样的设计：

```text
学生回答
   ↓
正确/错误
   ↓
知识掌握
   ↓
自评置信度
   ↓
发现“我觉得会，但其实不会”
```

于是：

**Confidence 是一个非常有价值的变量。**

因为它可以识别：

> “学生自己认为会”与“学生实际上不会”之间的差距。

这就是你现在代码里的：

```python
self_confidence
        ↓
actual_score
        ↓
illusory_confidence
        ↓
C discount
```

这个机制本身**没有问题**。

甚至我认为它是 CogMirror 很有价值的一条信号。

问题在于：

> **它不应该叫 C 维度。**

---

# 三、为什么“Confidence”不适合成为认知架构的核心维度？

因为 Confidence 更像：

> **对自己认知状态的估计**

而不是：

> **认知状态本身。**

举一个非常典型的例子。

学生知道：

```python
for i in range(10):
    print(i)
```

然后你问：

> 为什么 range(10) 不包含 10？

学生回答：

> 因为 Python 的 range 是左闭右开。

这个学生：

* 知道语法规则
* 能解释规则
* 可能有很高 confidence

但他是否理解：

```text
range()
list slicing
array indexing
loop boundary
off-by-one
```

这些东西之间的共同结构？

不一定。

也就是说：

```text
Confidence = 我觉得自己懂不懂
```

而：

```text
Conceptual Connection = 我脑中的知识节点之间是否形成了结构
```

这是完全不同的两个东西。

---

# 四、所以 C=“概念联结”其实比 C=“置信度”更适合作为 5D

我甚至认为这一点是 CogMirror 从 ECOS 走向“认知架构”的一个关键升级。

因为如果你的目标只是：

> 学习状态监测

那么：

```text
K
P
S
Confidence
External Support
```

完全可以工作。

但是如果你的目标是：

> **建立人的认知地图**

那么你真正想知道的是：

```text
这个人知道什么？
这个人能做什么？
这个人知道什么时候该怎么做？
这个人能不能把知识组织成结构？
这个人能不能监控自己的认知？
```

这五个问题对应：

```text
K = Knowledge
P = Procedure
S = Strategy
C = Conceptual Connection
X = Metacognition
```

这就形成了一个非常漂亮的认知层级：

```text
            X
       Metacognition
      “我知道我怎么知道”
              ↑
              │
             C
      Conceptual Connection
       “知识之间怎么连接”
              ↑
              │
             S
          Strategy
       “什么时候用什么”
              ↑
              │
             P
         Procedure
         “具体怎么做”
              ↑
              │
             K
         Knowledge
          “知道什么”
```

这已经开始具有**认知架构的结构性**了。

---

# 五、但是 X=“外部支架”为什么又会出现？

这个问题更有意思。

因为：

> **External Support 实际上是一个非常重要的变量，但它不是一个好的“认知维度”。**

例如：

```text
学生第一次写递归：

独立完成      → 需要较少支架

给一个提示：
“想想函数调用自己”
               ↓
完成

再给完整例子：
“参考这个代码”
               ↓
完成
```

这里的 External Support 非常有价值。

因为它告诉我们：

> 学生目前的能力是在什么程度的外部帮助下表现出来的。

这其实是非常重要的信息。

但它描述的是：

> **“认知表现发生时的环境条件”**

而不是：

> **“学生脑中的认知结构”。**

所以它更适合成为：

```text
Scaffolding Dependency
```

或者：

```text
Support Requirement
```

这样的**状态变量 / 观测变量 / 上下文变量**。

而不是 5D 的第五个认知维度。

---

# 六、这一点可以用一个非常重要的区分解决

我建议你以后把 CogMirror 的状态模型拆成三层：

```text
                 CogMirror Cognitive State
                           │
              ┌────────────┴────────────┐
              │                         │
        Latent Cognitive State     Observable Evidence
              │                         │
        ┌─────┴─────┐             ┌─────┴──────┐
        │           │             │            │
       5D          Bloom          Confidence   Support
        │                         │            │
 K P S C X                    Self-report   Hints
```

也就是：

### 第一层：真正的认知潜变量

```text
θ = [K, P, S, C, X]
```

其中：

```text
K = Knowledge
P = Procedure
S = Strategy
C = Conceptual Connection
X = Metacognition
```

这是你真正要画进“认知地图”的东西。

---

### 第二层：学习认知结构

例如：

```text
BloomProfile
SOLO
Threshold Concepts
Misconceptions
Knowledge Graph
```

这些不是和 5D 完全同一层的东西。

它们是：

> **5D 认知状态在具体知识领域中的结构化投影。**

---

### 第三层：观测证据

这里才应该放：

```text
self_confidence
actual_score
illusory_confidence
scaffolding_level
hint_count
time_to_answer
number_of_revisions
explanation_quality
code_execution_result
error_pattern
```

这些东西非常重要。

但它们是：

> **Evidence → 推断 → Latent State**

而不是：

> Evidence = Latent State

---

# 七、这样一来，你现在代码中的 C 机制反而可以保留下来

这是我特别建议你不要做的一个事情：

**不要因为把 C 改成 Conceptual Connection，就把现在的 Confidence 机制删掉。**

恰恰相反。

应该把它升级成：

```text
Self-Confidence
        │
        ├──── actual performance
        │
        ↓
Calibration
        │
        ↓
Metacognition evidence
        │
        ↓
X（Metacognition）
```

也就是说，现在：

```text
self_confidence
       ↓
C
```

应该变成：

```text
self_confidence
       +
actual_performance
       +
revision_behavior
       +
error_awareness
       +
reflection
       ↓
Metacognitive Evidence
       ↓
X
```

这其实比现在的架构**强很多**。

---

# 八、而“外部支架”也不应该被删除，而应该换一个位置

现在：

```text
X = External Support
```

建议变成：

```text
Scaffolding Context
```

例如：

```json
{
  "hint_count": 2,
  "hint_level": 0.4,
  "solution_exposure": 0.1,
  "external_support": 0.6
}
```

然后它可以影响：

> **我们对 K/P/S/C/X 的观测可信程度。**

举例：

学生完成了一道递归题。

### 情况 A

```text
独立完成
score = 1.0
hint = 0
```

这是很强的能力证据。

### 情况 B

```text
score = 1.0
hint = 3
system basically guided the whole process
```

这时候：

```text
P ≠ 1.0
```

至少不能直接认为程序能力已经达到 1.0。

所以：

```text
Scaffolding
      ↓
Evidence Reliability
      ↓
Latent State Update
```

这个关系才是合理的。

---

# 九、这也解释了为什么当前代码会出现一个结构性问题

现在 `belief_engine.py` 的逻辑是：

```text
MIRT
 ↓
5D theta
 ↓
C misconception discount
 ↓
illusory confidence
 ↓
C discount
```

这意味着 C 同时承担了：

```text
概念联结
+
错误概念
+
自我置信度
```

这是典型的**构念污染（construct contamination）**。

因为：

> 一个学生可能概念结构很好，但是自我评价很差。

例如：

```text
真实能力：0.85
自我置信：0.40
```

这是：

> **低置信度 + 高概念理解**

而不是：

> **低概念联结**

反过来也一样：

```text
真实能力：0.40
自我置信：0.90
```

这是：

> **高估自己的能力**

而不是：

> **概念结构差**

所以当前代码实际上把两个正交变量压缩成一个变量了。

---

# 十、这也是为什么我认为 PRD 的方向更符合“认知架构”

如果我们把问题提升一层：

> **一个认知架构中的参数，应该描述什么？**

我会给出一个标准：

一个核心 latent dimension 至少应该满足：

### 1. 它描述认知系统内部的稳定属性

例如：

```text
Knowledge
Procedure
Strategy
Conceptual Structure
Metacognition
```

### 2. 它能够被多种行为证据共同推断

例如 C：

```text
解释概念
类比
迁移
解决变式问题
构建关系
纠错
```

共同推断：

```text
Conceptual Connection
```

而不是：

```text
一次 self-confidence 问卷
```

### 3. 它能够跨任务迁移

这是非常重要的。

一个真正的认知维度应该不是：

> “这道题我有多自信？”

而应该是：

> “这个人在不同问题中形成概念结构的能力如何？”

### 4. 它能够成为后续教学策略的条件变量

例如：

```text
K低
→ 补知识

P低
→ 做程序练习

S低
→ 做策略比较

C低
→ 做概念映射/类比/关联

X低
→ 做反思/自我解释/预测-验证
```

这样 5D 才真正具有**决策价值**。

---

# 十一、不过我还想进一步挑战一下 PRD：X=元认知也需要小心

这里我认为你们现在的方向已经对了，但还没有完全解决。

因为：

```text
S = Strategy
X = Metacognition
```

确实存在高度重叠。

比如：

> “遇到递归问题，我应该先画调用树。”

这是 Strategy。

但：

> “我发现自己一遇到递归就容易直接写代码，所以我以后应该先画调用树。”

这是 Metacognition。

所以一个非常好的区分方式是：

| 维度 | 核心问题             |
| -- | ---------------- |
| K  | **我知道什么？**       |
| P  | **我能做什么？**       |
| S  | **我应该怎么做？**      |
| C  | **这些知识之间如何连接？**  |
| X  | **我如何知道自己是否知道？** |

我非常推荐你采用这五句话。

因为这比现在的英文名称更容易形成整个系统的设计原则。

---

# 十二、因此我建议 CogMirror 最终采用这样的“5D + Evidence”架构

我会把它定义成：

```text
                    CogMirror
                        │
                Cognitive State
                        │
        ┌───────────────┴───────────────┐
        │                               │
   Latent 5D State                Evidence State
        │                               │
 ┌──────┼──────┐                ┌───────┼────────┐
 │      │      │                │       │        │
 K      P      S                Cnf.   Scaffold  Behavior
 │      │      │
 └──────┼──────┘
        │
     C      X
   概念联结  元认知
```

更准确地说：

```text
5D：

K Knowledge
P Procedure
S Strategy
C Conceptual Connection
X Metacognition


Evidence：

Self-confidence
Actual performance
Scaffolding
Hints
Execution
Explanation
Revision
Error patterns
Response time
...
```

然后建立：

```text
Evidence
   ↓
Observation Model
   ↓
Bayesian / MIRT / BKT / other inference
   ↓
5D latent state
   ↓
Cognitive Map
   ↓
Instructional Policy
```

这才是真正意义上的：

> **认知架构，而不是认知指标集合。**

---

# 十三、对当前代码，我不建议现在“大改”，而建议做一次语义迁移

因为你现在代码的数学骨架其实已经有价值：

```text
Observation
    ↓
BKT
    ↓
MIRT
    ↓
Bloom
    ↓
Misconception
    ↓
TC
    ↓
Confidence
```

没必要推倒。

应该做的是：

### Phase A：先修正语义

把：

```python
C = "Confidence"
X = "External Support"
```

改成：

```python
C = "Conceptual Connection"
X = "Metacognition"
```

但**不要立即把所有算法都改掉**。

因为这时候会出现：

```text
C 名字变了
算法仍然在测 confidence
```

这反而更危险。

---

### Phase B：建立 Evidence 层

例如：

```python
@dataclass
class EvidenceState:
    self_confidence: float | None
    actual_score: float
    scaffolding_level: float
    hint_count: int
    execution_success: bool | None
    explanation_quality: float | None
    revision_count: int
```

然后：

```text
Evidence
    ↓
Evidence → Dimension mapping
    ↓
5D inference
```

---

### Phase C：重新设计 C

C 不应该再由：

```text
confidence gap
```

直接决定。

而应该主要由：

```text
概念解释
+
概念迁移
+
类比
+
关系判断
+
错误修正
+
变式问题
```

这些证据推断。

例如：

```text
学生知道变量是什么
        +
理解变量生命周期
        +
能解释作用域
        +
能迁移到 closure
        ↓
C ↑
```

这才是“概念联结”。

---

### Phase D：重新设计 X

X 则应该利用你现在已经非常有价值的：

```text
self_confidence
```

但再增加：

```text
预测
→ 行动
→ 结果
→ 比较
→ 修正
```

这个闭环。

例如：

> “你认为自己能写出来吗？”

→ Prediction

> “请写。”

→ Action

> 实际运行失败

→ Outcome

> “哪里与你预期不同？”

→ Reflection

> “以后遇到这种问题，你会怎么判断？”

→ Regulation

这整条链才是真正的：

```text
Metacognition
```

而不是单纯：

```text
Confidence
```

---

# 十四、最终判断

所以，如果让我现在替 CogMirror 做这个架构决策，我会非常明确：

> **PRD 的 C/X 定义应该成为最终语义基准。**

即：

```text
C = Conceptual Connection
    概念联结

X = Metacognition
    元认知
```

而当前代码：

```text
C = Confidence
X = External Support
```

不要理解成“错误设计”，而应该理解成：

> **ECOS 遗留下来的两个高价值观测变量，被错误地提升成了 5D latent dimension。**

这是为什么会产生差异。

更重要的是：

**Confidence 和 External Support 都不要删除。**

应该把它们从：

```text
5D Dimension
```

降级成：

```text
Evidence / Context Variables
```

然后让它们参与 5D 推断。

---

## 我认为这里还有一个更深的架构结论

这次 C/X 的冲突，其实暴露出了 CogMirror 下一阶段最应该解决的问题：

> **5D 到底是什么？**

如果把它定义成：

> “五种我们方便测量的指标”

那么当前代码的 C/X 没什么问题。

但如果把它定义成：

> **“一个学习者认知系统的五个潜在状态变量”**

那么 PRD 的 C/X 明显更合理。

而我认为，**既然 CogMirror 的核心愿景已经从 ECOS 的“学习诊断工具”走向“认知镜像/认知架构”，就必须选择第二种定义。**

这也意味着下一步不要急着继续往 UI、题库、LLM 对话上堆功能。

**应该先把 `5D = latent state`、`Evidence = observation`、`Bloom/SOLO/TC = structured projection` 这三个层级正式定下来。**

一旦这个地基确定，后面 MIRT、BKT、对话诊断、动态追问、认知地图乃至教学策略，都会自然地知道自己应该放在哪里。
