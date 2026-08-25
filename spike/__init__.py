"""Phase 0.5 对话式认知诊断 Spike.

独立实验，不进 MVP 判分链路：只回答一个问题——LLM 基于一段锚定对话给出的
能力估计，是否与独立 ground truth 存在稳定相关性（PRD 8a）。

约定：
- 本目录不打包进 wheel（pyproject packages 只有 cogmirror）
- 不改动 cogmirror/ 下任何代码（PRD 8b：Phase 0.5 前不重命名 C/X）
- 5D 语义用 PRD 8b 的 L1 基准：C=概念联结、X=元认知（与 cogmirror 内
  C=置信度/X=外部支架 的 L3 证据语义不同，见 spike/graph.py DimensionId）
"""
