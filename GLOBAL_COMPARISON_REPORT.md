# Global 路线对照实验

## 固定评估协议

- Per-Pcs 原 MEVO 的前 50 个测试用户，共 304 条 Query。
- 生成输入使用相同的 BM25 Top-8 历史和相同的 `task_0` Parent。
- Global 路线只生成一个标题，不使用 Ranker。
- 测试 Query 的 Gold 不进入 Prompt；Gold 只用于离线评估。

## 结果

| 方法 | ROUGE-1 | ROUGE-L | SacreBLEU | 有效预测 |
|---|---:|---:|---:|---:|
| Parent `task_0` | 约 0.4699 | 0.4135 | - | 304/304 |
| 当前 Global IDPO（Trace + Crossover，output-only 推理） | 0.4671 | 0.4137 | 9.9437 | 301/304 |
| 严格门控 Global IDPO（Mutation-only + output-only） | **0.4792** | **0.4244** | **10.3933** | **304/304** |
| 旧 Shared Ranker（10 候选） | 0.4750 | 0.4215 | 约 10.39 | 304/304 |
| 旧 Per-user IDPO + Head | 0.4869 | 0.4387 | 11.4802 | 304/304 |

严格门控版本使用 2154 个 pair、50 个训练用户、270 次 DPO 更新。筛选条件为：

\[
r^+ - r^- \ge 0.03,\qquad r^+ - r_{parent} \ge 0.02
\]

并且只保留 Student Mutation，去除 Teacher Crossover，Trace 权重为 0。

## 配对分析

相对当前 Global IDPO，严格门控版本：

- ROUGE-L 平均逐 Query 增益：`+0.01480`
- 提升 Query：`135`
- 下降 Query：`110`
- 持平 Query：`59`
- Bootstrap 95% CI：`[-0.00320, 0.03304]`

相对原始 `task_0` Parent：

- ROUGE-L 平均逐 Query 增益：`+0.01099`
- 提升 Query：`152`
- 下降 Query：`110`
- 持平 Query：`42`
- Bootstrap 95% CI：`[-0.00627, 0.02757]`

因此改进方向是明确的，但当前 304 Query 规模下置信区间仍跨过 0，不能声称已经显著超过旧方案。

## 结论

当前 Global IDPO 效果不佳的主要原因得到验证：允许 chosen 劣于 Parent，以及 Trace/Crossover 与单输出推理协议不一致，会把大量噪声带入 DPO。加入 Parent improvement gate、关闭 Crossover、统一为 output-only 后，ROUGE-L 从 `0.4137` 提升到 `0.4244`，并消除了 3 个无效预测。

下一步应在更多训练用户上重复该严格门控版本，并按用户均衡采样 LOO pair；同时保留同一 304 Query 的 paired bootstrap，确认提升是否稳定。
