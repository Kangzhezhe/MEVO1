# 相同 50 Query 的三种监督信号质量对比

## 实验设置

三组使用完全相同的 50 个 Query、200 个 Parent 和 Gold。Query 通过固定哈希
抽样，三组 ID 顺序完全一致。

- S0 Output-only：只监督最终 Gold；
- S1 Free Trace：自由 Gold-aware `task_correction/profile_signal/edit_action` + Gold；
- S2 Atomic Trace：`type/source_span/target_span` 原子操作、双历史证据和随机用户
  历史对照 + Gold。

S2 对元答案措辞、整句 Parent、整句 Gold 和不足两条历史证据执行硬过滤。四次
独立生成仍不满足约束的 Query 记为 `quality_rejected`，不通过放宽规则凑齐样本。

## 定量结果

| 指标 | S0 Output-only | S1 Free Trace | S2 Atomic Trace |
|---|---:|---:|---:|
| Query | 50 | 50 | 50 |
| 有效 Trace | 0 | 200 | 180 |
| 通过严格门控的 Query | N/A | 未过滤 | 45/50 |
| 精确 Gold Output | 200/200 | 200/200 | 180/180（通过门控部分） |
| 元答案/Reference 泄漏 | N/A | 57/200 | 0/180 |
| 完整 Gold 写入 Trace | N/A | 49/200 | 0/180 |
| 含个性化证据的 Trace | N/A | 51/200 | 77/180 |
| Parent→Gold 变化解释覆盖率 | N/A | 0.632 | 0.815 |

S2 的 180 条有效 Trace 包含 606 个原子操作，平均每条 3.37 个；其中 518 个
task operation、88 个 personalized operation。所有 personalized operation 都引用
恰好两条真实历史。45 个有效 Query 中，25 个至少包含一个个性化操作。

50 个 Query 中有 12 个存在重复 Parent。这是三组共同的数据问题，会降低实际
独立监督样本量，不属于 Trace 方法本身的收益。

## 质量判断

### S0：最干净，但只提供结果监督

优势是不存在解释错误或答案提前泄漏，适合作为不可缺少的训练基线。缺点是模型
可能直接学习 `q -> Gold`，无法知道模型是否利用了 Parent 和 History。

### S1：结构完整，但存在明显的事后解释

28.5% 的 Trace 含“align with the reference”等元话语，24.5% 的 edit action
直接包含完整 Gold。它经常把“具体编辑动作”退化成“将标题改成完整答案”。因此
S1 可以作为失败对照，不适合直接扩大到完整训练集。

### S2：任务编辑质量最好，但个性化归因仍需二次过滤

S2 消除了检测到的元答案泄漏和整句 Gold 复述，将变化解释覆盖率从 0.632 提高
到 0.815，并能把大改写拆成多个局部 add/remove/replace 操作。5 个 Query 因
Teacher 连续生成整句替换而被拒绝，说明硬门控确实阻止了伪原子操作。

但“双历史证据”只是必要条件，不是充分条件。人工检查发现两类情况：

1. 可信个性化：两条历史标题都使用 `a class of`，S2 将添加该结构归为用户偏好；
2. 可疑归因：历史论文与当前 Query 同属 runtime verification，Teacher 把领域
   术语变化解释为用户的“抽象化偏好”；这更可能是主题相关性，而非稳定风格。

一个粗粒度词汇审计显示：88 个个性化操作中，56 个 pattern 与真实历史的词汇
联系强于随机历史，10 个持平，22 个反而与随机历史联系更强。该指标不能替代
语义判断，但说明随机历史写进同一 Prompt 并不能自动保证反事实归因正确。

## 推荐结论

监督质量排序不能简单写成 S2 > S0 > S1，因为二者作用不同：

- 最可靠的最终答案监督：S0；
- 最好的可解释编辑监督候选：经过门控的 S2；
- 当前不建议使用：S1 自由 Trace。

下一轮建议使用混合策略：

```text
45 个 S2 通过 Query：Atomic Trace + Gold
5 个 S2 拒绝 Query：退回 Output-only Gold
所有 personalized operation：再做历史支持语义审计
```

训练损失可先采用：

\[
L = 1.0L_{output}+0.1L_{task\_operation}+0.2L_{validated\_personalized\_operation}.
\]

只有通过“真实历史支持且随机历史不支持”的语义审计后，才能计算 personalized
operation loss。否则该操作只保留为 task operation，或直接删除。

本实验只比较监督数据质量，尚不能证明 S2 一定提高最终生成指标。下一步应在相同
Train300/Dev100 上训练 S0、S2 和 S2 去掉 personalized operation 三组模型，验证
原子任务轨迹与个性化轨迹分别是否带来实际收益。
