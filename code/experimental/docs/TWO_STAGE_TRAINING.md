# MEVO 两阶段训练方案：Gold-aware SFT + Per-user IDPO

## 1. 目标与边界

目标是在不维护显式用户 Factor 库的前提下，让 Editor 先学习稳定的编辑能力，
再从单个用户的反馈中学习个性化偏好。两阶段分别解决：

1. **全局任务能力**：如何把普通 Parent 修正为高质量目标输出；
2. **用户个性化能力**：同样正确的多个输出中，当前用户更偏好哪一个。

Gold 只允许出现在阶段一的训练监督构造和离线标签计算中。任何 validation/test
生成、阶段二 Teacher Judge、Editor 推理和 Scorer 输入都不能看到 Gold。

## 2. 阶段一：Gold-aware Global SFT

### 2.1 Parent 构造

对训练样本 \((q,H_u,a)\)，先在不知道 Gold \(a\) 的条件下生成四个 Parent：

\[
C^{parent}=\{c_1^{task},c_2^{task},c_1^{profile},c_2^{profile}\}.
\]

- Task-only Parent 只输入当前 Query；
- Profile-conditioned Parent 输入当前 Query 和 BM25 检索历史；
- 两类 Parent 都不能读取 Gold。

### 2.2 Teacher 监督

Teacher 对每个 Parent 查看：

\[
(q,H_u,c_j,a),
\]

但只返回可审计的简洁推理摘要，不返回最终答案：

```json
{
  "parent_id": "...",
  "task_correction": "需要修正的任务内容",
  "profile_signal": {
    "evidence_ids": ["h2"],
    "observation": "与本次编辑有关、且由历史支持的用户偏好"
  },
  "edit_action": "从 Parent 到目标输出的具体编辑动作"
}
```

程序随后强制设置 `output=a`。因此 Teacher 的职责是解释编辑，不是另生成一个
可能偏离 Gold 的答案。若 Parent 已等于 Gold，则 `decision=keep`，否则为
`decision=revise`。

### 2.3 Student 输入与损失

Student 的输入严格不含 Gold：

\[
E_\theta(q,H_u,c_j)\rightarrow (t_j,a).
\]

每个 Query 最多产生 4 条样本；3,643 个 train Query 的上限为 14,572 条。
同一 Query 的样本权重归一为 1，避免 Parent 数量影响 Query 权重。损失为：

\[
L_{SFT}=0.2L_{trace}+1.0L_{output}.
\]

Trace 和 Output 分别按 span token 数归一化，最终标题仍是主要优化目标。这里的
Trace 是 Teacher 的可审计推理摘要，不宣称是其隐藏思维链。

### 2.4 无泄漏候选重建

Gold-aware SFT 样本只用于训练 Editor，不是候选。训练完成后，使用本地 Editor
重新生成三个 split：

```text
train seeds      -> local Editor -> train candidate pool
validation seeds -> local Editor -> validation candidate pool
test seeds       -> local Editor -> test candidate pool
```

Ranker 的 train/validation/test 全部读取这些本地、target-blind 候选。这样可以
避免把精确 Gold 混进候选 Oracle、Ranker 标签分布或最终评估。

## 3. 阶段二：Per-user IDPO

阶段二从阶段一策略 \(\pi_0\) 出发，目标是把用户偏好注入每用户参数，而不是
再次学习数据集 Gold。

### 3.1 用户级 on-policy 数据

对用户历史做 Leave-One-Out。每次隐藏一条历史输出，以其输入作为伪 Query，
剩余历史作为可见 Profile。当前用户策略 \(\pi_u\) 对同一输入采样多个编辑输出：

\[
y_1,\ldots,y_K\sim\pi_u(\cdot\mid q,H_u,c).
\]

这里必须使用随机解码而不是 greedy decoding。pilot 默认设置为
`temperature=0.8, top_p=0.95, K=4`；同一 Prompt 的重复响应在进入 Judge 前
去重，少于两条有效响应时不构造偏好对。

### 3.2 Teacher Pairwise Judge

Teacher 只查看当前输入、可见历史、两个候选及其证据化编辑摘要，不查看被隐藏
的 Gold，判断哪个候选在任务正确性和用户偏好上更好，得到
\((x,y^+,y^-)\)。低置信度或平局 pair 应过滤，不能强行制造偏好。

### 3.3 IDPO 更新

以冻结的全局策略 \(\pi_0\) 为 reference，对每个用户训练轻量 LoRA/Adapter：

\[
L_{IDPO}=-\log\sigma\left(\beta\left[
\log\frac{\pi_u(y^+\mid x)}{\pi_0(y^+\mid x)}-
\log\frac{\pi_u(y^-\mid x)}{\pi_0(y^-\mid x)}
\right]\right).
\]

使用 reference 约束可减少少量用户数据导致的灾难性漂移。模型和 checkpoint
必须以 `user_id` 为键，同一用户的多个 Query 共享一个 Adapter。

## 4. 实验判定

阶段一至少报告：Parent Oracle、本地 Editor Oracle、Editor Top-1、ROUGE-1/L、
BLEU、重复率、有效 JSON 率，以及正确 Profile 对随机 Profile 的消融。只有本地
Editor Oracle 高于 Parent Oracle，才能说明编辑真正扩展了候选空间。

阶段二至少比较：Global Editor、Per-user IDPO、随机用户 Adapter，并按用户报告
提升/下降比例与置信区间。若 Per-user 不超过 Global，需先检查 pair 的 Judge
一致性、有效独立历史数和用户可辨识性，不能只增加 Adapter 容量。

## 5. 阶段二实现

阶段二现已实现一轮完整 per-user IDPO：

| 阶段 | 文件 | 作用 |
|---|---|---|
| 17 | `17_idpo_rollout.py` | 同一 Prompt 随机采样多条当前策略 response |
| 18 | `18_idpo_teacher_judge.py` | Teacher 不看隐藏 Gold，过滤 tie/低置信度 |
| 19 | `19_build_idpo_pairs.py` | 构造同 Prompt 的 chosen/rejected DPO pair |
| 20 | `20_train_user_editor_idpo.py` | 以阶段一 Adapter 为 reference，按 user_id 训练 LoRA |

首轮 pilot 配置为 `config_idpo_pilot.yaml`。运行：

```bash
bash code/run_idpo_pilot.sh \
  code/config_idpo_pilot.yaml validation
```

Round 0 使用阶段一全局 Adapter 采样。下一轮把
`idpo.policy_adapter_root` 指向上一轮的 `user_adapters`，将 `idpo.round` 改为
1 后重新执行 17-20，即可形成真正的 iterative on-policy 更新。reference 始终
固定为阶段一 Adapter，防止多轮用户微调漂移。

当前 Scorer 的 per-user Linear Head 是候选排序适配，不等同于 Editor IDPO；
IDPO pair 已保留 `chosen_output/rejected_output`，可直接作为后续 per-user Ranker
的 Teacher 偏好监督。
