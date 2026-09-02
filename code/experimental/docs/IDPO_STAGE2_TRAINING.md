# 第二阶段：Per-User IDPO 训练流程

当前第二阶段不是重新训练整个 Qwen，而是以第一阶段共享 SFT Editor 为起点，为每个目标用户分别训练一个 LoRA Adapter。当前运行的是第 0 轮、LOO Gold 监督的 per-user IDPO。

## 总体流程

```text
目标用户完整历史
    ↓ Leave-One-Out
历史伪 Query + 隐藏历史标题 Gold
    ↓
共享 SFT Editor 对同一 Prompt 随机生成 8 个改写
    ↓
用隐藏历史 Gold 计算 ROUGE-L
    ↓
最高分作为 chosen，最低分作为 rejected
    ↓
按 user_id 聚合偏好对
    ↓
每用户独立训练一个 IDPO LoRA
    ↓
per-user Editor 为真实当前 Query 生成候选
    ↓
per-user Ranker Head 选择 Top-1
```

## 1. 构造每个用户的 LOO 数据

对于用户 $u$ 的完整历史：

$$
H_u=\{(x_{u,i},y_{u,i})\}_{i=1}^{M_u}
$$

依次将每条历史拿出来作为伪任务：

$$
q_{u,i}=x_{u,i},\qquad y^*_{u,i}=y_{u,i}
$$

其余历史作为可见个性化上下文：

$$
H_{u,-i}=H_u\setminus\{(x_{u,i},y_{u,i})\}
$$

全部剩余历史先参与 BM25 检索，最终只把 Top-8 放进 Editor Prompt。当前前 50 个用户共构造了 `7,078` 个 LOO Query，平均每用户约 142 条。真实 Test Query 和对应 Gold 不参与这一步。

实现：[10_build_adaptation_queries.py](./10_build_adaptation_queries.py)

## 2. 构造初始 Parent

每个 LOO Query 已经由 Teacher 生成一个 task-only Seed，作为需要被修改的 Parent：

```text
输入：LOO Abstract
输出：初始标题 Parent
```

Teacher 在这里看不到 LOO Gold。Seed 完成后，后续 IDPO 不再调用 Teacher API。

## 3. 当前 Editor 做 On-Policy Rollout

第 0 轮使用共享 S1 Editor：

$$
\pi_{\mathrm{SFT}}=\text{Qwen2.5-1.5B}+\text{Shared SFT LoRA}
$$

Editor Prompt 输入为：

```text
CURRENT_INPUT：LOO Abstract
PARENT：初始标题
RETRIEVED_HISTORY：该用户剩余历史的 Top-8
```

不输入显式用户 Factor，也不输入 Gold。当前只做 Mutation，不做 Crossover。对同一个 Prompt 使用：

```text
temperature = 0.8
top_p = 0.95
samples = 8
```

采样 8 个 JSON 回答，每个回答包含：

```json
{
  "decision": "revise|keep",
  "task_correction": "...",
  "profile_signal": {
    "evidence_ids": ["..."],
    "observation": "..."
  },
  "edit_action": "...",
  "output": "最终标题"
}
```

去除 JSON 无效、证据不合法和重复标题后，至少需要 4 个有效回答，否则该 LOO Query 不构造偏好。

实现：[17_idpo_rollout.py](./17_idpo_rollout.py)

## 4. 用隐藏 LOO Gold 构造偏好

Rollout 完成后才读取被留出的历史标题 $y^*_{u,i}$，对每个输出标题计算：

$$
r_{i,j}=\operatorname{ROUGE-L}(o_{i,j},y^*_{u,i})
$$

选择：

$$
r^+=\max_j r_{i,j},\qquad r^-=\min_j r_{i,j}
$$

只有满足：

$$
r^+-r^-\ge 0.03
$$

才保留偏好对。最高分回答为 chosen，最低分回答为 rejected。每个 LOO Query 最多形成一个 Pair，所以最多约 `7,078` 对，实际数量会因无效输出和低 Margin 而减少。

需要注意，是否优化 Trace 由配置决定：

- 原始实验 `rollout_response_mode=output_only`，chosen/rejected 只有
  `{"output":"..."}`，DPO 不包含显式 Trace；
- Trace-aware 对照使用 `rollout_response_mode=conditional_preference_trace`，
  chosen/rejected 保存第一阶段的完整条件偏好 Trace 和标题。ROUGE-L 仍然只
  评价最终标题，Trace 必须先通过证据 ID、当前输入片段和结构门控。

实现：[18_idpo_gold_score.py](./18_idpo_gold_score.py)

## 5. 按真实用户训练独立 LoRA

所有 Pair 按 `user_id` 聚合。对每个用户 $u$：

$$
\pi_{\mathrm{ref}}=\pi_{\mathrm{SFT}},\qquad
\pi_u^{(0)}=\pi_{\mathrm{SFT}}
$$

Reference Adapter 完全冻结；Policy 只训练 LoRA，Qwen Backbone 冻结。每开始一个新用户，Policy 都重置回共享 SFT 参数，所以用户之间不会串联训练。

定义：

$$
\Delta_u=
\log\pi_u(y^+\mid p)-\log\pi_u(y^-\mid p)
$$

$$
\Delta_{\mathrm{ref}}=
\log\pi_{\mathrm{ref}}(y^+\mid p)-\log\pi_{\mathrm{ref}}(y^-\mid p)
$$

当前 DPO Loss 为：

$$
\mathcal L_{\mathrm{DPO}}
=
-\log\sigma\left[
\beta(\Delta_u-\Delta_{\mathrm{ref}})
\right]
$$

其中 $\beta=0.1$。只累计回答部分 Token 的 Log Probability，不计算 Prompt Token。
原始 output-only 实验使用整段序列分数；Trace-aware 实验仿照第一阶段分别归一化：

$$
S_\pi(r\mid p)
=0.2\frac{1}{|T|}\sum_{t\in T}\log\pi(r_t\mid p,r_{<t})
+1.0\frac{1}{|Y|}\sum_{t\in Y}\log\pi(r_t\mid p,r_{<t})
$$

其中 $T$ 是 Trace span，$Y$ 是最终 `output` span。DPO 公式中的
$\log\pi(y\mid p)$ 替换为该加权序列分数，避免长 Trace 淹没标题监督。

训练设置为 `FP16`、1 epoch、学习率 `5e-6`、batch size 2、梯度累积 2，有效 batch 4。少于 10 个有效 Pair 的用户跳过。最终每个用户保存一个独立 PEFT Adapter：

```text
user_adapters/user_<user_id>/
```

实现：[20_train_user_editor_idpo.py](./20_train_user_editor_idpo.py)

## 6. 真实 Query 推理与 Ranker

推理时根据真实 `user_id` 加载对应 IDPO Adapter：

```text
真实 Query + Parent + 用户检索历史
        ↓
per-user Editor
        ↓
个性化候选池
```

真实 Query 的 Gold 只在生成完成后用于计算 Oracle ROUGE，不进入生成 Prompt。

随后共享 DeBERTa Backbone 冻结，对每个用户用 LOO 候选训练独立 Linear Head。Ranker 输入严格为：

```text
Query + Candidate
```

不输入 History、Factor 或用户 ID。Head 从共享 Ranker Head 初始化，使用 LOO ROUGE 偏好训练，并保留 1 条 LOO Query 选择 Global/User 分数融合系数。最终报告共享 Head、per-user Head 和 Candidate Oracle 的 ROUGE 与 Hit@1。

## 当前方案的严格定位

它已经实现了“共享 SFT 初始化 + 每用户独立 DPO LoRA + 每用户独立 Ranker Head”，但当前只跑第 0 轮。真正的多轮 IDPO 还需要将本轮 `user_adapters` 设置为下一轮 rollout policy，再重新采样和训练。

个性化信号来自“同一用户大量历史 Gold 对该用户候选的偏好”，而不是显式 Factor。当前主要限制是：偏好仍由单一 ROUGE-L 决定，Trace 本身没有单独验证，因而标题偏好可靠时，Trace 仍可能包含不准确的个性化解释。
