# MEVO Global Semi-structured CoT + IDPO

这是从旧版 `MEVO` 独立出来的全局路线，不修改旧项目。目标是训练一个能泛化到新用户的共享 Editor：

```text
Per-Pcs train users
  -> BM25 Top-8 history
  -> target-blind Parent seeds
  -> Teacher semi-structured CoT + Gold-aware SFT labels
  -> shared Qwen LoRA SFT
  -> training-user LOO on-policy mutation rollouts
  -> ROUGE-L Gold labels (offline only)
  -> one global IDPO LoRA
  -> test user history + one generated title
```

## Trace contract

Teacher 离线构造的半结构化轨迹为：

```json
{"evidence_ids":["h2"],"edit_reason":"历史中观察到的可迁移编辑倾向，并说明它适用于当前输入","edit_action":"一条具体局部编辑动作","output":"Gold标题"}
```

SFT 将 Trace 作为辅助 span，最终标题作为主 span：`L = 0.2 L_trace + 1.0 L_output`。推理只请求最终标题，不要求模型输出 CoT。

## IDPO and evolutionary operations

- Mutation：当前 SFT Editor 对同一 prompt 采样 8 个半结构化响应；LOO Gold 仅用于离线选择最高/最低 ROUGE-L，形成 DPO pair。
- Crossover：Mutation 子代先进入候选池，Teacher 再读取两个子代、当前 Query 和 Top-8 历史，进行 sequence-level 融合。Mutation、Crossover 和初始 Parent 组成同一个离线进化候选池；Gold 只用于从整个池中选择 chosen/rejected。默认 dpo_include_teacher_crossover=true，因此 crossover 可以真正影响下一轮共享 Editor。
- 全局 IDPO：所有训练用户 pairs 合并后连续训练一个共享 LoRA，不按用户重置、不保存 per-user Adapter。

## Run

```bash
cd /home/liux/kk/MEVO_global_cot
bash code/run_global_pipeline.sh
```

正式实验通过 `code/run_timestamped_global.sh <config>` 启动。日志和结果分别写入
`logs/YYYYMMDD_HHMMSS_experiment_name.log` 与
`result/YYYYMMDD_HHMMSS_experiment_name/`；详细结构见 `ARTIFACT_NAMING.md`。

## 数据隔离

训练阶段 LOO 的 held-out title 只用于离线 ROUGE 标签；它不进入 rollout prompt。测试用户不参与全局 Adapter 更新，只在推理时提供检索历史。`user_limit`、`limit` 和 `profiles_per_user` 可用于小规模 smoke。
