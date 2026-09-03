# Base Parent + Teacher SFT 数据构造试验

`build_base_parent_teacher_sft.py` 是一个独立的小规模数据构造脚本：

```text
Llama2-7B 按 Base Prompt 生成 Parent
        ↓
Teacher 查看 Query + Top-8 History + Parent + Gold
        ↓
判断 valid / repairable / unusable
        ↓
输出最小编辑 Trace 和 Gold-aware SFT 样本
```

运行示例：

```bash
/home/liux/kk/MEVO/.venv/bin/python \
  code/26_9_1/build_base_parent_teacher_sft.py \
  --config config_global_llama2_7b_visgpt_prime_matched.yaml \
  --split train --limit 8 \
  --output dataset/editor_sets/base_parent_teacher_pilot
```

如果已有相同 Base Prompt 的缓存，可通过 `--parent-file` 跳过本地模型加载：

```bash
--split test --limit 8 \
--parent-file result/20260828_221331_mevo_global_llama2_7b_visgpt_prime_matched/predictions/full_test100/base_text/test_predictions.jsonl
```

输出文件：

- `01_base_parent_records.jsonl`：Base 真实 Parent 和原始响应；
- `02_teacher_annotations.jsonl`：Teacher 的质量判断和编辑解释；
- `03_sft_examples.jsonl`：可直接用于 SFT 的样本；
- `report.json`：数量和质量统计。

Teacher 看到 Gold 只用于离线构造标签；`03_sft_examples.jsonl` 的 `prompt` 不包含 Gold。
默认丢弃 `parent_quality=unusable` 的样本。

## Direct Gold SFT（无 Teacher、无 Trace）

`build_direct_parent_gold_sft.py` 只复用真实 Base Parent，构造纯文本主任务：

```text
Query + Top-8 History + Parent -> Gold 标题
```

它输出 `all_sft.jsonl`、`train_sft.jsonl` 和 `validation_sft.jsonl`，并生成
`analysis.md` 检查 Parent 与 Gold 的距离。`--parent-records` 可直接读取已有的
`01_base_parent_records.jsonl`；不提供时会调用本地 Base，但不会调用 Teacher。

## Distilling Step-by-Step 风格多任务 SFT

`build_multitask_rationale_sft.py` 在 Direct Gold 主任务之上，离线调用 Teacher
生成简短自然语言编辑解释。每个 Query 最多产生两个独立任务样本：

```text
[TITLE]     Query + History + Parent -> Gold 标题
[RATIONALE] Query + History + Parent -> 简短编辑解释
```

这不是把 Trace 和标题拼成一个长 JSON。Teacher 只在生成 rationale 时查看 Gold，
Student Prompt 不包含 Gold；推理时只使用 `[TITLE]` 任务。rationale 失败时仍保留
对应的 Title 样本，并在 `manifest.json` 中统计失败数量，不会阻断主任务数据构建。

示例：

```bash
python code/26_9_1/build_multitask_rationale_sft.py \
  --config config_global_llama2_7b_visgpt_prime_matched.yaml \
  --split train --limit 0 \
  --parent-records /data/liux/MEVO_global_cot/dataset/editor_sets/\
    20260902_094902_direct_parent_gold_full/01_base_parent_records.jsonl \
  --output /data/liux/MEVO_global_cot/dataset/editor_sets/\
    multitask_title_rationale_sft
```

训练时可将输出目录配置给现有 `06_train_editor_lora.py`，并将 rationale 样本的
`sample_weight`（默认 0.1）作为辅助损失权重。推理配置必须使用与训练一致的
`[TITLE]` 前缀；不要把 `[RATIONALE]` 作为推理时的必需输入。

### Pilot 检查结果

在 8 条真实 Base Parent 上运行后，得到 8 个 Title 样本和 8 个 Rationale 样本；
其中 3 条被 Teacher 标记为有历史证据支持，5 条为 task-only，0 条失败。产物位于：

`/data/liux/MEVO_global_cot/dataset/editor_sets/multitask_rationale_test8_pilot_v4`

该 pilot 使用 test 样本，仅用于质量检查，不能用于正式 SFT 训练。正式训练必须使用
train split，并按 Query 先划分 train/validation，再生成两个任务样本。

## Crossover-only SFT

当前 Llama2-7B Crossover/Dual-operator 流程使用 4096-token 上下文。SFT、
Parent 生成与 Adapter 推理统一使用“保留 Prompt 头尾”的截断方式；推理会为
生成的 64 tokens 预留上下文，避免默认右截断删除 Parent 和 `OUTPUT:`。

正式 Crossover v2 保持纯标题单任务：Teacher 只负责严格判断
`merge/keep_a/keep_b/reject`，程序再次检查贡献描述与 decision 是否一致。
`merge` 监督 Gold，`keep_a/keep_b` 监督对应 Parent，`reject` 不进入训练；
Teacher 的解释只保留作门控诊断，不进入 Student loss。入口为
`run_crossover_gate_v2_full.sh`。

[`run_crossover_sft.py`](./run_crossover_sft.py) 是独立的 Crossover 数据、训练和
评估入口。正式数据协议如下：

```text
Query + Top-8 History
        ↓
Llama2-7B：1 greedy + 3 sampled Parents
        ↓
target-blind MMR 选择 Parent A/B
        ↓
Teacher 只用 Gold 判断 Pair 是否真的可融合
        ↓
[CROSSOVER_TITLE] Query + History + Parent A + Parent B -> Gold
```

Student 始终输出一行纯文本标题。Teacher 不生成最终标签，Gold 由程序写入
`output_text`，且不会进入 Student Prompt。正式运行分三段，以适配数据环境和
GPU 训练环境：

```bash
ROOT=/home/liux/kk/MEVO_global_cot
CROSS_DATA=/data/liux/MEVO_global_cot/dataset/editor_sets/crossover_only_sft_v1
CROSS_RUN=/data/liux/MEVO_global_cot/result/crossover_only_sft_v1

# 1. 真实 Base Parent Pool + VisGPT Pair 门控 + SFT JSONL
/home/liux/kk/MEVO/.venv/bin/python \
  "$ROOT/code/26_9_1/run_crossover_sft.py" \
  --stage build --data-dir "$CROSS_DATA" --run-dir "$CROSS_RUN" \
  --pool-source base_model --teacher-mode api

# 2. Llama2-7B FP16 LoRA
/home/liux/miniconda3/envs/hydra/bin/python \
  "$ROOT/code/26_9_1/run_crossover_sft.py" \
  --stage train --data-dir "$CROSS_DATA" --run-dir "$CROSS_RUN"

# 3. 标准 test100；无 Pair Query 回退 greedy Parent，仍计入608分母
/home/liux/miniconda3/envs/hydra/bin/python \
  "$ROOT/code/26_9_1/run_crossover_sft.py" \
  --stage eval --data-dir "$CROSS_DATA" --run-dir "$CROSS_RUN" \
  --pool-source base_model --teacher-mode api
```

Parent Pool 和 Teacher Pair 标注都支持断点缓存。只有
`pool_source=base_model + teacher_mode=api + 非 smoke` 的 manifest 才会标记为
可用于正式报告。

## Dual-operator SFT

[`run_dual_operator_sft.py`](./run_dual_operator_sft.py) 不重复生成 Parent 或调用
Teacher，而是复用 Crossover 目录，构造：

```text
[MUTATION_TITLE]  Query + History + Parent A            -> Gold
[CROSSOVER_TITLE] Query + History + Parent A + Parent B -> Gold
```

有合格 Pair 的 Query 使用 `0.7/0.3` Mutation/Crossover 权重；无合格 Pair 时只
保留权重为 `1.0` 的 Mutation。正式 Dual-operator Adapter 从 Base 初始化，不从
Crossover-only Adapter 顺序训练。

```bash
DUAL_DATA=/data/liux/MEVO_global_cot/dataset/editor_sets/dual_operator_sft_v1
DUAL_RUN=/data/liux/MEVO_global_cot/result/dual_operator_sft_v1

/home/liux/kk/MEVO/.venv/bin/python \
  "$ROOT/code/26_9_1/run_dual_operator_sft.py" \
  --stage build --crossover-data-dir "$CROSS_DATA" \
  --data-dir "$DUAL_DATA" --run-dir "$DUAL_RUN"

/home/liux/miniconda3/envs/hydra/bin/python \
  "$ROOT/code/26_9_1/run_dual_operator_sft.py" \
  --stage train --crossover-data-dir "$CROSS_DATA" \
  --data-dir "$DUAL_DATA" --run-dir "$DUAL_RUN"

/home/liux/miniconda3/envs/hydra/bin/python \
  "$ROOT/code/26_9_1/run_dual_operator_sft.py" \
  --stage eval --crossover-data-dir "$CROSS_DATA" \
  --data-dir "$DUAL_DATA" --run-dir "$DUAL_RUN"
```

评估分别保存 Mutation、Crossover 和 Gold Oracle。Oracle 仅诊断候选池上限，
不能作为正式可部署结果；后续由 Shared Ranker 完成不可见 Gold 的 Top-1 选择。
