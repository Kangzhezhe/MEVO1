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
