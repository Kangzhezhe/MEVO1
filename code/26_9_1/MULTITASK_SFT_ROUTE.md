# MEVO 多任务 Editor SFT 流程交接说明

本文档描述 `code/26_9_1` 当前实现的完整流程，便于其他 Agent 继续运行、排查和扩展。

## 1. 路线和边界

当前实现的是共享 Editor 的多任务 SFT，不包含 IDPO、Ranker 或 per-user Adapter：

```text
训练 Query + Top-8 History
          ↓
Base Llama2-7B 生成 Parent
          ↓
Title 主任务 + Rationale 辅助任务
          ↓
共享 Llama2-7B LoRA 多任务 SFT
          ↓
test100（100 用户、608 Query）单次标题推理
          ↓
ROUGE-1 / ROUGE-L / SacreBLEU
```

训练时学习两个任务，推理时只执行 Title 任务；Rationale 不是推理阶段的必经中间步骤。

## 2. 数据集和评估口径

必须使用与 Base/Trace 正式实验相同的数据集：

| 阶段 | processed split | 用途 |
|---|---|---|
| 训练 | `global_llama2_7b_visgpt_train` | 构造 SFT 样本 |
| 验证 | `global_llama2_7b_visgpt_validation` | SFT eval loss |
| 测试 | `global_llama2_7b_visgpt_test100_full` | 100 用户、608 Query 最终评估 |

`29_evaluate_global.py` 会在用户数或 Query 数不符合 100/608 时直接报错，避免不公平比较。

## 3. Parent 构建

入口：[`build_direct_parent_gold_sft.py`](./build_direct_parent_gold_sft.py)

每个 Query 使用一个真实 Base Parent：

```text
Query + Top-8 History → Base Llama2-7B → Parent
```

该阶段不调用 Teacher、不生成 Trace，也不把 Gold 放入 Prompt。之后构造主任务：

```text
Query + Top-8 History + Parent → Gold 标题
```

典型输出：

```text
01_base_parent_records.jsonl
all_sft.jsonl
train_sft.jsonl
validation_sft.jsonl
quality_analysis.jsonl
analysis.md
manifest.json
```

`quality_analysis.*` 只用于离线诊断 Parent 与 Gold 的距离，不进入学生 Prompt。旧版本若没有 `01_base_parent_records.jsonl`，后续脚本可读取 `all_sft.jsonl`，并从同一 split 的 `02_retrieved.jsonl` 补回字段。

## 4. Teacher rationale 辅助任务

入口：[`build_multitask_rationale_sft.py`](./build_multitask_rationale_sft.py)

Teacher 离线输入包含：

```text
current_input
retrieved_history（最多 Top-8）
parent
gold
```

Gold 只用于 Teacher 解释 Parent 到 Gold 的必要修正；学生 Prompt 不包含 Gold。

Teacher 标注接口：

```json
{
  "rationale": "one or two short sentences",
  "evidence_ids": ["visible_history_id"],
  "quality": "history_supported | task_only"
}
```

JSON 只是离线标注接口，不是学生推理时必须输出的格式。构建器检查 rationale 长度、Gold 泄漏、不可见 evidence ID 和元信息；最多保留两条证据。没有可靠历史证据时允许 `task_only`，不会强行伪造个性化信号。rationale 失败时仍保留 Title 样本。

## 5. 多任务 SFT 样本

每个 Query 最多产生两行独立样本，而不是拼接成长 JSON：

```text
[TITLE]
Query + Top-8 History + Parent
→ Gold 标题
sample_weight = 1.0
```

```text
[RATIONALE]
Query + Top-8 History + Parent
→ Teacher 的一到两句编辑解释
sample_weight = 0.1
```

两类样本的 `trace_text` 都为空；Prompt 不包含 Gold；`[TITLE]` 和 `[RATIONALE]` 是任务控制前缀。Title 是主任务，Rationale 是辅助任务。

## 6. SFT 训练

实际训练器：[`../06_train_editor_lora.py`](../06_train_editor_lora.py)

统一入口：[`run_multitask_sft_eval.py`](./run_multitask_sft_eval.py)

模型配置：

```text
基座：/data/liux/PriME/models/Llama-2-7b-ms-hf
FP16 LoRA，rank=8，alpha=16，dropout=0.05
target_modules=[q_proj, k_proj, v_proj, o_proj]
```

损失只计算响应 token，不计算 Prompt：

```text
Title 行：      output loss × 1.0
Rationale 行：  output loss × 0.1
Prompt token：  label=-100
```

这是共享 Causal LM 的 prompt-conditioned multi-task SFT，不是两个独立模型。

默认训练参数：`max_length=2048`、`batch_size=1`、`gradient_accumulation_steps=16`、`epochs=2`、`learning_rate=2e-4`。

配置：[`config_multitask_title_rationale_sft.yaml`](./config_multitask_title_rationale_sft.yaml)

## 7. 推理和评估

推理：[`../28_generate_global_predictions.py`](../28_generate_global_predictions.py)

评估：[`../29_evaluate_global.py`](../29_evaluate_global.py)

多任务模式下，推理会加载共享 SFT `final_adapter`，自动使用 `[TITLE]` 前缀，只生成一次最终标题，不生成 rationale。评估使用标准 test100 的 608 条 Query，并报告 users、queries、valid_predictions、ROUGE-1、ROUGE-L、SacreBLEU 和 prediction_error_count。

## 8. 一键运行

入口：[`run_multitask_sft_eval.sh`](./run_multitask_sft_eval.sh)

Parent 完成后执行：

```bash
bash code/26_9_1/run_multitask_sft_eval.sh \
  /data/liux/MEVO_global_cot/dataset/editor_sets/20260902_094902_direct_parent_gold_full/01_base_parent_records.jsonl
```

执行顺序：

```text
build_multitask_rationale_sft.py
        ↓
run_multitask_sft_eval.py
        ├─ 06_train_editor_lora.py
        ├─ 28_generate_global_predictions.py
        └─ 29_evaluate_global.py
```

每次运行使用独立的时间戳数据、结果和日志目录，不覆盖已有 Base/Trace 结果。

## 9. 当前运行状态

当前 Parent 构建：

```text
tmux: mevo_direct_parent_gold_sft_build
log:  logs/20260902_094902_direct_parent_gold_full_build.log
```

自动衔接任务：

```text
tmux: mevo_multitask_after_parent
log:  logs/20260902_multitask_after_parent.log
```

衔接脚本：[`continue_after_parent.sh`](./continue_after_parent.sh)

它等待日志出现 `DIRECT_PARENT_GOLD_BUILD_EXIT=0`，成功后优先查找 `01_base_parent_records.jsonl`，否则回退到 `all_sft.jsonl`，继续 rationale 构建、SFT 和评估。

## 10. 注意事项

1. `build_base_parent_teacher_sft.py` 是旧式 Teacher Trace/质量标注脚本，主要用于诊断，不要与 Direct + Multitask 路径重复运行。
2. 不要把 test split 的 rationale 数据用于正式训练。
3. 当前实现没有 IDPO；如后续增加 IDPO，应作为 SFT 后的独立阶段。
4. 当前自动评估只衡量最终标题；rationale 质量通过 `rationale_analysis.md` 和 `teacher_rationales.jsonl` 离线检查。
5. 所有比较必须保持相同 Parent、History、test100 用户集合、608 Query 和文本清洗规则。
