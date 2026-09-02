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
