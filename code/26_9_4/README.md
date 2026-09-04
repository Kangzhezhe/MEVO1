# 26_9_3：无 Seed 的统一 SFT / Editor / Crossover 流程

本目录完全取消 **Seed Parent**。所有 Query、History 和 Gold 来自标准
`01_prepared.jsonl` / `02_retrieved.jsonl`；正式流程不读取
`03_seeds.jsonl`，不使用 `candidates[0]`，Parent 生成 Prompt 中也不存在
预先提供的标题。

## 共享 Parent 阶段

为了让 Direct-Parent Editor、Multitask Editor 和 Crossover 使用完全相同的
Base 输出，可先独立生成共享 Pool：

```bash
python -B code/26_9_3/build_shared_parent_pool.py \
  --split all --ablation main \
  --data-dir /data/liux/MEVO_global_cot/dataset/parent_pools/shared_main_v1
```

其输出包含 `train_parent_pool.jsonl` 和 `test_parent_pool.jsonl`。每个 Query
固定保存五个 Parent（1 greedy + 4 sampling），重复保留，`unique_parent_count`
单独记录。当前固定宽度主实验不使用 Candidate Dropout。Editor/Multitask 读取第一个
greedy Parent，Crossover 读取完整五个。
No-history 消融必须另建 `--ablation no_history` Pool，不能和 main Pool 混用。

代码中的 `training.seed`、`manual_seed` 是随机数种子，只用于实验复现，不是
Seed Parent。

## 实验定义

| experiment | Parent 生成 | SFT 输入 | SFT 输出 |
|---|---|---|---|
| `base` | 无 | Query + Top-8 History | 无训练，直接标题 |
| `base_top1` | 无 | Query + Top-1 History | 无训练，直接标题 |
| `base_query_only` | 无 | Query only | 无训练，直接标题 |
| `direct_no_history_sft` | 无 | Query only | Gold 标题 |
| `direct_top1_sft` | 无 | Query + Top-1 History | Gold 标题 |
| `direct_sft` | 无 | Query + Top-8 History | Gold 标题 |
| `editor_sft` | Base: Query + Top-8 History → Parent | Query + Top-8 History + Parent | Gold 标题 |
| `multitask_sft` | 与 Editor 相同 | Title 与 Rationale 两个任务 | Gold / Teacher rationale |
| Crossover Main | Base: Query + Top-8 History → 5 Parents | Query + Top-8 History + 5 Parents | Gold 标题 |

Teacher 只在 `multitask_sft` 中生成辅助 rationale。Teacher 不参与 Base、Direct、
Editor Parent 或 Crossover Parent 的生成。

## 入口

SFT 输入表实验统一入口：

```bash
/home/liux/miniconda3/envs/hydra/bin/python -B \
  code/26_9_3/run_sft_input_ablation.py \
  --experiment direct_sft --stage all --max-steps 430
```

将 `direct_sft` 替换为 `base`、`editor_sft` 或 `multitask_sft`。
`base` 只运行评估，不训练。

匹配历史数量的因果对照由以下控制器完成：No-history Pool 合并后先统计 Raw
Llama2 Zero-shot；主实验全部完成后，三卡并行运行 Query-only Direct SFT、
Frozen Base + Top-1 和 Direct SFT + Top-1，避免它们插队影响主实验。最终统一
汇总 k=0/1/8 的 Base 与 Direct SFT。

```bash
python -B code/26_9_3/scripts/run_query_only_causal_followup.py \
  --run-name 20260903_190315_noseed_three_gpu_queue
```

Editor/Multitask 复用共享 Parent：

```bash
python -B code/26_9_3/run_sft_input_ablation.py \
  --experiment editor_sft --stage all --max-steps 430 \
  --shared-parent-pool-dir /data/liux/MEVO_global_cot/dataset/parent_pools/shared_main_v1
```

固定5候选 Gold-supervised Crossover：

```bash
/home/liux/miniconda3/envs/hydra/bin/python -B \
  code/26_9_3/run_gold_multi_parent_crossover.py \
  --stage all --ablation main --max-steps 430 \
  --shared-parent-pool-dir /data/liux/MEVO_global_cot/dataset/parent_pools/shared_main_v1
```

两个脚本正式评估都会校验 test100 必须为 100 用户、608 Query。训练 Prompt
全部 mask，只对输出位置计算 loss。

## 三卡队列与断点恢复

本机 RTX 5000 Ada 和远端两张 RTX3090 的统一调度入口是：

```bash
python -B code/26_9_3/scripts/run_three_gpu_queue.py
```

GPU 作业由用户态 GPU Task Spooler 执行，不需要 root：

| 节点 | Task Spooler socket | 并发槽位 | 可用判定 |
|---|---|---:|---|
| 本机 RTX 5000 Ada | `/tmp/ts-liux-mevo.sock` | 1 | 空闲显存至少 90% |
| RTX3090 | `/tmp/ts-gp4_liux-mevo.sock` | 2 | 每张卡空闲显存至少 90% |

因此其他用户占用 GPU 时任务保持 `allocating`，释放后自动获得 GPU。查看队列：

```bash
TS_SOCKET=/tmp/ts-liux-mevo.sock ~/bin/ts
ssh RTX3090 'TS_SOCKET=/tmp/ts-gp4_liux-mevo.sock ~/bin/ts'
```

若 Main Parent Pool 和 Direct SFT 已完成，可复用原运行名从 Base 评估继续，
不会重新生成 Main Pool 或重训 Direct SFT：

```bash
python -B code/26_9_3/scripts/run_three_gpu_queue.py \
  --run-name 20260903_190315_noseed_three_gpu_queue \
  --resume-after-main
```

恢复模式会先校验 Main Pool 恰好包含 3643 条训练记录和 608 条测试记录，
并检查 Direct SFT 的测试报告存在；条件不满足时会直接退出。

## 对应的新结果表

旧 `SFT_INPUT_ABLATION_RESULTS.md` 使用了 task seed，不能与本目录结果直接混用。
本目录完成后应生成新的无 Seed 表：

| 方法 | 真实输入 |
|---|---|
| No-seed Base | Query + Top-8 History |
| No-seed Direct Gold SFT | Query + Top-8 History |
| No-seed Parent-aware Editor SFT | Query + Top-8 History + Base Direct Parent |
| No-seed Multitask Editor SFT | 同上，外加 Rationale 辅助任务 |
| No-seed Multi-parent Crossover | Query + Top-8 History + 5 Base Direct Parents |
