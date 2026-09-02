# MEVO 当前最佳归档流程

本目录根路径保存目前效果最好的正式实现，不需要再从 `code/26_8_7/`
启动。历史日期子目录仅用于复现实验演进过程。

目录约定：

- 根目录：当前 Top-8 简化 Trace + Per-user IDPO 正式流程；
- `common/`：跨阶段共享实现；
- `prompts/`：当前正式流程实际使用的三个提示词；
- `legacy/`：旧Factor入口的兼容性说明；相关文件因旧测试约定仍保留在根目录；
- `experimental/`：未进入当前最佳方案的消融代码和配置；
- `26_*`：按日期冻结的原始实验版本，不作为当前入口。

## 当前正式方案

```text
Per-Pcs train/base
  -> BM25 Top-8 历史
  -> 2 task-only + 2 profile-conditioned Parent
  -> Teacher 构造简化个性化 Trace
  -> Shared Qwen2.5-1.5B LoRA SFT
  -> Mutation/Crossover 候选池
  -> Shared DeBERTa Ranker

目标用户完整历史 Leave-One-Out
  -> Shared SFT Editor on-policy rollout
  -> 历史 Gold 构造 chosen/rejected
  -> Per-user IDPO LoRA
  -> Per-user Linear Ranker Head
  -> 当前 Query Top-1
```

简化 Trace 的固定结构为：

```json
{
  "evidence_ids": ["h2"],
  "edit_reason": "与当前编辑有关的历史证据和理由",
  "edit_action": "一条具体修改动作",
  "output": "最终标题"
}
```

Top-8 是可选证据池，每条 Trace 只引用 1--2 条适用历史。SFT 和 IDPO 均采用：

\[
L=0.2L_{\mathrm{trace}}+1.0L_{\mathrm{output}}.
\]

当前 Query 的 Gold 不参与生成、IDPO 训练或 Ranker 适配。IDPO 标签来自目标
用户历史的 Leave-One-Out Gold。

## 正式入口

阶段一配置：`config_simple_trace_top8_full.yaml`

阶段二配置：`config_simple_trace_top8_idpo_first50.yaml`

完整流程直接启动：

```bash
cd /home/liux/kk/MEVO
bash code/start_simple_trace_top8_full_tmux.sh
```

前台运行：

```bash
bash code/run_simple_trace_top8_full_experiment.sh
```

从已生成 Seed 或偏好对恢复阶段二：

```bash
bash code/run_idpo_after_seeds.sh
bash code/run_idpo_after_pairs.sh
```

Smoke 测试：

```bash
bash code/run_pipeline.sh code/config_simple_trace_top8_smoke.yaml
```

## 主流程文件

| 阶段 | 文件 | 作用 |
|---|---|---|
| 01--03 | `01_prepare.py`、`02_retrieve.py`、`03_generate_seeds.py` | 数据准备、Top-8 检索和 Parent 生成 |
| 31 | `31_build_simple_conditional_traces.py` | 简化个性化 Trace 构造与门控 |
| 05--07 | `05_build_editor_sft.py`、`06_train_editor_lora.py`、`07_generate_editor_pool.py` | Shared SFT 与候选生成 |
| 08--09 | `08_build_scorer_data.py`、`09_train_scorer.py` | Shared DeBERTa Ranker |
| 17--21 | `17_idpo_rollout.py` 至 `21_evaluate_user_editor_idpo.py` | Per-user IDPO 数据、训练与评估 |
| 22 | `22_train_idpo_ranker_user_heads.py` | Per-user Linear Ranker Head |

`pipeline_common.py` 和 `idpo_common.py` 是两阶段共享的数据契约与工具。

## 已完成实验结果

同口径前 50 个测试用户、304 条 Query：

| 方法 | ROUGE-1 | ROUGE-L | SacreBLEU | Candidate Oracle R-L |
|---|---:|---:|---:|---:|
| 原 SFT + Per-user IDPO | 0.474992 | 0.421548 | 10.3886 | 0.540839 |
| Top-8 简化 Trace + IDPO + Per-user Head | **0.486895** | **0.438715** | **11.4802** | **0.544160** |

对应实验产物位于：

- `result/perpcs_simple_trace_top8_pipeline_full_v1/`
- `dataset/idpo/perpcs_simple_trace_top8_idpo_first50_v1/round_0/`
- `logs/20260814_simple_trace_top8_full/`

注意：Crossover 在当前实验中平均降低约 `0.0517` ROUGE-L；它仍保留在复现
配置中，但后续优化应优先验证关闭 Crossover、增加 Mutation 的方案。
