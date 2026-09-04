# Llama2-7B SFT 输入与监督方式对比

> **历史口径说明：** 本文档中的旧结果实际使用了 Teacher 生成的
> `task_seed_0`。其中 Base/Direct 将该 seed 直接放入 `PARENT TITLE`，
> Parent-aware/Multitask 则使用几乎完全复制该 seed 的 Base Parent。
> 因而下表是 Seed-assisted 历史结果，不是严格的无 Seed 输入消融。
> 无 Seed 重跑入口见 `code/26_9_3/README.md`。

## 实验口径

以下实验使用相同的评估口径：

- 基座模型：Llama2-7B
- 数据集：LaMP-5 / Per-Pcs
- 测试用户：100
- 测试 Query：608
- 检索历史：Top-8
- 推理方式：单次文本生成
- 评估指标：ROUGE-1、ROUGE-L、SacreBLEU

## 主要结果

| 方法 | 模型输入 | 训练监督 | ROUGE-1 | ROUGE-L | SacreBLEU |
|---|---|---|---:|---:|---:|
| Llama2-7B Base | Query + Top-8 History | 无训练 | 0.4699 | 0.4133 | 11.2331 |
| Direct Gold SFT | Query + Top-8 History | Gold 标题 | 0.4992 | 0.4498 | 12.8012 |
| Parent-aware SFT | Query + Top-8 History + Parent | Gold 标题 | 0.5014 | 0.4515 | 12.7148 |
| 多任务 SFT | Query + Top-8 History + Parent | Gold 标题 + Rationale 辅助任务 | **0.5064** | **0.4574** | **13.2451** |

## 相对 Base 的提升

| 方法 | Δ ROUGE-1 | Δ ROUGE-L | Δ SacreBLEU |
|---|---:|---:|---:|
| Direct Gold SFT | +0.0293 | +0.0365 | +1.5681 |
| Parent-aware SFT | +0.0315 | +0.0382 | +1.4817 |
| 多任务 SFT | **+0.0365** | **+0.0441** | **+2.0120** |

## 结果分析

1. `Query + History → Gold` 的 Direct Gold SFT 已经显著超过未经训练的 Llama2-7B Base，说明直接任务监督是当前性能提升的主要基础。
2. 在输入中加入 Parent 后，ROUGE-1 和 ROUGE-L 只有小幅提升，SacreBLEU 略有下降。这说明模型能够利用 Parent，但收益有限，且 Parent 质量可能不稳定。
3. 在 Parent-aware 主任务基础上加入 Rationale 辅助任务后，三个指标均达到当前最好结果。
4. 相比 Parent-only，Rationale 多任务训练带来 `+0.0050` ROUGE-1、`+0.0060` ROUGE-L 和 `+0.5303` SacreBLEU，说明简短编辑解释具有额外监督价值。
5. 当前结果支持“Direct Gold SFT 提供主要任务能力，Parent 提供有限编辑条件，Rationale 辅助任务进一步改善表示学习”的判断。后续仍应通过多个随机种子验证增益的稳定性。

## 实验产物

### Direct Gold SFT：Query + History → Gold

- 评估报告：`/data/liux/MEVO_global_cot/result/20260901_161506_mevo_global_llama2_7b_visgpt_direct_gold_sft_base_protocol/reports/sft_text/global_test_report.json`
- 训练报告：`/data/liux/MEVO_global_cot/result/20260901_161506_mevo_global_llama2_7b_visgpt_direct_gold_sft_base_protocol/editor/training_report.json`
- 日志：`/home/liux/kk/MEVO_global_cot/logs/20260901_161506_mevo_global_llama2_7b_visgpt_direct_gold_sft_base_protocol.log`

### Parent-aware SFT：Query + History + Parent → Gold

- 最终指标：ROUGE-1 `0.501405`、ROUGE-L `0.451474`、SacreBLEU `12.714839`
- 训练：430 steps，约 1 小时 54 分钟
- Train loss：`1.201764`
- Eval loss：`1.187689`
- 当前本机未保留对应的最终评估报告路径，表中采用实验运行时记录的最终指标。

### 多任务 SFT：Title + Rationale

- 评估报告：`/data/liux/MEVO_global_cot/result/20260902_142935_multitask_title_rationale_sft/reports/global_test_report.json`
- 训练报告：`/data/liux/MEVO_global_cot/result/20260902_142935_multitask_title_rationale_sft/editor/training_report.json`
- 日志：`/home/liux/kk/MEVO_global_cot/logs/20260902_multitask_after_parent.log`
