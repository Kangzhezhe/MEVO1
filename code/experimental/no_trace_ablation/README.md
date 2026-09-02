# No-Trace两阶段消融

研究问题：Shared SFT和Per-user IDPO都不训练Trace，只监督最终答案时，是否优于
当前Top-8简化Trace正式方案？

唯一目标改动：

| 阶段 | 正式方案 | 本消融 |
|---|---|---|
| Shared SFT | `0.2 L_trace + 1.0 L_output` | `1.0 L_output` |
| IDPO rollout | Trace + output | 只有output |
| IDPO log-prob | Trace权重0.2，output权重1.0 | Trace权重0，output权重1.0 |

保持不变：Per-Pcs划分、BM25 Top-8、4个Parent、10候选、Qwen2.5-1.5B LoRA、
DeBERTa、LOO用户历史、8次on-policy采样和Per-user Linear Head。

为保证候选起点一致，阶段一复用正式方案的Seed；阶段二复用相同LOO Parent。
所有结果写入独立的`perpcs_no_trace_*`路径，不覆盖正式实验。

启动：

```bash
cd /home/liux/kk/MEVO
bash code/experimental/no_trace_ablation/start_tmux.sh
```

主要对比指标：ROUGE-1、ROUGE-L、SacreBLEU、Candidate Oracle R-L、Hit@1、
Per-user IDPO Oracle增量和Per-user Head增量。
