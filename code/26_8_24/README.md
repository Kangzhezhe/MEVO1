# MEVO 2026-08-24 正式流程

本目录只保存当前效果最好的正式路线，不包含S0/S1/S2、复杂Trace、Factor、
Teacher Judge、目标函数矩阵等消融实验。

## 方法

```text
阶段一：Shared Editor / Ranker
Per-Pcs train/base
  -> BM25 Top-8历史
  -> 2个task-only + 2个profile-conditioned Parent
  -> Teacher生成简化个性化Trace
  -> Qwen2.5-1.5B LoRA SFT
  -> 10候选生成
  -> Shared DeBERTa Ranker

阶段二：Per-user适配
目标用户完整历史Leave-One-Out
  -> Shared SFT Editor on-policy采样
  -> 历史Gold构造chosen/rejected
  -> Per-user IDPO LoRA
  -> Per-user Linear Ranker Head
  -> 当前Query Top-1
```

简化Trace：

```json
{
  "evidence_ids": ["h2"],
  "edit_reason": "与当前编辑相关的历史证据和理由",
  "edit_action": "一条具体修改动作",
  "output": "最终标题"
}
```

Top-8是候选历史证据池，每条Trace只引用1--2条历史。SFT与IDPO使用：

\[
L=0.2L_{\mathrm{trace}}+1.0L_{\mathrm{output}}.
\]

## 配置

- `config.yaml`：完整阶段一训练、候选生成和Shared Ranker；
- `config_idpo.yaml`：前50用户全历史LOO、Trace-aware IDPO和Per-user Head；
- `config_smoke.yaml`：只验证数据和Trace流程，不进入GPU训练。

三个配置均已展开为独立配置，不依赖任何消融配置。

## 运行

完整两阶段实验：

```bash
cd /home/liux/kk/MEVO
bash code/26_8_24/start_simple_trace_top8_full_tmux.sh
```

前台运行：

```bash
bash code/26_8_24/run_simple_trace_top8_full_experiment.sh
```

Smoke：

```bash
bash code/26_8_24/run_pipeline.sh code/26_8_24/config_smoke.yaml
```

阶段二恢复：

```bash
bash code/26_8_24/run_idpo_after_seeds.sh
bash code/26_8_24/run_idpo_after_pairs.sh
```

## 数据隔离

- Teacher构造SFT监督时可以读取训练样本Gold；
- validation/test候选生成严格target-blind；
- 阶段二只使用目标用户历史记录的LOO Gold；
- 当前测试Query的Gold不参与IDPO、候选生成或Ranker Head适配。

## 已完成结果

前50测试用户、304条Query：

| 方法 | ROUGE-1 | ROUGE-L | SacreBLEU | Candidate Oracle R-L |
|---|---:|---:|---:|---:|
| Top-8简化Trace + IDPO + Per-user Head | 0.486895 | 0.438715 | 11.4802 | 0.544160 |
