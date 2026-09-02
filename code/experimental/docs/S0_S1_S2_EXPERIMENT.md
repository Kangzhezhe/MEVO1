# S0 / S1 / S2 监督方式对比实验

本实验只改变 Editor 的训练监督，固定以下条件：

- Per-Pcs Train300 / Dev100 / Test100；
- 相同 Query ID；
- 相同 BM25 Top-8 History；
- 相同的2个 Task-only + 2个 Profile-conditioned Parent；
- Qwen2.5-1.5B-Instruct FP16 LoRA；
- 本地 Editor target-blind 重建候选池；
- DeBERTa-v3-base 共享 Scorer，输入只包含 `q+c`；
- 暂不加入 per-user Head。

## 三种监督

### S0：Output-only

```text
Input:  q + history + Parent
Target: {"output": Gold}
Loss:   L_output
```

目标是验证最简单的 `Parent → Gold` 编辑学习是否有效。

### S1：Free Gold-aware Trace

Teacher 在训练数据构造阶段看到 Gold，生成：

```text
task_correction + profile_signal + edit_action
```

程序强制最终 `output=Gold`。训练损失为：

\[
L=L_{output}+0.2L_{trace}
\]

S1 作为自由 Trace 对照，已知存在事后解释、完整 Gold 复制等噪声。

### S2：Atomic Trace

Teacher 生成局部 `add/remove/replace/...` 原子操作；个性化操作至少引用两条
真实 History。无法通过原子门控的 Query 不放宽规则，SFT 回退到该 Query 的 S0
Output-only 标签。

\[
L=L_{output}+0.2L_{atomic\_trace}
\]

## 运行方式

Teacher 数据阶段和 GPU 阶段分离，避免多个 Adapter 同时占用显存：

```bash
# S0：完整流程
bash code/start_tmux.sh code/config_s0.yaml \
  mevo_s0_output_only_train300 20260801_s0_output_only

# S1/S2：先运行到 SFT 数据，pipeline.run_gpu=false
bash code/start_tmux.sh code/config_s1.yaml \
  mevo_s1_free_trace_train300 20260801_s1_free_trace
bash code/start_tmux.sh code/config_s2.yaml \
  mevo_s2_atomic_trace_train300 20260801_s2_atomic_trace
```

S0/S1/S2 的正式数据和结果目录完全分离。完成 GPU 阶段后，用：

```bash
bash code/run_editor_to_ranker.sh code/config_s1.yaml
bash code/run_editor_to_ranker.sh code/config_s2.yaml
python code/14_compare_supervision.py
```

比较报告会同时包含监督质量、Editor 训练损失、Ranker Dev/Test 的 ROUGE-L、
Hit@1、Hit@5、Mean Regret 和 Pair Accuracy。
