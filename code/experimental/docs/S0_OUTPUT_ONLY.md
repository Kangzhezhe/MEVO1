# S0 Output-only 全流程

S0 是第一轮最简单、最干净的基线：不训练 Teacher 生成的事后解释，只训练
Editor 将 target-blind Parent 改写为训练样本的 Gold。

## 数据和模型接口

```text
Teacher（训练数据阶段）
  输入：当前 Query；可选用户检索历史
  输出：2 个 Task-only Parent + 2 个 Profile-conditioned Parent

Editor SFT
  输入：current_input + retrieved_history + parent
  输出：{"output": Gold}

本地 Editor 推理
  输入：current_input + retrieved_history + parent(s)
  输出：{"output": candidate}

Ranker
  输入：Query + candidate
  输出：candidate score
```

Teacher 生成 Parent 时不看 Gold。SFT 阶段只在程序写入标签时使用 Gold，Student
Prompt 中没有 `target` 或 `reference_output`。Gold 也不会进入本地 Editor 的
validation/test Prompt 或 Ranker 输入。

## 本轮设置

- 数据：Per-Pcs `Train300 / Dev100 / Test100`；
- 每个 Query：2 个 Task-only + 2 个 Profile-conditioned Parent；
- Editor：Qwen2.5-1.5B-Instruct，FP16 LoRA，2 epochs；
- Editor Loss：\(L_{S0}=L_{output}\)；
- 候选池：4 个 Parent + 4 个 Mutation + 2 个 Crossover，共10个；
- Ranker：DeBERTa-v3-base，输入仅 `q+c`，共享全局 Ranker；
- Ranker 在当前 GPU 余量下使用 batch 1、梯度累积4、max length 256；
- 本轮暂不训练 per-user Head，先独立报告共享 Ranker，避免混入第二阶段因素。

## 为什么不生成 Gold-aware Trace

S0 的目的是回答一个最基础的问题：仅用最终 Gold 监督，Editor 是否能学会从
Parent 产生更好的候选。如果 S0 不能超过 Parent Oracle 或共享 Ranker 基线，
加入更复杂 Trace 不能直接解释收益。S1 Free Trace 不进入正式训练；S2 Atomic
Trace 作为下一轮增量对照。

## 产物

- SFT：`dataset/editor_sets/perpcs_s0_output_only_train300_v1/`；
- Editor：`result/perpcs_s0_output_only_editor_train300_v1/`；
- Ranker 数据：`dataset/ranker_sets/perpcs_s0_output_only_train300_v1/`；
- Ranker：`result/perpcs_s0_output_only_scorer_train300_v1/`；
- 报告：`result/perpcs_s0_output_only_pipeline_train300_v1/`。

启动命令：

```bash
bash code/start_tmux.sh \
  code/config_s0.yaml \
  mevo_s0_output_only_train300 \
  20260801_s0_output_only
```
