# Qwen3-32B RAG/PAG Baseline

该实验在与 IDPO 完全相同的 Per-Pcs Test 前 50 用户上提供两个不训练模型的
简单对比基线，共 304 条 Query。

## 定义

- **RAG**：BM25 从目标用户完整历史中检索 Top-8 输入/输出样例；Qwen3-32B
  根据 `Current Query + Top-8 History` 一次生成一个标题。
- **PAG (Profile-Augmented Generation)**：Qwen3-32B 先根据目标用户完整历史
  标题生成一次稳定写作 Profile；之后根据 `Current Query + User Profile` 一次
  生成一个标题。

PAG 不再同时输入 Top-8 History，否则实际测试的是 `RAG + PAG`，无法判断稳定
Profile 本身是否优于实例检索。两个方法使用同一个模型、`temperature=0`，不使用
Test Gold、候选池、Editor、Ranker 或 per-user 参数训练。

## 指标

报告官方兼容 ROUGE-1、ROUGE-L 及其 95% bootstrap CI，同时报告与 HYDRA
一致口径的 SacreBLEU。另提供 query-macro、user-macro，以及 PAG 相对 RAG 的
逐 Query 胜/平/负和配对 bootstrap CI。

## 启动与恢复

```bash
cd /home/liux/kk/MEVO
tmux new-session -d -s mevo_rag_pag_first50 \
  "bash code/run_rag_pag_test_first50.sh 2>&1 | tee logs/20260803_rag_pag_first50.log"
```

API 原始响应按 Prompt 缓存，Profile 和预测每 10 条原子写盘；使用相同命令即可
断点续跑。默认 API 并发为 32，可在 YAML 的 `baseline.concurrency` 中调整。

## 2026-08-03 正式结果

实验完整结束，共生成 50 个 PAG Profile 和 608 个标题，无缺失或重复样本。

| 方法 | ROUGE-1 | ROUGE-L | SacreBLEU | 平均标题词数 |
|---|---:|---:|---:|---:|
| RAG (BM25 Top-8) | 0.452634 | 0.393803 | 9.0575 | 13.54 |
| PAG (完整历史 Profile) | 0.372550 | 0.311495 | 5.1407 | 17.92 |
| Gold | - | - | - | 10.68 |

ROUGE-L 上，PAG 相对 RAG 的逐 Query 平均变化为 `-0.082351`，95% 配对
bootstrap CI 为 `[-0.099502, -0.066018]`；PAG/RAG/平局分别为
`77/218/9` 条。差异没有跨过 0。

当前定义下，局部且与 Query 对齐的 Top-8 历史实例明显优于将完整历史压缩成一个
静态 Profile。抽样检查发现 Profile 虽被要求只描述稳定输出偏好，仍会混入常见研究
主题，并让生成标题过长、加入 Gold 中没有的次要细节。这个结果不等于所有 PAG 都
无效，而是说明“单次完整历史摘要直接替代 Query-conditioned 检索”在该子集上存在
明显的信息压缩和适用性问题。

完整数值与逐样本输出分别位于：

- `result/perpcs_rag_pag_qwen32b_test_first50_v1/metrics.json`
- `result/perpcs_rag_pag_qwen32b_test_first50_v1/predictions.jsonl`
- `result/perpcs_rag_pag_qwen32b_test_first50_v1/pag_profiles.jsonl`
