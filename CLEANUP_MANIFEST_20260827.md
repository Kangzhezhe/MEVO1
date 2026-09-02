# 实验产物清理记录（2026-08-27）

本次清理保留：

- `code/` 全部流程代码与配置。
- `GLOBAL_COMPARISON_REPORT.md`、`README.md`、`TECHNICAL_ROUTE.md`。
- Global 路线 50 用户与 100 用户的最终预测和指标报告。
- Shared SFT、严格门控 Global IDPO、半结构化 Trace pilot 的最终 LoRA adapter 与训练报告。
- `MEVO/doc/perpcs_final_comparison.md` 及旧 MEVO 的代码、配置和必要文档。

已删除：

- Global 路线的候选池、检索结果、seed、Teacher cache、IDPO rollout/preferences/pairs、SFT 原始样本。
- 所有训练过程 checkpoint（保留最终 adapter）。
- 与汇报表无关的旧日志、消融预测和重复中间产物。

说明：最终预测和报告是可直接复核指标的最小产物；删除中间数据后不能从本地缓存重新运行完整 Teacher 生成阶段。

## 挂载数据清理

`/home/liux/kk/MEVO/dataset` 和 `/home/liux/kk/MEVO/result` 是指向
`/data/liux/MEVO` 的挂载目录。已删除该位置下的全部候选池、ranker 数据、IDPO
中间数据、Teacher cache，以及旧实验结果；保留 `dataset/raw`、模型基座和以下结果：

- `perpcs_idpo_gold_test_first50_pro6000_v1`
- `perpcs_rag_pag_qwen32b_test_first50_v1`
- `perpcs_no_trace_*_full_v1`
- `perpcs_simple_trace_top8_*_full_v1`
- `perpcs_conditional_preference_*_full_v1`

挂载目录清理后约占 6.5G，原始数据与模型基座未删除。

## Global 路线产物迁移

`MEVO_global_cot/logs` 和 `MEVO_global_cot/result` 已迁移至
`/data/liux/MEVO_global_cot`，项目内使用符号链接访问。顶层日志和结果实验包统一
采用 `YYYYMMDD_HHMMSS_experiment_name` 前缀；详细规范见 `ARTIFACT_NAMING.md`。
