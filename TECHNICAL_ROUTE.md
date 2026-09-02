# 全局半结构化 CoT + IDPO 技术路线

## 目标

训练一个共享 Editor，测试时不为目标用户新建 Adapter。测试用户只通过 BM25 Top-8 历史影响输入，因此属于 global cold-start inference；训练时的 LOO Gold 只来自训练用户。

## 阶段一：Shared SFT

Teacher 看到 Query、Top-8 History、Parent 和 Gold，离线输出短 Trace；Student 输入不含 Gold：

    {"evidence_ids":["h2"],"edit_reason":"历史中观察到的编辑倾向以及它适用于当前输入的原因","edit_action":"一条具体可执行的局部编辑","output":"Gold标题"}

训练时 Trace span 权重 0.2，最终标题 span 权重 1.0。正式推理只保留标题输出，因此不把自由长 CoT 当作额外推理开销。

## 阶段二：Global IDPO

1. 每个训练用户的历史记录轮流作为 LOO 伪 Query；被留出的标题是隐藏 Gold。
2. Shared SFT Editor 在同一 target-blind prompt 上采样 8 个 Mutation response。
3. 用隐藏 Gold 只在离线阶段计算 ROUGE-L，最高分和最低分且差距至少 0.03 的 response 构成 chosen/rejected。
4. 所有用户的 pair 合并，冻结 Shared SFT 作为 reference，连续训练一个全局 LoRA Adapter。

响应 log-prob 采用半结构化 Trace 和标题两个 span：0.2 * trace + 1.0 * output。

## 进化操作

Mutation 参考 EDIT 的关键编辑思想：一次响应只描述一个局部操作（insert/delete/replace/reorder/compress），避免把多个无关改写混成自由长解释。

Crossover 使用完整序列级融合：Student Mutation 先生成子代，Teacher 再读取两个子代、Query 和历史，直接生成一个完整融合标题及简短动作说明。初始 Parent、Mutation 和 Crossover 共同构成进化候选池；LOO Gold 只在离线选择 chosen/rejected 时出现。默认 crossover 进入偏好池，从而形成完整的 Mutation -> Crossover -> Selection -> Global IDPO 迭代。配置 global_idpo.dpo_include_teacher_crossover=false 可做纯 Student Mutation 消融。

## 评估

测试阶段加载唯一的 Global IDPO Adapter，对 Query、Top-8 History 和 Parent 生成一个标题；报告 ROUGE-1、ROUGE-L、SacreBLEU 和解析失败数。Ranker、候选池和 crossover 质量可以作为训练期诊断，但不进入主推理链。

## 当前默认假设

- 基座：Qwen2.5-1.5B-Instruct，FP16 LoRA（rank 16，alpha 32）。
- Teacher：VisGPT OpenAI-compatible endpoint，VIS_API_KEY 环境变量。
- 数据：Per-Pcs；训练用户为 base，测试为 test 前 100 条。
- 历史：BM25 Top-8，输入中对 abstract/title 做长度压缩。
- 全局 IDPO 只用训练用户，不做 per-user Adapter。

尚需单独做的实验是：DPO 是否只加权标题差异 token，以及 Qwen1.5B 迁移到 Llama-2-7B 后的公平对比。
