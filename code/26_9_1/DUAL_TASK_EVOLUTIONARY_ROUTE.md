# MEVO：双任务 Editor、两阶段个性化 Ranker 与推理期遗传搜索

## 1. 文档状态

本文档记录当前讨论后确定的一版完整技术路线，作为后续代码实现、实验设计和论文描述的统一依据。

需要注意：本文描述的是**目标方案**。当前仓库已经具备部分 SFT、Global IDPO、候选生成和 Ranker 代码，但尚未完整实现本文定义的“双任务 SFT + 双任务 IDPO + 两代推理搜索 + per-user Residual Ranker”全流程。

## 2. 核心研究问题

面向有历史记录的个性化文本生成任务，研究以下问题：

> 能否先用共享 Editor 学习任务级的单候选改写与双候选融合能力，再使用共享质量模型和目标用户轻量偏好参数，在推理阶段通过多代遗传搜索产生更符合当前用户的答案？

当前路线不是严格的“零历史用户冷启动”，而是：

- 共享 Editor 和共享 Ranker 可以直接服务新用户；
- 当目标用户具有历史数据时，额外训练一个很小的 per-user Ranker Residual Head；
- 当前测试 Query 的 Gold 在训练和推理过程中始终不可见。

## 3. 总体框架

```text
训练用户 Query + History + Gold
                ↓
      Base Llama2-7B 生成真实 Parent
                ↓
    构造 Mutation / Crossover 两类监督
                ↓
      Shared Dual-task Editor SFT
                ↓
    两类 on-policy rollout + Gold 选择
                ↓
       Shared Dual-task Global IDPO
                ↓ 冻结 Editor
    模拟推理期多代候选分布并用 Gold 标注
                ↓
          Shared Task Ranker
                ↓ 冻结共享 Backbone
       目标用户历史 Leave-One-Out
                ↓
       Per-user Ranker Residual Head

推理：
Query + Top-8 History
        ↓
Base 生成初始种群
        ↓
两代 Mutation / Crossover / Selection / Elitism
        ↓
Shared Ranker + User Residual Head 选择 Top-1
```

## 4. 符号定义

- $q_i$：当前 Query，例如 LaMP-5 的论文摘要；
- $H_i$：当前 Query 可见的用户历史；
- $R_i=\operatorname{Retrieve}(q_i,H_i)$：BM25 检索得到的 Top-8 历史；
- $y_i$：训练样本 Gold；
- $p_{i,a},p_{i,b}$：Base 或上一代产生的 Parent 候选；
- $c_{i,j}$：Editor 产生的候选；
- $P_i^{(g)}$：第 $g$ 代候选种群；
- $\pi_{\mathrm{SFT}}$：双任务 SFT Editor；
- $\pi_{\mathrm{IDPO}}$：双任务 Global IDPO Editor；
- $s_{\mathrm{shared}}$：共享 Ranker 分数；
- $s_{u}$：用户 $u$ 的最终个性化 Ranker 分数。

## 5. 数据协议与边界

### 5.1 训练和测试必须一致

真实 Base Parent 必须由同一个 Base Llama2-7B 协议产生，训练和测试保持以下设置一致：

- Base checkpoint；
- Query 模板；
- Top-8 历史表示；
- 最大输入长度；
- 解码与输出清洗规则；
- Parent 是否采样及采样参数。

否则 Editor 会出现明显的训练—测试 Parent 分布偏移。

### 5.2 Gold 使用边界

训练 Gold 可以用于：

- Mutation/Crossover SFT 的最终标题监督；
- 训练用户 rollout 的离线质量标注；
- Shared Ranker 的候选质量标签；
- 目标用户**已有历史记录**的 LOO Ranker 适配。

训练 Gold 不得用于：

- 候选生成 Prompt；
- 正式测试 Query 的 Editor 输入；
- 正式测试 Query 的 Ranker 输入；
- 测试时遗传搜索的 Fitness 计算。

## 6. 阶段一：Shared Dual-task Editor SFT

Mutation 和 Crossover 是同一个共享 Editor 的两种操作任务。二者都只输出最终标题纯文本，不输出 JSON，不监督自由 Trace。

### 6.1 Mutation 任务

Mutation 使用一个 Parent：

$$
z_i^{\mathrm{mut}}=(q_i,R_i,p_{i,a})
$$

监督目标：

$$
(q_i,R_i,p_{i,a})\rightarrow y_i
$$

推荐 Prompt 结构：

```text
[MUTATION]

[Query]
...

[Retrieved History]
...

[Parent]
...

[Output]
```

Mutation Parent 必须优先来自真实 Base 模型分布，而不是 Teacher 人工制造的低质量 Parent。

### 6.2 Crossover 任务

Crossover 使用两个 Parent：

$$
z_i^{\mathrm{cross}}=(q_i,R_i,p_{i,a},p_{i,b})
$$

监督目标：

$$
(q_i,R_i,p_{i,a},p_{i,b})\rightarrow y_i
$$

推荐 Prompt 结构：

```text
[CROSSOVER]

[Query]
...

[Retrieved History]
...

[Parent A]
...

[Parent B]
...

[Output]
```

不能随机组合两个高度重复的 Parent。优先选择具有互补 Gold 内容的 Pair：

$$
\operatorname{Coverage}(p_a\cup p_b,y)
>
\max\left(
\operatorname{Coverage}(p_a,y),
\operatorname{Coverage}(p_b,y)
\right)
$$

至少过滤以下 Pair：

- 两个 Parent 归一化后相同；
- 一个 Parent 几乎完全包含另一个；
- 两个 Parent 都与 Gold 低相关；
- 两个 Parent 的有效内容没有互补性；
- Gold 相对两个 Parent 都没有明显改进空间。

Teacher 可以用于离线判断互补性或生成诊断信息，但第一版 Student 监督仍只使用最终标题纯文本。

### 6.3 联合 SFT 目标

训练集为：

$$
\mathcal D_{\mathrm{SFT}}
=
\mathcal D_{\mathrm{mut}}
\cup
\mathcal D_{\mathrm{cross}}
$$

损失为：

$$
\mathcal L_{\mathrm{SFT}}
=
\mathcal L_{\mathrm{mut}}
+
\lambda_{\mathrm{cross}}
\mathcal L_{\mathrm{cross}}
$$

第一版建议：

- Mutation/Crossover 样本比例约为 70%/30%；
- $\lambda_{\mathrm{cross}}=0.5$；
- Llama2-7B FP16 LoRA；
- Mutation 与 Crossover 共用一个 LoRA；
- 使用明确的任务前缀区分操作；
- Prompt token 全部 mask，只计算标题输出 token 的 Loss；
- 对 Query 做样本权重归一化，避免 Parent 数量多的 Query 主导训练。

## 7. 阶段二：Shared Dual-task Global IDPO

Global IDPO 继续优化同一个共享 Editor，不在这一阶段训练 per-user Editor Adapter，也不与 Ranker 联合更新。

### 7.1 Mutation 偏好对

固定相同 Mutation 条件：

$$
z_i^{\mathrm{mut}}=(q_i,R_i,p_{i,a})
$$

从当前 SFT Editor 采样：

$$
c_{i,1}^{\mathrm{mut}},\ldots,c_{i,K}^{\mathrm{mut}}
\sim
\pi_{\mathrm{SFT}}(\cdot\mid z_i^{\mathrm{mut}})
$$

用训练 Gold 离线打分：

$$
r(c,y)=\operatorname{ROUGE-L}(c,y)
$$

选择：

$$
c_{i,+}^{\mathrm{mut}}=\arg\max_c r(c,y_i),
\qquad
c_{i,-}^{\mathrm{mut}}=\arg\min_c r(c,y_i)
$$

### 7.2 Crossover 偏好对

固定相同 Crossover 条件：

$$
z_i^{\mathrm{cross}}=(q_i,R_i,p_{i,a},p_{i,b})
$$

采样：

$$
c_{i,1}^{\mathrm{cross}},\ldots,c_{i,K}^{\mathrm{cross}}
\sim
\pi_{\mathrm{SFT}}(\cdot\mid z_i^{\mathrm{cross}})
$$

在同一 Parent Pair 下选择 Crossover chosen/rejected。

禁止将 Mutation chosen 和 Crossover rejected 组成一对，因为它们不共享同一个条件 Prompt。

### 7.3 Pair 门控

只保留具有足够区分度的 Pair：

$$
r(c_i^+,y_i)-r(c_i^-,y_i)\ge 0.03
$$

同时过滤：

- chosen 与 rejected 归一化后相同；
- 空标题或格式异常；
- 输出包含 Prompt、JSON 残片或多余解释；
- 输出包含 Query 不支持的新事实；
- Crossover 输出只是机械拼接两个 Parent。

### 7.4 双任务 IDPO 目标

$$
\mathcal D_{\mathrm{IDPO}}
=
\mathcal D_{\mathrm{pref-mut}}
\cup
\mathcal D_{\mathrm{pref-cross}}
$$

$$
\mathcal L_{\mathrm{IDPO}}
=
\mathcal L_{\mathrm{DPO-mut}}
+
\mu_{\mathrm{cross}}
\mathcal L_{\mathrm{DPO-cross}}
$$

第一版建议 $\mu_{\mathrm{cross}}=0.5$。冻结的参考策略为 $\pi_{\mathrm{SFT}}$，训练策略以同一个 SFT LoRA 初始化。

## 8. 阶段三：Shared Task Ranker

Global IDPO 完成后冻结 Editor，再生成 Ranker 训练候选。这样 Ranker 看到的候选分布与正式推理使用的最终 Editor 一致。

### 8.1 Ranker 输入

$$
x_{i,j}^{\mathrm{rank}}
=(q_i,R_i,c_{i,j})
$$

推荐文本结构：

```text
[Query]
...

[Retrieved History]
...

[Candidate]
...
```

第一版不要输入候选来源、代数、Mutation/Crossover 标签，避免 Ranker 学习操作类型捷径。

### 8.2 候选分布

Shared Ranker 的训练池应覆盖：

- Base 候选；
- Mutation 子代；
- Crossover 子代；
- 第一代和第二代候选；
- 高流畅但低事实一致性的困难负例；
- 高相似度、低 margin 的困难候选。

### 8.3 监督和损失

第一版连续标签使用：

$$
r_{i,j}=\operatorname{ROUGE-L}(c_{i,j},y_i)
$$

Ranker 输出：

$$
s_{i,j}=f_\phi(q_i,R_i,c_{i,j})
$$

推荐使用 Listwise 与 Pairwise 联合损失：

$$
\mathcal L_{\mathrm{Ranker}}
=
\mathcal L_{\mathrm{Listwise}}
+
0.5\mathcal L_{\mathrm{Pairwise}}
$$

Pairwise 只使用真实 reward margin 不小于 0.03 的候选对。

不采用“Gold 为正、所有生成候选为负”的粗粒度二分类，因为该标签无法学习生成候选之间的相对质量。

### 8.4 推荐模型

- DeBERTa-v3-base 或当前已验证的 DeBERTa Ranker；
- 共享 Backbone + 共享评分 Head；
- max length 1024；
- 以完整 Query slate 为训练单位；
- group batch size 1；
- 使用梯度累积扩大有效 batch；
- 在验证集根据 Top-1 ROUGE-L 或 mean regret 选择 checkpoint。

## 9. 阶段四：Per-user Ranker Residual Head

每个目标用户只训练一个轻量 Residual Head，不复制完整 DeBERTa Ranker。

### 9.1 LOO 数据构造

目标用户历史为：

$$
H_u=\{(q_i,y_i)\}_{i=1}^{n_u}
$$

对每条历史构造：

$$
R_i=\operatorname{Retrieve}
\left(q_i,H_u\setminus\{(q_i,y_i)\}\right)
$$

然后使用冻结的最终 Editor 生成候选，并用该历史条目的 $y_i$ 标注候选质量。

### 9.2 残差结构

$$
s_u(q,R,c)
=
s_{\mathrm{shared}}(q,R,c)
+
\lambda_u s_{\mathrm{residual},u}(q,R,c)
$$

其中用户 Head 零初始化，初始行为等于 Shared Ranker。

历史数量自适应权重可定义为：

$$
\lambda_u=\frac{n_u}{n_u+\tau}
$$

历史较少时依赖 Shared Ranker，历史充分时增加用户偏好影响。

训练时：

- 冻结 DeBERTa Backbone；
- 冻结 Shared Head；
- 只更新用户 Residual Head；
- 使用与 Shared Ranker 相同的 listwise/pairwise 目标；
- 加入 L2 正则和 early stopping；
- 历史过少或有效候选对不足时回退到 Shared Ranker。

## 10. 推理阶段：两代遗传搜索

遗传算法真正发生在推理阶段。Ranker 是测试时不可见 Gold 的替代 Fitness Function。

### 10.1 初始种群

由 Base Llama2-7B 生成 4 个候选：

$$
P^{(0)}=\{c_1^{\mathrm{base}},\ldots,c_4^{\mathrm{base}}\}
$$

### 10.2 个性化适应度

$$
f_u(c)
=
s_{\mathrm{shared}}(q,R,c)
+
\lambda_u s_{\mathrm{residual},u}(q,R,c)
$$

没有用户 Head 时令残差项为零。

### 10.3 每代操作

第一版每代保留：

- Top-2 Elite；
- 4 个 Mutation 子代；
- 2 个 Crossover 子代。

$$
P^{(g+1)}
=
\operatorname{Elite}_2(P^{(g)})
\cup
\operatorname{Mutation}_4(P^{(g)})
\cup
\operatorname{Crossover}_2(P^{(g)})
$$

运行两代：

```text
P0：4个Base候选
 ↓ Ranker选择Parent
P1：2 Elite + 4 Mutation + 2 Crossover
 ↓ Ranker选择Parent
P2：2 Elite + 4 Mutation + 2 Crossover
 ↓ Ranker
最终Top-1
```

每代必须执行：

- 文本归一化与去重；
- 空输出和格式异常过滤；
- 最低事实一致性门控；
- 精英保留，防止下一代整体退化；
- Crossover Parent 不得相同或高度重复。

## 11. 模块训练顺序

严格采用顺序训练，不同时联合更新 Editor 和 Ranker：

```text
1. 构造真实 Base Parent
2. 构造 Mutation/Crossover SFT 数据
3. 训练 Shared Dual-task SFT Editor
4. 构造两类 on-policy 偏好对
5. 训练 Shared Dual-task Global IDPO Editor
6. 冻结最终 Editor，生成多代 Ranker 训练池
7. 训练 Shared Ranker
8. 冻结 Shared Ranker，为目标用户构造 LOO 候选
9. 训练 per-user Residual Head
10. 两代遗传搜索并评估最终 Top-1
```

不建议联合训练的原因包括：

- Editor 更新会改变候选分布；
- Ranker 同时更新会造成 Reward 漂移；
- Editor 容易对 Ranker 产生 Reward hacking；
- 难以区分增益来自生成还是选择；
- per-user Head 不应反向污染共享 Editor。

## 12. 实验与消融

### 12.1 Editor 消融

| 编号 | 方法                               | 验证目标             |
| ---- | ---------------------------------- | -------------------- |
| E0   | Base Llama2-7B                     | 原始能力             |
| E1   | Mutation-only SFT                  | 单 Parent 编辑贡献   |
| E2   | Dual-task SFT                      | Crossover SFT 贡献   |
| E3   | Dual-task SFT + Mutation-only IDPO | Mutation IDPO 贡献   |
| E4   | Dual-task SFT + Dual-task IDPO     | 完整双任务 IDPO 贡献 |

当前确定的正式方案为 E4，但仍需保留 E1/E2/E3 作为消融。

### 12.2 Ranker 消融

| 编号 | Ranker                       | 验证目标         |
| ---- | ---------------------------- | ---------------- |
| R0   | 无 Ranker，固定候选          | 生成基线         |
| R1   | Shared Ranker                | 任务级选择能力   |
| R2   | Shared Ranker + History 输入 | 上下文个性化贡献 |
| R3   | Shared + per-user Residual   | 用户适配贡献     |
| R4   | Gold Oracle                  | 候选池理论上限   |

### 12.3 搜索消融

| 编号 | 搜索方式                  | 验证目标         |
| ---- | ------------------------- | ---------------- |
| G0   | 单次 Editor               | 无搜索基线       |
| G1   | Best-of-N + Ranker        | 控制生成预算     |
| G2   | 一代 Mutation             | 单代变异贡献     |
| G3   | 两代 Mutation             | 迭代贡献         |
| G4   | 两代 Mutation + Crossover | 完整遗传操作贡献 |

Best-of-N 和遗传搜索必须使用相同总生成调用预算，否则无法证明增益来自遗传操作而不是更多采样。

## 13. 关键评估指标

最终生成指标：

- ROUGE-1；
- ROUGE-L；
- SacreBLEU；
- 有效输出率。

候选池和操作指标：

- Candidate Oracle ROUGE-L；
- Mutation 相对 Parent 的平均增益；
- Crossover 相对较强 Parent 的平均增益；
- 每代 Oracle 改变量；
- 候选唯一率；
- Crossover 进入最终 Top-1 的比例。

Ranker 指标：

- Pair Accuracy；
- NDCG@K；
- Spearman 相关性；
- Top-1 ROUGE-L；
- Selection Regret。

$$
\operatorname{Regret}
=
r(c_{\mathrm{oracle}})
-
r(c_{\mathrm{ranker}})
$$

最关键的诊断关系是：

- Oracle 低：Editor/遗传操作没有生成足够好的候选；
- Oracle 高但 Top-1 低：Ranker 选择能力不足；
- 第一代 Oracle 提升、第二代下降：迭代或 Parent 选择存在误差累积；
- Crossover Oracle 无增益：Crossover 数据或融合任务设计无效。

## 14. 主要风险

1. **Crossover 退化为重新生成**：模型忽略两个 Parent，只根据 Query 直接生成标题。
2. **Ranker 错误逐代放大**：第一代错误 Parent 会导致后续搜索方向偏离。
3. **训练和测试 Parent 分布不一致**：会直接削弱 Editor 改写能力。
4. **个性化 Head 过拟合**：目标用户历史较少时应加强收缩或回退共享 Ranker。
5. **遗传搜索只是增加计算量**：必须与 compute-matched Best-of-N 比较。
6. **ROUGE 奖励限制**：合理但措辞不同的候选可能被低估。
7. **IDPO 有效 Pair 不足**：候选重复或 reward margin 太小会导致实际更新次数不足。

## 15. 方法定位与推荐命名

建议将完整方法描述为：

> Dual-operator Editor Learning with Personalized Evolutionary Decoding

或者：

> Shared Dual-task SFT/IDPO Editor + Per-user Residual Fitness + Multi-generation Evolutionary Search

其中：

- SFT/IDPO 学习共享 Mutation 和 Crossover Operator；
- Shared Ranker 学习任务级 Fitness；
- per-user Residual Head 学习用户相对群体的选择偏差；
- 多代遗传循环在推理阶段真实执行，而不是只作为训练叙事。

## 16. 第一版推荐默认配置

| 项目                                |                                  默认值 |
| ----------------------------------- | --------------------------------------: |
| History                             |                              BM25 Top-8 |
| Editor                              |                     Llama2-7B FP16 LoRA |
| Editor Tasks                        |                    Mutation + Crossover |
| SFT Mutation/Crossover 比例         |                               70% / 30% |
| SFT Crossover loss weight           |                                     0.5 |
| IDPO Crossover loss weight          |                                     0.5 |
| Rollout candidates per fixed prompt |                                       4 |
| Pair minimum ROUGE-L margin         |                                    0.03 |
| Ranker                              | Shared DeBERTa + per-user Residual Head |
| Initial Base population             |                                       4 |
| Generations                         |                                       2 |
| Elites per generation               |                                       2 |
| Mutations per generation            |                                       4 |
| Crossovers per generation           |                                       2 |
| Final output                        |                            Ranker Top-1 |

这组配置的目标不是直接追求最大搜索预算，而是先以较低成本验证：双任务 Editor、用户 Ranker 和多代搜索是否分别提供可测量的增益。
