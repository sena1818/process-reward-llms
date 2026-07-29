# Step Preference Modeling for Process Reward Models

课程项目想法：**Process Reward in Large Language Models**

主论文：Lightman et al., 2023, *Let's Verify Step by Step*

---

## 0. 当前状态

这个项目现在已经不是一个纯粹的设想。最关键的数据前提已经在 PRM800K 上做过审计。

已有审计结果文件：

- `Process_Reward_LLMs/projects/step_preference_prm/outputs/audit/data_audit.json`
- `Process_Reward_LLMs/projects/step_preference_prm/outputs/audit/pair_stats.json`

重要数据统计：

| 统计项 | 数量 |
|---|---:|
| solution samples | 101,599 |
| unique problem texts | 11,293 |
| annotated step nodes | 733,768 |
| candidate labels | 1,040,685 |
| `+1` labels | 680,279 |
| `0` labels | 105,542 |
| `-1` labels | 254,864 |
| 有多个 candidate 的 exact-prefix nodes | 89,362 |
| 有多个 rating level 的 exact-prefix nodes | 68,075 |
| exact-prefix preference pairs 总数 | 379,652 |
| `+1 > -1` pairs | 236,539 |
| `+1 > 0` pairs | 54,766 |
| `0 > -1` pairs | 88,347 |

结论：

> PRM800K 中 exact-prefix preference pairs 的数量足够。本项目可以真实地使用“同一上下文下的 step preference”，而不需要依赖跨题配对、弱配对、Monte Carlo rollout 或 final-answer-based labeling。

---

## 1. 一句话总结

V0 从 PRM800K 的 `$+1/-1$` step-level human ratings 派生 **exact-prefix comparisons**，并训练一个 **hybrid pointwise + pairwise Process Reward Model (PRM)**。V1 才检验 `0` 是否应进入严格 ordinal model。目标是相比普通 pointwise PRM，在 step ranking、first-error localization 和 label efficiency 上表现更好。

更短的说法：

> 不再只把 `$+1/-1$` step 当作独立分类样本；还利用同一 prefix 下的 `$+1>-1$` 局部比较，以 Bradley--Terry-form pairwise ranking term 训练 PRM。

---

## 2. 这个项目修改的是哪一部分？

按照老师 slide 里的分类，这个项目主要属于：

```text
Step-score estimation / reward modeling pipeline
```

它不是严格意义上的：

```text
Post-training method, such as DPO
```

原因很重要：

- DPO / Step-DPO 训练的是 **generator / policy**。
- 本项目训练的是 **verifier / PRM**。

安全表述：

> Our pair term uses a Bradley--Terry preference form, but it trains a process reward model rather than a reference-regularized generator policy.

中文解释：

> 我们使用 Bradley--Terry 形式的 pairwise term 训练过程奖励模型，也就是训练“裁判”而不是训练“解题者”；因此它不是 DPO。

---

## 3. 动机

*Let's Verify Step by Step* 的核心贡献是证明了 **process supervision** 比只看最终答案的 **outcome supervision** 更适合复杂数学推理。

原论文中的 PRM 训练可以理解为一种 pointwise classification：

```text
输入：problem + previous steps + current step
目标：预测 current step 的过程标签（原论文为三类 label token）
```

这个做法很强，但它没有显式利用同一 prefix 下可派生的局部比较；V0 只使用语义最稳的 `$+1>-1$`：

```text
在同一道题、同一个 previous prefix 下：
  +1 下一步 > -1 下一步
```

`0` 的语义是“可接受但未推进”，是否足以支持对任意候选的严格 ordinal preference 是 V1 需要检验的假设，而不是 V0 的已知事实：

```text
V1 hypothesis: +1 > 0 > -1
```

核心 gap：

> In V0, opposite step labels under an identical reasoning context induce a clean local comparison; V1 separately tests whether neutral labels support an ordinal extension.

注意不要这样说：

```text
原论文丢弃了错误链。
```

这个说法不准确，因为 PRM800K 明明包含 negative step labels。

更严谨的说法是：

```text
原论文主要使用 pointwise supervision。本项目在同一份 human step labels 上额外提取 pairwise / ordinal supervision signal。
```

---

## 4. 研究问题

主问题：

> 在 pointwise classification 之外加入 exact-prefix pairwise preference supervision，是否能训练出更好的 PRM？

具体问题：

1. Hybrid pointwise + pairwise training 是否提高 step-level F1 / AUC？
2. 它是否提高 same-prefix pairwise ranking accuracy？
3. 它是否改善 first-error localization？
4. 在低数据量场景下，hybrid loss 的优势是否更明显？
5. 它是否也能改善 Best-of-N reranking，还是主要改善 process-level metrics？

---

## 5. 数据

主要数据集：

```text
PRM800K from Let's Verify Step by Step
```

本地已有数据位置：

```text
Process_Reward_LLMs/projects/step_preference_prm/data/raw/prm800k/
```

文件：

```text
phase1_train.jsonl
phase1_test.jsonl
phase2_train.jsonl
phase2_test.jsonl
```

label 含义：

| Label | 含义 | 在本项目中的角色 |
|---|---|---|
| `+1` | 正确，或者有效推进推理 | preferred step |
| `0` | 没明显错误，但没有有效推进 | intermediate step |
| `-1` | 错误或无效步骤 | rejected step |

V0/V1 对标签语义的处理：

```text
V0: +1 > -1 only
V1 hypothesis: +1 > 0 > -1
```

实现决策：

> 第一版实验可以先不加入 `0` label。V0 只使用最干净的 `+1` 和 `-1`，也就是只构造 `+1 > -1` pairs。`0` label 保留给后续 ablation，用来回答“中性步骤是否提供额外信息，还是引入噪声”。

对应实验分层：

```text
V0: only +1 / -1
  pointwise: +1 vs -1, ignore 0
  pairwise: only +1 > -1

V1: include 0 as ordinal middle label
  pointwise: test either a clearly labelled soft-target ablation
             (-1 -> 0.0, 0 -> 0.5, +1 -> 1.0)
             or a proper scalar cumulative-link ordinal model
  pairwise: only add +1 > 0 and 0 > -1 after validating that this
            strict order is supported by the annotation semantics
```

数据划分规则：

> 必须按 problem split，而不是按 step 随机切分。

原因：

> 如果同一道题的不同步骤同时出现在 train/test，模型可能记住题目特征，而不是真正学会判断 step correctness。

推荐划分：

```text
train: 80% problems
val:   10% problems
test:  10% problems
```

---

## 6. Preference Pair 构造

### 6.1 主配对方式：Exact-Prefix Pairs

这是本项目最干净、最核心的数据构造方式。

约束：

```text
same problem
same previous reasoning prefix
different candidate next steps
different ratings
```

也就是说，两个 candidate step 的上下文完全一样，唯一不同的是当前下一步。

例子：

```text
Problem:
  Solve 3x + 2 = 11.

Prefix:
  3x + 2 = 11

Candidate A:
  3x = 9
  rating = +1

Candidate B:
  3x = 13
  rating = -1

Preference:
  A > B
```

已有 pair 类型及 V0/V1 地位：

| Pair | 审计数量 | 训练地位 |
|---|---:|---:|
| `+1 > -1` | 236,539 | V0 主线，权重 1 |
| `+1 > 0` | 54,766 | V1；若使用，权重仅在 validation 选择 |
| `0 > -1` | 88,347 | V1；若使用，权重仅在 validation 选择 |

总 exact-prefix preference pairs：

```text
379,652
```

第一版主实验只使用：

```text
+1 > -1 pairs: 236,539
```

理由：

> `+1 > -1` 的语义最干净，噪声最低，足够支撑主实验。`+1 > 0` 和 `0 > -1` 更适合后续 ablation，因为 `0` 的含义介于“无错但没推进”和“低质量”之间，可能对不同模型有噪声。

### 6.2 辅助配对方式：First-Error Pairs

对于有 known first error 的 trajectory，可以构造：

```text
prefix before first error > prefix including first error
```

这个可以作为 first-error localization 的辅助实验，但不要替代 exact-prefix pairs 作为主监督信号。

### 6.3 可选：Synthetic Hard Negatives

可以从正确步骤出发，构造最小错误版本。例如：

```text
correct:
  3x + 2 = 11
  3x = 9

corrupted:
  3x + 2 = 11
  3x = 13
```

但这个方向有风险：

- 合成错误可能太简单，模型只学会识别表面错误；
- 合成错误分布可能和真实模型错误不同，引入 distribution shift；
- 如果用得太多，评估可能虚高。

因此它只能作为 optional extension，不作为主贡献。

---

## 7. 模型架构

建议分阶段实现，不要一开始就依赖大 GPU。

### Tier 0：Sanity Baseline

最简单版本：

```text
frozen sentence embedding -> small MLP reward head
```

可选 encoder：

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`
- `intfloat/e5-small-v2`

用途：

- 验证数据加载；
- 验证 pair construction；
- 快速得到第一组结果；
- 帮助调试 loss 和 metrics。

缺点：

- 数学推理能力弱；
- 更像文本相似度模型，不太像真正 LLM-based PRM。

### Tier 1：主实验模型

推荐作为本项目主力：

```text
DeBERTa-base / RoBERTa-base encoder
+ scalar reward head
```

输入格式：

```text
Problem:
{problem}

Previous steps:
{prefix}

Candidate step:
{candidate}
```

输出：

```text
scalar reward r(context, candidate_step)
```

为什么适合作为主模型：

- 本地 RTX 4060 8GB / RTX 3060 16GB 更可控；
- 可以 end-to-end training；
- 比 frozen embedding baseline 更像 reward model；
- 训练成本足够低，可以做 ablation 和 label-efficiency curve。

建议训练设置：

```text
max_seq_len: 512 first, 1024 only if memory allows
batch_size: tune according to GPU memory
gradient_accumulation: use if batch is small
mixed_precision: fp16 or bf16 if available
```

### Tier 2：主 LLM-PRM

严格主实验使用：

```text
Qwen/Qwen2.5-Math-1.5B base + bf16 LoRA + scalar reward head
```

在 32 GB GPU 上，1.5B 的 bf16 LoRA 不需要先引入 4-bit 量化误差。固定主设置为 `max_seq_len=2048`、LoRA rank 16、alpha 32、dropout 0.05、gradient checkpointing；只有长度审计/长链 ablation 有必要时才跑 4096。若 4096 LoRA 在 physical batch 1 仍 OOM，或后续升级 7B，才使用 QLoRA，并将其标为单独的 adaptation/scale extension。

不要做：

```text
7B full fine-tuning
```

这对本项目完全没有必要，也不现实。

---

## 8. Loss Functions（摘要；完整推导见配套设计文档）

本节只保留最终 objective 与实验口径；严格推导以 [PRM_loss_design.md](PRM_loss_design.md) 为准。该文档从统一 Bernoulli score model 出发，在 discordant-pair 条件下推出 Bradley--Terry probability，再用 weighted composite likelihood 严格得到 $L_{\rm pt}+\lambda L_{\rm pair}$。

> **实现状态（2026-07-30）**：训练代码已经迁移到统一的
> exact-prefix node cohort。每个 candidate 在一个 batch 中只 forward
> 一次，pointwise/pairwise 均先在 node 内求 mean、再在 node 间求
> mean；hybrid 在同一批 nodes 上组合两项损失。实现已通过本地数据与
> 单元测试，但 Qwen-LoRA 的真实 GPU smoke/pilot 尚需在 UniCluster 上
> 完成，因此结果表仍不能写入未经运行的数值。

本节的训练单位是一个 exact-prefix annotated node，而不是扁平的 candidate 或 Cartesian pair。令

$$
n=(c_n,A_n),\quad A_n=\{(s_{ni},y_{ni})\}_{i=1}^{a_n},\quad
y_{ni}\in\{-1,0,+1\},
$$

其中 V0 先过滤 `$+1/-1$`：

$$
\mathcal I_n=\{i:y_{ni}\in\{-1,+1\}\},\qquad
b_n=|\mathcal I_n|,\qquad z_{ni}=\mathbb 1[y_{ni}=+1].
$$

再定义

$$
P_n=\{i:z_{ni}=1\},\quad R_n=\{j:z_{nj}=0\},\quad
m_n=|P_n||R_n|,
$$

并令 $\mathcal N$ 为同时包含正、负候选的 nodes。主 pointwise、pairwise、hybrid 三者都使用相同的 $\mathcal N$、同一个 scalar score $r_\theta(c,s)\in\mathbb R$，以及同一组 candidate ratings。这样 pair 是从已有人工 ratings 派生出的比较，不能被当作新的独立人工标注。

### 8.1 Pointwise Loss

V0 使用 binary BCE，忽略 `0`：

```text
+1 -> positive
-1 -> negative
 0 -> ignored in V0
```

对单个 candidate：

$$
\ell_{\rm BCE}(r,z)
=-[z\log\sigma(r)+(1-z)\log(1-\sigma(r))]
=\operatorname{softplus}(r)-zr,
$$

$$
\frac{\partial\ell_{\rm BCE}}{\partial r}=\sigma(r)-z.
$$

因此 `$+1` 被推向正 score，`-1` 被推向负 score，提供跨 context 的绝对锚定。为避免 candidate-rich node 主导训练，主 loss 必须先做 node 内平均：

$$
\ell_{\rm pt}(n)=\frac{1}{b_n}\sum_{i\in\mathcal I_n}
\ell_{\rm BCE}(r_{ni},z_{ni}),
$$

$$
L_{\rm pt}=\frac{1}{|\mathcal N|}\sum_{n\in\mathcal N}\ell_{\rm pt}(n).
$$

V1 若加入 neutral，主线兼容的选择是 **soft-label BCE**，而不是把它笼统称为 ordinal regression：

$$
\tilde z\in\{0,0.5,1\},\qquad
\ell_{\rm soft}(r,\tilde z)
=-[\tilde z\log\sigma(r)+(1-\tilde z)\log(1-\sigma(r))],
$$

$$
\frac{\partial\ell_{\rm soft}}{\partial r}=\sigma(r)-\tilde z.
$$

它隐含假设：neutral 的目标概率恰为 $0.5$，因而将 score 推向 $r=0$。但 PRM800K 的 `0` 更接近“没有明显错误但没有有效推进”，并不等于“半对半错”；所以这是需要显式检验的 V1 假设，不能作为 V0 事实。

若保留 3-class CE，它必须被标为**独立对照模型**：它使用 3 维 head，既不共享 scalar $r_\theta$，也不进入 pairwise 或 hybrid 对照，因而不属于主线公平比较。

### 8.2 Pairwise Ranking Loss

令 $p_{ni}=\sigma(r_{ni})$。对同一 node 中的两个 binary outcomes 作 conditional-independence working assumption，并条件在 $Z_{ni}\ne Z_{nj}$ 上，则

$$
\begin{aligned}
\Pr_\theta(Z_{ni}=1,Z_{nj}=0\mid Z_{ni}\ne Z_{nj},\cdots)
&=
\frac{p_{ni}(1-p_{nj})}
{p_{ni}(1-p_{nj})+(1-p_{ni})p_{nj}}\\
&=
\sigma(r_{ni}-r_{nj}).
\end{aligned}
$$

所以，对同一 node $n$ 中的 preferred candidate $s_{ni}$ 和 rejected candidate $s_{nj}$，单 pair conditional NLL 正是 Bradley--Terry / RankNet 形式：

$$
\ell_{\rm BT}(r_{ni},r_{nj})
=-\log\sigma\big(r_\theta(c_n,s_{ni})-r_\theta(c_n,s_{nj})\big)
=\operatorname{softplus}\big[-(r_{ni}-r_{nj})\big].
$$

令 $\Delta=r_{ni}-r_{nj}$，则

$$
\frac{\partial\ell_{\rm BT}}{\partial r_{ni}}=\sigma(\Delta)-1,
\qquad
\frac{\partial\ell_{\rm BT}}{\partial r_{nj}}=1-\sigma(\Delta).
$$

因此分数相同的正/负候选会被直接拉开。严格的 dataset aggregation 不是扁平 pair mean，而是：

$$
\ell_{\rm pair}(n)=\frac{1}{m_n}
\sum_{i\in P_n}\sum_{j\in R_n}
\ell_{\rm BT}(r_{ni},r_{nj}),
$$

$$
L_{\rm pair}=\frac{1}{|\mathcal N|}
\sum_{n\in\mathcal N}\ell_{\rm pair}(n).
$$

这使一个 $1\times1$ node 与一个 $3\times8$ node 的总梯度权重相同。若 $m_n$ 很大，可均匀抽 $K$ 对：

$$
\widehat\ell_{\rm pair}(n)=\frac1K\sum_{k=1}^K\ell_{\rm BT}(i_k,j_k),
\qquad (i_k,j_k)\sim\operatorname{Uniform}(P_n\times R_n),
$$

并有 $\mathbb E[\widehat\ell_{\rm pair}(n)]=\ell_{\rm pair}(n)$。

V0 只使用最干净的 `$+1>-1$` pairs。V1 若加入 `$+1>0`、`0>-1`，pair weight $w$ 不可拍定为 $0.5$；它应是 validation-only hyperparameter。加权项

$$
\ell_{\rm BT}^{(w)}=-w\log\sigma(r^+-r^-),\qquad
\frac{\partial\ell_{\rm BT}^{(w)}}{\partial r^+}=w[\sigma(r^+-r^-)-1]
$$

只是线性缩放梯度的 heuristic surrogate，不再是具有“半个偏好”语义的独立 likelihood。

为什么不能只用 pairwise：对任意仅依赖 context 的函数 $b(c)$，令 $r'(c,s)=r(c,s)+b(c)$，则

$$
r'(c,s^+)-r'(c,s^-)=r(c,s^+)-r(c,s^-).
$$

所以 $L_{\rm pair}(r')=L_{\rm pair}(r)$。pure pairwise 的每道题的 scores 可以整体任意平移，不存在可识别的跨题统一 threshold；这正是 first-error localization 必须保留 pointwise anchoring 的数学原因。

### 8.3 主方法：Hybrid Loss

定义每个 node 的 geometric-mean point component $C_{{\rm pt},n}$ 和 pair conditional component $C_{{\rm pair},n}$。node-balanced weighted composite likelihood 为

$$
\operatorname{CL}_\lambda(\theta)
=
\prod_{n\in\mathcal N}
\left[
C_{{\rm pt},n}(\theta)
C_{{\rm pair},n}(\theta)^\lambda
\right]^{1/|\mathcal N|}.
$$

取负对数后严格得到

$$
L_\lambda(\theta)=L_{\rm pt}(\theta)+\lambda L_{\rm pair}(\theta),
\qquad\lambda\ge0.
$$

这也是 multi-objective scalarization；但不是 full joint likelihood，因为 node 内 Cartesian pairs 由同一组 point labels 派生、彼此高度相关。它不创造新的独立标注信息，而是重新加权 absolute validity 与 exact-prefix local separation。

所有 score 初始化为零时：

$$
\ell_{\rm BCE}(0,z)=\ell_{\rm BT}(0,0)=\log2.
$$

由于两项都按 node 平均，且 score 全为零时各单项都为 $\log2$，$\lambda=1$ 是透明的起点；但这不保证实际梯度尺度相同，也不能从理论推出最优 $\lambda$。$\lambda$ 是 validation-only 的 composite-risk 权重，而非把不同指标硬相加。其梯度为

$$
\nabla_\theta L_\lambda
=\nabla_\theta L_{\rm pt}+\lambda\nabla_\theta L_{\rm pair}.
$$

固定 sweep：

$$
\lambda\in\{0,0.1,0.3,0.5,1.0\}.
$$

$\lambda=0$ 是同一训练框架下的 pointwise baseline；pure pairwise 则单独最小化 $L_{\rm pair}$。不预设“默认 0.3”或“0.5 更合理”。

若 first-error 是预注册主指标，则每个 seed 只能在 validation 上按如下规则选择 lambda：最大化 first-error within-1；并列时选 Brier 更小者；仍并列时选更小 $\lambda$。选择冻结后，test 只评估一次。不同 objective 的 raw training loss 不作横向比较；共同 test cohort 上的 classification、node-macro ranking、calibration 与 first-error 才是比较对象。

完整推导、Qwen-LoRA 选择和 node-level 实现接口见：

- [`idea/PRM_loss_design.md`](PRM_loss_design.md)（报告中应优先引用的 loss 推导）
- [`notes/2026-07-17_qwen_lora_prm_hybrid_loss_derivation_zh.md`](../notes/2026-07-17_qwen_lora_prm_hybrid_loss_derivation_zh.md)

---

## 9. 实验设计

### Experiment 0：Data Audit

已经完成。

报告里一定要放 audit table，因为它直接回答老师 slide 里的问题：

```text
Do I have the right data?
```

结论是：

> 有，而且数量足够支持 exact-prefix preference modeling。

### Experiment 1：Pointwise PRM Baseline

训练普通 pointwise PRM：

```text
input: problem + prefix + candidate step
target: step label
loss: pointwise
```

目的：

> 小规模复现 `Let's Verify Step by Step` 的 PRM 训练思路。

### Experiment 2：Pairwise-Only PRM

只用 pairwise loss：

```text
input: exact-prefix preference pairs
loss: pairwise ranking
```

目的：

> 检查 preference pairs 本身是否足以学到有意义的 step ranking。

预期风险：

> Pairwise-only 可能 pairwise accuracy 很好，但 threshold-based first-error localization 较差。

### Experiment 3：Hybrid PRM

主方法：

```text
loss = pointwise loss + lambda * pairwise loss
```

和 Experiment 1 使用同样的 train/val/test problem split、同一训练 node cohort、相同 candidate-forward 预算与相同 Qwen-LoRA 配置，保证公平比较。

推荐主实验矩阵：

| 实验 | 是否使用 `0` | Pair 类型 | `lambda` |
|---|---|---|---|
| Pointwise V0 | 否 | 无 | 0 |
| Pairwise V0 | 否 | `+1 > -1` | pairwise-only |
| Hybrid V0 | 否 | `+1 > -1` | `{0, 0.1, 0.3, 0.5, 1.0}` |
| Hybrid V1 ablation | 是 | `+1 > -1`, `+1 > 0`, `0 > -1` | optional；`w` 与 lambda 均只在 validation 调参 |

第一阶段先完成 V0。V1 是加分项，不要让 `0` label 影响主线可控性。

### Experiment 4：Label Efficiency

先抽取同一批 annotated nodes，再用不同 node budget 训练 pointwise 和 hybrid：

```text
10%, 30%, 50%, 100% of training nodes
```

假设：

> Hybrid training 在低数据量时可能更有效，因为同一批 ratings 同时提供绝对标签与局部顺序；这不是额外的人类标注，优势必须在同一 node budget 下检验。

### Experiment 5：Optional Best-of-N Reranking

只作为外部 downstream metric，不作为唯一评价。

流程：

1. 为每道题准备多个 candidate solutions；
2. PRM 对每个 solution 的每一步打分；
3. 聚合 step scores 得到 solution score；
4. 选择最高分 solution；
5. 检查 final-answer accuracy。

默认聚合方式：

```text
solution_score = minimum step score
```

理由：

> 一条推理链里只要有一个严重错误步骤，就应该强烈惩罚整个解法。

---

## 10. Evaluation Metrics

不要只做 Best-of-N。

### 10.1 Step-Level Metrics

指标：

- accuracy
- macro F1
- AUROC if using scalar scores

回答的问题：

> 模型能不能判断单步推理是否正确？

### 10.2 Pairwise Metrics

在 held-out exact-prefix pairs 上计算：

```text
accuracy = mean[ r(c, s+) > r(c, s-) ]
```

回答的问题：

> 模型是否学会了本项目提出的 preference structure？

### 10.3 First-Error Localization

对有 known first wrong step 的 trajectory：

```text
predicted first error = first step with score below threshold
```

阈值定义方式：

1. 在 validation set 上收集每个 step 的 score。
2. V0 中只用 `+1` 和 `-1` steps 校准阈值，忽略 `0`。
3. 在一组候选阈值上搜索，例如 score 分位点或等距网格。
4. 选择使 validation 指标最好的阈值。
5. 固定这个阈值，在 test set 上报告 first-error localization。

推荐主阈值选择准则：

```text
choose threshold that maximizes validation first-error within +/-1
```

备选准则：

```text
choose threshold that maximizes validation step-level F1
```

报告里可以同时说明：

> Threshold is selected only on the validation set and then fixed for test evaluation.

指标：

- exact match
- within +/- 1 step
- mean absolute error
- detection F1

回答的问题：

> 模型是否真的像 process verifier，而不只是 outcome predictor？

### 10.4 Best-of-N Accuracy

可选外部指标：

```text
N = 4, 8, 16
```

重要 caveat：

> Best-of-N 只能说明最终选出的答案是否正确，不能单独证明模型理解了过程。这一点和 `Lessons of Developing PRMs` 直接相关。

---

## 11. 主要结果表格模板

主结果表：

| Model | Step F1 | Pairwise Acc | First Error Exact | First Error +/-1 | Optional BoN |
|---|---:|---:|---:|---:|---:|
| Pointwise PRM | | | | | |
| Pairwise PRM | | | | | |
| Hybrid PRM | | | | | |

Label efficiency 表：

| Train Budget | Pointwise F1 | Hybrid F1 | Pointwise FE +/-1 | Hybrid FE +/-1 |
|---:|---:|---:|---:|---:|
| 10% | | | | |
| 30% | | | | |
| 50% | | | | |
| 100% | | | | |

---

## 12. 与其他 idea / 论文的区别

### 12.1 和 Math-Shepherd 的区别

Math-Shepherd 用 Monte Carlo rollouts 从 final-answer success 反推 step quality。

本项目：

- 不做 rollout；
- 不从 final answer 推断 step label；
- 直接使用 PRM800K 的 human step labels；
- 从这些 labels 中提取 local step preferences。

一句话：

> Math-Shepherd estimates future success; this project extracts local step preferences from existing human process labels.

### 12.2 和 Step-DPO 的区别

Step-DPO 训练 generator / policy，让模型更会生成正确推理步骤。

本项目训练 PRM / verifier，让模型更会判断步骤好坏。

一句话：

> Step-DPO trains the solver; this project trains the judge.

### 12.3 和 Q-value PRM 的区别

Q-value PRM 学的是 future value ranking：

```text
从当前 state/action 出发，未来得到正确最终答案的可能性有多高？
```

本项目学的是 local human preference：

```text
同一 prefix 下，哪个 candidate next step 被人类标得更好？
```

一句话：

> Q-value PRM ranks future value; this project ranks local human-rated step quality.

### 12.4 和 RetrievalPRM 的区别

RetrievalPRM 在 inference 时加入检索到的相似题或相似 step，帮助 PRM 判断 OOD 样本。

本项目修改的是 training objective：

```text
pointwise supervision -> pointwise + pairwise / ordinal supervision
```

一句话：

> RetrievalPRM changes the context at inference time; this project changes the supervision objective.

### 12.5 和早期 Learning from Failures 想法的区别

早期想法容易变成：

```text
从 final-answer wrong trajectories 中自动构造错误标签。
```

这会接近 Math-Shepherd / value estimation。

现在的版本更干净：

```text
从 existing human step labels 中构造 exact-prefix preference pairs。
```

---

## 13. 硬件与可行性

本地显卡：

```text
RTX 4060 8GB
```

可能可用的另一张本地卡：

```text
RTX 3060 16GB
```

本地可行：

- data audit；
- pair construction；
- frozen embedding + MLP baseline；
- DeBERTa/RoBERTa-base reward model；
- 子集上的 label-efficiency experiments。

本地有风险：

- Qwen2.5-Math-1.5B bf16 LoRA at 2048/4096；
- sequence length 超过 1024 的 RoBERTa encoder；
- batch size 太大；
- 大规模 Best-of-N generation。

本地不现实：

- 7B full fine-tuning；
- 不量化的 7B PRM training；
- 大规模 7B/13B verifier training。

建议：

> 先把本地 data pipeline、node-balanced loss 与小规模 smoke 跑通；随后租 32GB GPU 执行 Qwen-1.5B LoRA 主实验，而不是把 Qwen 当作可有可无的 extension。

如果租卡：

- 24GB GPU：适合 1.5B，很小心地做 7B QLoRA；
- 48GB GPU：7B QLoRA 更稳；
- A100/H100：对这个 seminar project 来说一般没必要，除非要做很大的 LLM-scale experiment。

---

## 14. 4-8 页 Paper Report 结构

建议结构：

### 1. Introduction

- PRM vs ORM；
- `Let's Verify Step by Step` 的贡献；
- gap：pointwise step labels 没显式利用 pairwise preference structure；
- 本项目贡献。

### 2. Related Work

- Lightman et al. PRM；
- DPO / Step-DPO；
- Q-value PRM；
- Lessons of Developing PRMs；
- 简短说明和 Math-Shepherd / RetrievalPRM 的边界。

### 3. Method

- PRM800K label structure；
- exact-prefix pair construction；
- pointwise loss；
- pairwise ranking loss；
- hybrid loss。

### 4. Experimental Setup

- data audit result；
- model architecture；
- problem-level train/val/test split；
- hardware constraints；
- implementation details。

### 5. Results

- step-level F1/AUC；
- pairwise accuracy；
- first-error localization；
- label-efficiency curve；
- optional Best-of-N。

### 6. Discussion

- 哪些指标提升，哪些没提升；
- 为什么 Best-of-N 不够；
- 和 Math-Shepherd、Step-DPO、Q-value PRM 的区别；
- 失败情况和限制。

### 7. Conclusion

- 简洁总结；
- limitations；
- future work：LLM-based PRM、retrieval、synthetic hard negatives。

---

## 15. 实现顺序

项目目录已经存在：

```text
Process_Reward_LLMs/projects/step_preference_prm/
```

下一步实现顺序：

1. 构建 problem-level train/val/test split。
2. materialize pointwise examples。
3. materialize exact-prefix pairwise examples。
4. 实现 DeBERTa/RoBERTa reward model。
5. 训练 pointwise baseline。
6. 训练 pairwise-only model。
7. 训练 hybrid model。
8. 评估 step-level、pairwise、first-error metrics。
9. 加 label-efficiency curves。
10. 如果时间还有，再加 optional Best-of-N。

已有脚本：

```text
scripts/00_audit_data.py
scripts/01_build_splits.py
scripts/02_build_pairs.py
scripts/download_prm800k.py
scripts/inspect_prm800k.py
```

---

## 16. 主要风险与 fallback

### Risk 1：Hybrid 没有提升 Step F1

Fallback：

- 报告 pairwise accuracy 和 first-error localization；
- 分析 pairwise supervision 是否改善 ranking，但没有改善 classification calibration。

### Risk 2：Pairwise-only 表现不好

Fallback：

- 这是可以预期的；
- 强调主方法本来就是 hybrid；
- 解释 threshold-based error detection 需要 pointwise calibration。

### Risk 3：Best-of-N 没提升

Fallback：

- 不要隐藏；
- 说明 process-level metrics 和 outcome-level metrics 不是一回事；
- 连接到 `Lessons of Developing PRMs` 的结论。

### Risk 4：LLM-PRM 太贵

Fallback：

- 使用 DeBERTa/RoBERTa-base 作为最终模型；
- 把方法表述为 architecture-agnostic，可扩展到 LLM-based PRM。

---

## 17. 最终要争取的 claim

精准、可防守的 claim：

> We extend the process-supervised PRM setup from `Let's Verify Step by Step` by converting human step labels into exact-prefix ordinal preferences. In a small-scale reproduction, a hybrid pointwise + pairwise PRM improves process-level ranking and first-error localization, especially under limited label budgets.

中文版本：

> 本项目在 `Let's Verify Step by Step` 的 process-supervised PRM 设置上，把 human step labels 转换成 exact-prefix ordinal preferences。小规模复现实验中，hybrid pointwise + pairwise PRM 有望改善 step ranking 和 first-error localization，尤其在低标注数据量下更具优势。

避免过度声称：

```text
我们超过了 OpenAI 原论文的完整 PRM。
我们解决了自动过程监督。
我们训练出了比 Step-DPO 更好的 generator。
我们可以完全替代人工标注。
```

最强 framing：

```text
same data source,
more informative supervision structure,
process-aware evaluation.
```

中文说法：

```text
同一份数据来源，
更充分利用标签结构，
用真正 process-aware 的指标评估。
```

---

## 18. 参考文献

- Lightman et al., 2023, *Let's Verify Step by Step*: https://arxiv.org/abs/2305.20050
- PRM800K dataset: https://github.com/openai/prm800k
- Rafailov et al., 2023, *Direct Preference Optimization*: https://arxiv.org/abs/2305.18290
- Lai et al., 2024, *Step-DPO*: https://arxiv.org/abs/2406.18629
- Li et al., 2024, *Process Reward Model with Q-Value Rankings*: https://arxiv.org/abs/2410.11287
