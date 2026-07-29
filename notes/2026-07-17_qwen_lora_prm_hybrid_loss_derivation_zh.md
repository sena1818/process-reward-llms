# Qwen-1.5B LoRA PRM：Pointwise、Pairwise 与 Hybrid 的严格推导

> **更新说明（2026-07-25）**：本文件保留 Qwen/LoRA、长度与实现决策。loss 的 canonical 数学叙述已移至 [`idea/PRM_loss_design.md`](../idea/PRM_loss_design.md)：它从统一 Bernoulli score model 推出 discordant-pair conditional probability，并用 weighted composite likelihood 严格得到 $L_{\rm pt}+\lambda L_{\rm pair}$。写报告时以该文档为准。

## 0. 已固定的主设置

```text
backbone: Qwen/Qwen2.5-Math-1.5B（base，不是 Instruct）
adaptation: bf16 LoRA + 新建 scalar reward head
main length: 2048 tokens
length sensitivity: 4096 tokens
V0 labels: 仅 +1 / -1；0 作为之后的 ordinal 扩展
```

本项目训练的是显式的 **process reward model / verifier**。给定题目、已有推理和候选下一步，它输出一个标量；它不直接改变 Qwen 的生成概率。因此本文借用 DPO 的 Bradley--Terry reward-model 推导作为理论参照，但不把本项目称为 DPO。

本文中的“严格”指三个 objective 有相同的数据单位、归一化和训练/评估协议；不表示已经证明 hybrid 必然优于任何单一 loss。hybrid 的 ranking--calibration 效果必须由 test 实验验证。

---

## 1. LoRA 与 QLoRA：为什么 1.5B 主实验使用 LoRA

### 1.1 两者的数学区别

对一个线性层，令预训练权重为

$$
W_0\in\mathbb R^{d_{\rm out}\times d_{\rm in}}.
$$

LoRA 冻结 $W_0$，只学习低秩更新：

$$
\Delta W=\frac{\alpha}{r}BA,
\qquad A\in\mathbb R^{r\times d_{\rm in}},\quad
B\in\mathbb R^{d_{\rm out}\times r},
$$

$$
h\mapsto\left(W_0+\frac{\alpha}{r}BA\right)h.
$$

QLoRA 可训练的仍是同样的 $A,B$；区别仅在冻结基座先被 4-bit 量化，计算时使用反量化近似：

$$
h\mapsto\left(\widetilde W_0+\frac{\alpha}{r}BA\right)h,
\qquad \widetilde W_0=\operatorname{dequant}(Q_{4\text{-bit}}(W_0)).
$$

所以 QLoRA 不是另一种偏好 loss，而是 “LoRA + 量化的 base”。它降低显存，但额外引入 $W_0\to\widetilde W_0$ 的近似。QLoRA 在原作者的任务中性能很好，但这不是对本 PRM 数据的无损保证。[QLoRA](https://arxiv.org/abs/2305.14314)

### 1.2 主决策

1.5B bf16 冻结权重大约为 $1.5\times10^9\times2\approx3$ GB；LoRA adapter 和 reward head 很小，也不为冻结 base 保存 Adam states 或 gradients。32 GB 下，2048/4096 训练的主要显存压力来自 activation，而不是这 3 GB base。

因此主实验用 **bf16 LoRA**，不先用 QLoRA：

- 不引入量化误差，使比较只改变 supervision objective；
- 调试更简单，不依赖 4-bit loading/kernel 细节；
- LoRA 不是弱微调：它冻结 base、学习低秩更新，原论文在多个下游任务中观察到可与全参微调相当或更好；本项目当然仍应实测。[LoRA](https://arxiv.org/abs/2106.09685)

建议统一配置：

```text
r=16, alpha=32, dropout=0.05, target_modules=all-linear
bf16, gradient_checkpointing=true, use_cache=false
```

QLoRA 只在两种情况下启用：

```text
1. 4096-token、physical batch=1 的 LoRA pilot 仍 OOM；
2. 后续扩展到 7B。
```

LoRA 与 QLoRA 不能混在同一张“loss 效果”主表中。若切换，必须标为 backbone/adaptation extension。

### 1.3 Qwen 的长度能力

该 Qwen2.5-Math 1.5B base config 给出 4096 positions；故 2048 是主长度、4096 是可行的长度敏感性实验，但不是无限历史窗口。[Qwen config](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/blob/main/config.json) 它比 RoBERTa-base 的 514 positions 长很多。[RoBERTa config](https://huggingface.co/FacebookAI/roberta-base/blob/main/config.json)

Qwen base 是此项目的正确选择：官方模型卡明确将 base 作为更适合微调的起点，而 Instruct 面向聊天。[Qwen model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)

---

## 2. 数据单位：一个 annotated node，不是一条 Cartesian pair

令一个原始标注 node 为

$$
n=(c_n,A_n),\qquad c_n=(q_n,p_n),
$$

其中 $q_n$ 是题目，$p_n=(u_{n1},\ldots,u_{nt_n})$ 是已经选中的推理前缀，且

$$
A_n=\{(s_{ni},y_{ni})\}_{i=1}^{a_n}
$$

是该**完全相同** `problem + prefix` 下的所有候选下一步及人工 rating。V0 保留 $y_{ni}\in\{-1,+1\}$，并写为

$$
z_{ni}=\mathbb 1[y_{ni}=+1]\in\{0,1\}.
$$

定义正、负候选集合和 pair 数：

$$
P_n=\{i:z_{ni}=1\},\quad R_n=\{j:z_{nj}=0\},\quad
m_n=|P_n||R_n|.
$$

严格对照使用的 cohort 是

$$
\mathcal N=\{n:|P_n|>0\ \text{and}\ |R_n|>0\}.
$$

若一个 node 有两个正项、八个负项，可派生 $2\times8=16$ 个 pairs；但这不是 16 次独立的人类偏好标注，而是同一个 node 的十个 candidate ratings 所蕴含的 16 个逻辑比较。

因此 main pointwise 也必须用这个 $\mathcal N$，而不能用全部扁平 `$+1/-1$` samples：

```text
pointwise：同一批 node 内每个 candidate 的绝对标签；
pairwise：同一批 node 内 candidate 的相对顺序；
hybrid：同一批 node 的两种监督。
```

“pointwise 使用全 PRM800K 数据”的模型可以另报为 more-data baseline，但不能作为严格 loss 对照。

---

## 3. 一个共享的 Qwen reward score，以及 recent-prefix 截断

三个训练方式共用一个 scalar function：

$$
r_\theta(c_n,s_{ni})\in\mathbb R.
$$

Qwen 加 LoRA 后输出最后一个非 padding token 的 hidden state $h_{\rm last}$，新的 reward head 为

$$
r_\theta(c,s)=w_r^\top h_{\rm last}(c,s)+b_r.
$$

V0 的绝对正例概率统一定义为

$$
p_\theta(z=1\mid c,s)=\sigma(r_\theta(c,s))
=\frac{1}{1+e^{-r_\theta(c,s)}}.
$$

### recent-prefix 的精确定义

输入自然顺序为：

```text
question -> oldest step -> ... -> newest step -> candidate
```

若普通右截断超长文本，最容易删掉的恰是 newest steps 或 candidate。这会损害“当前步骤是否接得上最近一步”的 PRM 判断。

设 tokenized question、历史和 candidate 为

$$
Q=\operatorname{tok}(q),\quad
H=\operatorname{tok}(u_1\Vert\cdots\Vert u_t),\quad
S=\operatorname{tok}(s).
$$

总长度限制 $L\in\{2048,4096\}$。candidate 预算为 $C$，question 的开头预算为 $Q_{\max}$，section labels/special tokens 长度为 $b_{\rm sep}$：

$$
b_s=\min(|S|,C),\quad b_q=\min(|Q|,Q_{\max}),
$$

$$
b_h=L-b_s-b_q-b_{\rm sep}.
$$

最终真正喂入模型的是

$$
\operatorname{pack}\left(
Q_{1:b_q},\;H_{|H|-b_h+1:|H|},\;S_{1:b_s}
\right).
$$

即：保留 question 开头、保留 candidate、保留历史 token 后缀（最好按完整 steps），也就是离 candidate 最近的推理过程。它不否认旧步骤可能重要，而是在有限 2048/4096 budget 下给局部 verifier 合理优先级。

必须先用同一个 tokenizer/pack 函数做长度审计，报告 $p50,p90,p95,p99$ 和

$$
\Pr\left(|Q|+|H|+|S|+b_{\rm sep}>L\right)
$$

以及 question/prefix/candidate 的截断率。2048 截断不显著时，不应为了“更长”盲跑 4096。

---

## 4. Pointwise：绝对 step-validity 的 Bernoulli loss

对单个 candidate，令 $r=r_\theta(c,s)$，$p=\sigma(r)$。Bernoulli 模型为

$$
\Pr_\theta(z\mid c,s)=p^z(1-p)^{1-z}.
$$

负对数似然：

$$
\ell_{\rm BCE}(r,z)
=-[z\log\sigma(r)+(1-z)\log(1-\sigma(r))].
$$

用 $\log\sigma(r)=r-\operatorname{softplus}(r)$ 化简：

$$
\ell_{\rm BCE}(r,z)=\operatorname{softplus}(r)-zr.
$$

这正对应 `binary_cross_entropy_with_logits`。它对 score 的梯度是

$$
\frac{\partial\ell_{\rm BCE}}{\partial r}=\sigma(r)-z.
$$

故在 $r=0$：正例梯度为 $-1/2$，梯度下降会增大 score；负例梯度为 $+1/2$，会减小 score。

不要扁平平均 candidates。node-balanced pointwise loss 应是

$$
\ell_{\rm pt}(n)=\frac1{a_n}\sum_{i\in A_n}
\ell_{\rm BCE}(r_{ni},z_{ni}),
$$

$$
L_{\rm pt}=\frac1{|\mathcal N|}\sum_{n\in\mathcal N}\ell_{\rm pt}(n).
$$

它提供绝对锚定：无论不同题目的 context 如何变化，`+1` 被推向正 logit、`-1` 被推向负 logit。因此它支持 calibration 和一个跨 context 的 first-error threshold。

---

## 5. Pairwise：同一 prefix 内的 Bradley--Terry loss

对同 node 的正、负候选，令

$$
r^+=r_\theta(c,s^+),\quad r^-=r_\theta(c,s^-),\quad
\Delta=r^+-r^-.
$$

Bradley--Terry logistic choice model 定义

$$
\Pr_\theta(s^+\succ s^-\mid c)=\sigma(\Delta).
$$

因此 pair loss 是

$$
\ell_{\rm BT}(r^+,r^-)
=-\log\sigma(r^+-r^-)
=\operatorname{softplus}[-(r^+-r^-)].
$$

梯度为

$$
\frac{\partial\ell_{\rm BT}}{\partial r^+}=\sigma(\Delta)-1,
\qquad
\frac{\partial\ell_{\rm BT}}{\partial r^-}=1-\sigma(\Delta).
$$

当双方同分（$\Delta=0$）时，梯度是 $-1/2,+1/2$，把正项推高、负项拉低；若已有很大正 margin，梯度趋于零。

### 它与 pointwise probability 的关系

令 $p^+=\sigma(r^+),p^-=\sigma(r^-)$，则

$$
\sigma(r^+-r^-)
=\frac{p^+(1-p^-)}{p^+(1-p^-)+(1-p^+)p^-}.
$$

所以 pairwise 是“两个互斥的正确/错误排列中，正 candidate 胜出的概率”，并非仅把两个分类输出机械相减。

一个 node 仅有一正一负时：

$$
\ell_{\rm pt}(n)=\frac12\left[
\operatorname{softplus}(-r^+)+\operatorname{softplus}(r^-)
\right],
$$

$$
\ell_{\rm pair}(n)=\operatorname{softplus}(r^- - r^+).
$$

两者在 $r^+=r^-=0$ 都是 $\log2$，且有相同的初始推拉方向，但不是相同函数。pointwise 关心绝对位置；pairwise 只关心相对差值。

### Pairwise-only 的平移不变性

对任意 context function $b(c)$，令

$$
r'(c,s)=r(c,s)+b(c).
$$

则

$$
r'(c,s^+)-r'(c,s^-)=r(c,s^+)-r(c,s^-).
$$

因此 BT loss 完全不变。pairwise-only 可以在每一道题整体上移或下移，仍保持全部 node 内排序；它无法仅靠自身确定一个统一的 “score 低于阈值就是错误” 标尺。这是 hybrid 仍需要 pointwise anchoring 的核心数学理由。

node-balanced pairwise risk 为

$$
\ell_{\rm pair}(n)=\frac1{m_n}
\sum_{i\in P_n}\sum_{j\in R_n}
\operatorname{softplus}[-(r_{ni}-r_{nj})],
$$

$$
L_{\rm pair}=\frac1{|\mathcal N|}\sum_{n\in\mathcal N}
\ell_{\rm pair}(n).
$$

先除 $m_n$ 使所有 node 等权。虽然单个 pair term 是 BT negative log-likelihood，但 node 内 Cartesian pairs 共享 candidate ratings，不能宣称它们条件独立。这里应称 node-balanced pairwise surrogate / composite risk。

---

## 6. Hybrid：为什么可以相加，lambda 如何选择

Hybrid 不是 pointwise 模型和 pairwise 模型的输出相加，而是同一个 $r_\theta$ 的两个 node-level risk 相加：

$$
L_\lambda(\theta)=L_{\rm pt}(\theta)+\lambda L_{\rm pair}(\theta),
\qquad \lambda\ge0.
$$

展开：

$$
L_\lambda=\frac1{|\mathcal N|}\sum_{n\in\mathcal N}
\left[
\frac1{a_n}\sum_{i\in A_n}\ell_{\rm BCE}(r_{ni},z_{ni})
+\lambda\frac1{m_n}\sum_{i\in P_n,j\in R_n}\ell_{\rm BT}(r_{ni},r_{nj})
\right].
$$

它们可相加的原因：

1. 同一个 input distribution（same nodes）与同一个 score function；
2. 都在 node 内、node 间用相同方式平均；
3. 都是 logistic negative-log-probability/surrogate，单位是 nats per node；
4. 监督不矛盾：BCE 要 `+1` 高、`-1` 低，BT 要同 node 的 `+1` 高于 `-1`。

这是一种标准的 composite empirical risk / multi-objective scalarization，**不是**独立观测的联合 likelihood。因为 pair labels 由同样的 absolute ratings 派生，不能假设 BCE 与 BT 两项独立、把它们相乘后声称新的独立数据证据。

### lambda 的量纲和梯度意义

若所有 score 初始为 0，任意 V0 candidate 的 BCE 与任意 pair 的 BT 都是

$$
\log2\approx0.693.
$$

且二者均先 node-average，所以训练初期

$$
L_{\rm pt}\approx L_{\rm pair}\approx\log2.
$$

这说明 lambda 不是将 accuracy 与 probability 等不同量纲相加，而是两个同量级 node risks 的相对权重。参数梯度为

$$
\nabla_\theta L_\lambda
=\nabla_\theta L_{\rm pt}
+\lambda\nabla_\theta L_{\rm pair}.
$$

lambda 越大，越强调 local ranking；这可能改善 pair accuracy，也可能损害绝对 calibration。该 trade-off 是项目要测量的对象，不能在理论上预先宣称总是双赢。

建议网格：

$$
\lambda\in\{0,0.1,0.3,0.5,1.0\}.
$$

其中 lambda=0 是 pointwise；pure pairwise 应单独仅训练 $L_{\rm pair}$，不要把 lambda 取无穷大。所有 lambda 使用同一 node sample list、initialization、updates、optimizer 和 LoRA config。

若 first-error 是主目标，可在每个 seed 的 validation 上按以下规则选择：

```text
最大化 first-error within-1；并列时取 Brier 更小；再并列取更小 lambda。
选择后固定，只在 test 评估一次。
```

不能看 test 后再选最佳 lambda。

---

## 7. DPO：正确的参考关系

DPO 也从 Bradley--Terry preference 形式出发，但训练的是 generator policy $\pi_\theta$。典型 DPO loss 是

$$
L_{\rm DPO}=-\mathbb E\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}
\right]\right).
$$

它用 generator/reference log-ratio 替换显式 reward difference，直接改变生成分布。我们的 PRM pairwise loss 是

$$
L_{\rm pair}=-\mathbb E\log\sigma[
r_\theta(c,s^+)-r_\theta(c,s^-)
].
$$

| | 本项目 PRM | DPO |
|---|---|---|
| 更新对象 | verifier LoRA + scalar head | generator policy |
| logistic 输入 | scalar score difference | policy/reference log-ratio difference |
| reference policy | 不需要 | 需要 |
| 用途 | ranking、calibration、first-error | 改变生成结果 |

因此可写：

> We use a Bradley--Terry pairwise likelihood, which also appears as the reward-model starting point in DPO. Unlike DPO, we train an explicit scalar process verifier rather than directly optimizing a reference-regularized generator policy.

不可写“we apply DPO to train the PRM”。[DPO](https://arxiv.org/abs/2305.18290)

---

## 8. 公平性：数据、计算与统计

### 数据公平

1. 先按 `problem_id` 切 train/validation/test，再 materialize nodes。
2. 对每个 label budget $b\in\{10\%,30\%,50\%,100\%\}$，以固定 seed 选出同一 $\mathcal N_b\subset\mathcal N_{\rm train}$ 给全部方法。
3. 报告人工标注成本：

$$
|\mathcal N_b|\quad\text{and}\quad\sum_{n\in\mathcal N_b}a_n,
$$

而不是 $\sum_n m_n$ 个派生 pairs。

### 计算公平：candidate forward 一次

一个 node batch 内，先 flatten 所有 candidates，Qwen 对每一个 candidate 只 forward 一次，得到 score tensor；之后按 node offsets 在 score 上计算 BCE 和所有 BT differences。

若 batch 为 $\mathcal B$，则

$$
\widehat L_{\rm pt}(\mathcal B)=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}\ell_{\rm pt}(n),
$$

$$
\widehat L_{\rm pair}(\mathcal B)=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}\ell_{\rm pair}(n).
$$

所有模式的 candidate-forward 数相同：

$$
\sum_{n\in\mathcal B}a_n.
$$

不要在 data loader 提前展开 Cartesian pairs；那会反复 forward 同一个 candidate，使 candidate-rich node 获得更多算力和梯度。

若某个 node 的 $m_n$ 极大，可均匀抽 $K$ 对：

$$
\widehat\ell_{\rm pair}(n)=\frac1K\sum_{k=1}^{K}\ell_{\rm BT}(i_k,j_k),
\qquad(i_k,j_k)\sim\operatorname{Uniform}(P_n\times R_n).
$$

则 $\mathbb E[\widehat\ell_{\rm pair}(n)]=\ell_{\rm pair}(n)$。必须记录 $K$ 与 sampling seed。

### 共同 test protocol

| 能力 | 测试单位 | 主指标 |
|---|---|---|
| absolute validity | 同一 $\mathcal N_{\rm test}$ candidates | AUROC、macro-F1、Brier、ECE |
| local ranking | 同一 $\mathcal N_{\rm test}$ nodes | node-macro pair accuracy、margin |
| process localization | held-out selected trajectories | exact、within-1、MAE、detection rate |

ranking 主指标为

$$
\operatorname{Acc}_{\rm node}=
\frac1{|\mathcal N_{\rm test}|}\sum_n\frac1{m_n}
\sum_{i\in P_n,j\in R_n}\mathbb 1[r_{ni}>r_{nj}].
$$

至少三个 seeds；置信区间以 `problem_id` cluster bootstrap，而不是以相关 pair 为独立样本。threshold、temperature、rank 与 lambda 只可用 validation 选择。

---

## 9. 对应代码实现

新增 `nodes_v0/{train,val,test}.jsonl`，每行一个 node，包含 `context_id`、problem、prefix、全部 V0 candidates 和 labels。现有 flat pointwise/pair files 仅保留作 audit，不作为严格主训练输入。

`NodeCollator` 对 node batch flatten candidates，返回 tokens、labels、`node_offsets`、context ids 和 truncation telemetry。它必须调用上文同一个 `pack_recent_prefix`。

loss API：

```text
nodewise_pointwise_loss(scores, labels, node_offsets)
nodewise_pairwise_loss(scores, labels, node_offsets, max_pairs_per_node=None)
hybrid_loss(pointwise_loss, pairwise_loss, lambda_pair)
```

新增 Qwen reward wrapper：`AutoModel(Qwen base + LoRA) -> final non-pad hidden -> Linear(1536,1)`。训练时关闭 cache、开启 checkpointing。

推荐顺序：1k node smoke（shape、loss、VRAM）→ 10k pilot（tokens/s、截断率）→ 2048 全矩阵（3 seeds）→ 只对代表性设置跑 4096 → 最后才做 neutral V1 / 7B QLoRA。

## References

- [Lightman et al., *Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050)
- [Christiano et al., *Deep RL from Human Preferences*](https://arxiv.org/abs/1706.03741)
- [Rafailov et al., *Direct Preference Optimization*](https://arxiv.org/abs/2305.18290)
- [Hu et al., *LoRA*](https://arxiv.org/abs/2106.09685)
- [Dettmers et al., *QLoRA*](https://arxiv.org/abs/2305.14314)
- [Qwen2.5-Math-1.5B model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
