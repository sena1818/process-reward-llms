# GPO、DPO、IPO 与本项目 scalar PRM 的关系：一手资料核查

> 核查日期：2026-07-17。本文只引用论文作者的 arXiv、会议 proceedings 或 OpenReview 原文。目标是避免把「训练 reward model」和「用 preference 直接训练 generator/policy」混为同一个方法。

## 结论先行

**本项目的 `pointwise BCE + Bradley--Terry (BT) pairwise` 是训练一个独立的、输出标量分数的 process reward model (PRM)；它不是 DPO、IPO、GRPO，也不应称为 GPO。**

- 可把 [Tang et al., 2024 的 *Generalized Preference Optimization*（GPO）](https://proceedings.mlr.press/v235/tang24b.html)作为**概念上的 related work**：它说明许多 *policy* 的 pairwise preference objectives 可由不同凸函数统一看待，并包含 DPO、IPO、SLiC。
- 但它不能为本项目的 hybrid loss 提供“这是 DPO/GPO 的一个特例”的证明。GPO 的可学习对象是语言模型策略 $\pi_\theta$，并显式使用参考策略 $\pi_{\rm ref}$ 的 log-ratio；本项目可学习对象是 scalar score $r_\theta(c,s)$，没有 $\pi_{\rm ref}$，测试时用于 verifier/reranking/first-error detection。
- 对本项目最贴切的基础引用应是：[Lightman et al., 2023/ICLR 2024](https://arxiv.org/abs/2305.20050)（PRM800K 与 process supervision）和 [Christiano et al., 2017](https://arxiv.org/abs/1706.03741) 或 [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) 中写出的 Bradley--Terry reward-model likelihood。若只需要一篇关于 scalar RM 的 BT 公式，DPO 论文的第 3--4 节很方便，但引用它时必须说明该处是在**从 reward model 推导到 policy loss**。

## 1. “GPO”不是唯一术语，正文第一次出现必须写全称与作者年份

`GPO` 至少指下列三个彼此不同的工作；不加限定地写“参考 GPO”会产生歧义。

| 缩写全称 | 原始论文与所学对象 | 与本项目的关系 |
|---|---|---|
| **Generalized Preference Optimization** | [Tang et al., ICML 2024](https://proceedings.mlr.press/v235/tang24b.html)：先把 scalar RM 的 pairwise score loss 写成一般分类 surrogate，再将 score difference 替换为 reference-relative policy log-ratio，得到直接优化生成策略 $\pi_\theta$ 的 GPO；DPO、IPO、SLiC 是该 policy family 的特例。 | 可作为 scalar PRM 的 pairwise surrogate 的理论背景；但**正式的 GPO objective 不是**本项目的 scalar PRM loss。本文若提 GPO，默认且只指这一篇。 |
| **Group Preference Optimization** | [Zhao, Dang & Grover, ICLR 2024](https://arxiv.org/abs/2310.11523)：用 meta-learned transformer 预测不同人群的偏好，再用于 group-specific alignment。 | 研究人群偏好个性化，数据单位和目标皆不同；不引用为本项目方法依据。 |
| **General Preference Optimization**（preference-score based） | [Zhang et al., 2024](https://arxiv.org/abs/2410.02197)：先学能表示一般/可能非传递偏好的 representation，再以其 score 做 policy optimization。 | 作者明确将其与 Bradley--Terry scalar RM 的传递性限制区分。若本项目坚持单一 scalar $r_\theta$ 与 BT，不能称为它的实现；只有改成非标量 comparator 才相关。 |

另有名称相近但更不应混淆的 [**GRPO**（Group Relative *Policy* Optimization）](https://arxiv.org/abs/2402.03300)：它是 DeepSeekMath 提出的 PPO 变体，而不是上述任一个 GPO。

## 2. 先固定本项目究竟训练什么

令 $n$ 为一个人工标注 node，$c_n=(q_n,\mathrm{prefix}_n)$ 是题目和完全相同的既有推理前缀，候选 step 为 $s_{ni}$，二值标签为 $z_{ni}\in\{0,1\}$（V0 的 `$-1/+1$`）。PRM 只有一个实值 head：

$$
r_\theta(c_n,s_{ni})\in\mathbb R,\qquad
p_\theta(z_{ni}=1\mid c_n,s_{ni})=\sigma(r_\theta(c_n,s_{ni})).
$$

对同一 node 的正、负候选集合 $P_n,R_n$，两个组成部分为

$$
\ell_{\rm pt}(n)=-{1\over |P_n|+|R_n|}\sum_i
\left[z_{ni}\log\sigma(r_{ni})+(1-z_{ni})\log(1-\sigma(r_{ni}))\right],
$$

$$
\ell_{\rm pair}(n)=-{1\over |P_n||R_n|}
\sum_{i\in P_n}\sum_{j\in R_n}\log\sigma(r_{ni}-r_{nj}).
$$

第二式就是 BT / RankNet 型的成对偏好负对数似然：$\Pr(s_{ni}\succ s_{nj}\mid c_n)=\sigma(r_{ni}-r_{nj})$。这与 DPO 论文重述的 BT 关系一致；该论文也明确指出 BT 只依赖 reward difference，且 $r(c,s)\mapsto r(c,s)+b(c)$ 不改变 pair probability。[DPO 原文第 3--5 节](https://arxiv.org/abs/2305.18290)

主实验用同一 node cohort 的 node macro average：

$$
L_{\rm pt}={1\over |\mathcal N|}\sum_{n\in\mathcal N}\ell_{\rm pt}(n),\quad
L_{\rm pair}={1\over |\mathcal N|}\sum_{n\in\mathcal N}\ell_{\rm pair}(n),\quad
L_\lambda=L_{\rm pt}+\lambda L_{\rm pair}.
$$

这不是“pointwise model + pairwise model”两个模型相加，而是**一个共享的 $r_\theta$**接受两种监督。详细的 fairness、node weighting 与 $\lambda$ 推导应以项目的专门设计文档为准；这里仅讨论它与 GPO 类方法的文献边界。

## 3. 为什么 BT reward-model loss 与 DPO 看起来相同，却不是同一训练

两者的 common core 是 logistic link，但进入该 link 的量不同。

| | scalar PRM 的 BT 训练 | DPO 的训练 |
|---|---|---|
| 被更新的函数 | score head / verifier $r_\theta(c,s)$ | 自回归 generator policy $\pi_\theta(y\mid x)$ |
| logistic 的输入 | $r_\theta(c,s^+)-r_\theta(c,s^-)$ | $\beta[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}]$ |
| reference policy | 无 | 必需的 $\pi_{\rm ref}$ |
| 目的 | 估计可单独部署的 score；比较/阈值/first-error | 直接改变生成分布；免去单独 RM 后的 RL 阶段 |
| 当前项目能否当作本方法 | 是 | 否；除非研究问题改成“用 step preference 直接微调 Qwen generator” |

这不是纯粹的命名差异。DPO 从 KL-regularized RLHF 的最优策略出发，把一个 reward 的等价代表写成

$$
\hat r_\theta(x,y)=\beta\log{\pi_\theta(y\mid x)\over\pi_{\rm ref}(y\mid x)},
$$

再代回 BT likelihood，得到直接训练 policy 的 loss。[DPO 原文 Eq. (6)--(7)](https://arxiv.org/abs/2305.18290) 因而，DPO 的“implicit reward”不是本项目通常意义下、带独立 scalar head 的 PRM；也没有理由把其 $β$ 当成混合损失中的 $\lambda$。

反过来说，训练完本项目的 PRM 后，**可以**把它的 score 当外部 reward 来做 RL/PPO/GRPO 或作为数据筛选器；那是额外的 generator post-training 阶段，不是 PRM 的 BCE/BT 训练本身。

## 4. Tang et al. 的 Generalized Preference Optimization（GPO）到底统一了什么

Tang et al. 把离线 pairwise **policy optimization** 写成对 policy/reference-policy log-ratio difference 的凸 surrogate：

$$
\rho_\theta=
\log{\pi_\theta(y_w\mid x)\over\pi_{\rm ref}(y_w\mid x)}-
\log{\pi_\theta(y_l\mid x)\over\pi_{\rm ref}(y_l\mid x)},
\qquad
\min_\theta\;\mathbb E[f(\beta\rho_\theta)].
$$

论文以 DPO 的 logistic、IPO 的平方型 surrogate、SLiC 的 hinge 等为例，并讨论各自隐含的离线正则化；准确的缩放约定应直接遵循其 [ICML 原文](https://proceedings.mlr.press/v235/tang24b.html)。关键替换是

$$
\underbrace{r_\theta(x,y_w)-r_\theta(x,y_l)}_{\text{RM/BT 的 score difference}}
\quad\longrightarrow\quad
\underbrace{\beta\rho_\theta}_{\text{policy/reference log-ratio difference}}.
$$

因此可做、不可做的表述分别是：

- 可以说：本项目的 pairwise 分量也采用 BT logistic surrogate；GPO 提供了“改变 pairwise surrogate 可能改变优化/正则性质”的一般背景。
- 不可以说：`L_pt + λL_pair` 是 GPO/DPO/IPO 的一个特例，或 GPO 理论已经证明该 hybrid 的 calibration 优势。GPO 没有 pointwise binary-label likelihood，也没有同 node 的 annotation-balanced weighting，更没有为 score-head 输出定义跨 context 阈值。
- 若将来做 **generator** 消融，可在相同 step preference data 上比较 DPO/IPO/GPO 变体；那回答“哪种偏好优化更好地改变 Qwen 的生成”，与本项目“哪种监督更好地训练 verifier”是另一篇实验设计，不能和 PRM 主表混合。

## 5. IPO 与 GRPO：是否应放进当前比较表

### IPO：不应作为当前 PRM 的 loss baseline

[Gheshlaghi Azar et al., AISTATS 2024](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html) 提出的 IPO（Identity Preference Optimization）属于直接从 pairwise preference 学**policy**的 $\Psi$PO 家族；原文目的正是避开“先由 pairwise preference 拟合 pointwise reward、再优化 policy”的若干近似。它改变的是 reference-relative policy objective，不是让独立 score head 用平方损失代替 BT。

所以：

- 若论文问题是 PRM/reranker，主表不比较 `BCE vs BT vs hybrid vs IPO`；最后一项的对象不同，结论不成立。
- 若只是研究“BT logistic 是否为最适合 scalar PRM 的 pairwise surrogate”，可另行比较 score difference 上的 logistic / hinge / squared-margin，但应命名为 **RM surrogate ablation**，不要称作 IPO；它没有 IPO 的 policy/reference-policy 理论语义。

### GRPO：不应作为当前 PRM 的 loss baseline

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 将 GRPO 定义为 PPO 的变体，用同一 prompt 采样的一组生成结果计算相对 advantage，从而更新**生成策略**并节省 critic/value model 的内存。它需要 rollout、奖励与 policy-gradient loop；不是监督学习的 reward-score estimator。

因此 GRPO 与当前 PRM 只能有两种正确关系：

1. **不做 generator RL 时**：不进入任何 BCE/BT/hybrid 的表格或 related-work 方法对比；最多在背景中说明它是下游可能消费者。
2. **做额外下游实验时**：固定已经训练好的 PRM，把其 score 作为 GRPO reward 的一个输入，并把这整个过程标为“PRM-guided policy optimization”。这时比较的是不同 PRM 对 generator RL 的下游影响，仍不能称 GRPO 训练了 PRM。

## 6. 对 hybrid 与公平比较的、可被严格支持的说法

GPO/DPO 文献不能替代本项目自己的统计论证。对当前数据，最稳妥的解释是一个**共享参数的 composite empirical risk**：

$$
L_\lambda(\theta)=L_{\rm pt}(\theta)+\lambda L_{\rm pair}(\theta),\quad\lambda\ge0.
$$

两个量都是每个标注 node 的平均 negative log-likelihood，单位皆为 nats；若全体 score 初始为零，则 $L_{\rm pt}=L_{\rm pair}=\log2$。所以 $\lambda$ 是有量纲一致、可在 validation set 选择的相对梯度权重，而不是任意把 probability 与 ranking metric 相加。

但由同一 `$+1/-1$` rating 导出的 point label 与 pair label **并不条件独立**。除非额外建立联合标注生成过程，不能声称 $L_\lambda$ 是完整独立观测的联合 likelihood，更不能将一个 node 的 $|P_n||R_n|$ 个 Cartesian pairs 当作那么多独立人工偏好。应明确把它表述为受控的 multi-objective/composite loss，并执行以下限制：

1. 三种 objective 使用同一 problem-disjoint split、backbone、context truncation、underlying annotated node cohort、seed 与优化预算；
2. 先在 node 内平均候选或 pairs，再在 node 间平均，防止 candidate-rich nodes 支配梯度；
3. 只用 validation 选择 $\lambda$（及阈值、温度），test set 只报告一次；
4. 不比较不同目标的 raw training loss；而在同一 test node cohort 上报告 absolute classification/calibration、node-macro ranking 与 first-error 指标。

这正是 pointwise 与 pairwise **可以公平比较**的基础：控制的是人工标注 node 和训练条件，比较的是预先确定的共同测试任务；不是要求 BCE 的数值与 BT 的数值在训练过程中相等。

## 7. 建议放入论文的最短表述与引用组合

可写：

> We train a scalar process reward model, rather than directly optimizing the generator. Pointwise BCE anchors an absolute step-validity score, while a Bradley--Terry likelihood enforces local ordering among candidates sharing an exact reasoning prefix. We treat their node-balanced weighted sum as a composite objective and select its mixing coefficient on validation data. This is distinct from DPO/IPO/GPO, which directly optimize a reference-regularized language-model policy from preference pairs.

推荐主引用顺序：

1. [Lightman et al., *Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050)：PRM800K、process supervision 和项目数据语境。
2. [Christiano et al., *Deep RL from Human Preferences*](https://arxiv.org/abs/1706.03741)：从 pairwise human preferences 学 reward 的经典设置。
3. [Rafailov et al., *DPO*](https://arxiv.org/abs/2305.18290)：给出本项目 BT score difference 与 DPO reference-log-ratio 的精确桥梁，同时明确两者差异。
4. [Tang et al., *Generalized Preference Optimization*](https://proceedings.mlr.press/v235/tang24b.html) 与 [Gheshlaghi Azar et al., *IPO*](https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html)：仅在 related work 中说明 direct policy optimization 的广义/替代 family。
5. [Shao et al., *DeepSeekMath*](https://arxiv.org/abs/2402.03300)：仅在讨论下游 generator RL 时引用 GRPO。

## 核查范围与限制

- 本文的“不同”指**当前的学习对象和实验问题不同**，不是说这些方法不能串成两阶段系统：PRM 可以在后续作为 generator RL 的 reward。
- GPO 是高歧义缩写；正式稿应始终写出全称、作者、年份，尤其不要把 GPO 与 GRPO 写在同一缩写栏而不定义。
- 本文不主张 BCE+BT hybrid 是新的 GPO 变体；其科学价值应来自 node-balanced 公平协议、calibration--ranking trade-off 和实验证据，而非术语嫁接。
