# Loss Design for Pointwise + Pairwise Process Reward Modeling

### 统一数据对象、概率模型与 weighted composite likelihood 的严格推导

*配套文档：`idea.md`（项目总纲）*
*本文只解决 loss design：数据对象是什么、pair 如何构造、两种单项 loss 从哪里来，以及为什么可以写成 $L_{\rm pt}+\lambda L_{\rm pair}$。*

---

## 0. 要证明的最终形式

给定题目、相同推理前缀和一个候选下一步，模型输出 scalar score

$$
r_\theta(c,s)\in\mathbb R.
$$

本项目在同一批 exact-prefix annotated nodes 上定义

$$
\boxed{
L_\lambda(\theta)
=
L_{\rm pt}(\theta)
+
\lambda L_{\rm pair}(\theta),
\qquad \lambda\ge 0.
}
$$

这里：

- $L_{\rm pt}$ 是 absolute step-validity 的 Bernoulli negative log-likelihood；
- $L_{\rm pair}$ 是在“两个候选标签不同”的条件下，正确相对顺序的 conditional logistic / Bradley--Terry negative log-likelihood；
- 两项共享同一个 $r_\theta$、同一个 outer sampling unit（annotated node）和同一批候选；
- node 内分别平均 candidates 与 pairs，避免候选多或 Cartesian pairs 多的 node 获得更大权重；
- 两项之和是一个 **weighted composite negative log-likelihood**，不是把重复使用的标注错误地当作相互独立后得到的 full joint likelihood。

核心数学链条是：

$$
\text{binary point model}
\Longrightarrow
\text{conditional pair model}
\Longrightarrow
\text{weighted composite likelihood}
\Longrightarrow
L_{\rm pt}+\lambda L_{\rm pair}.
$$

---

## 1. 原始数据对象与 exact-prefix 配对

### 1.1 Annotated node

一个原始标注 node 记为

$$
n=(c_n,A_n),
\qquad
c_n=(q_n,p_n),
$$

其中 $q_n$ 是题目，$p_n$ 是已经选中的推理前缀；同一 node 内的所有候选拥有**完全相同**的 context $c_n$。候选集合为

$$
A_n=\{(s_{ni},y_{ni})\}_{i=1}^{a_n},
\qquad
y_{ni}\in\{-1,0,+1\}.
$$

PRM800K 原论文训练的是 positive / neutral / negative 三类 label-token probability。本文的 V0 为了让 pointwise 与 pairwise 共用一个 scalar score，先只保留 $+1/-1$：

$$
\mathcal I_n
=
\{i\in\{1,\ldots,a_n\}:y_{ni}\in\{-1,+1\}\},
\qquad
b_n=|\mathcal I_n|.
$$

对 $i\in\mathcal I_n$，定义

$$
z_{ni}=\mathbb 1[y_{ni}=+1]\in\{0,1\},
\qquad
y_{ni}=2z_{ni}-1.
$$

这里的二值化是本项目的 V0 建模选择，不是对 Lightman et al. 三分类训练目标的原样复述。

### 1.2 正负集合与 pair set

定义

$$
P_n=\{i\in\mathcal I_n:z_{ni}=1\},
\qquad
R_n=\{j\in\mathcal I_n:z_{nj}=0\}.
$$

只有同时包含正、负候选的 node 才能构造干净的 V0 pair：

$$
\mathcal N
=
\{n:|P_n|>0,\ |R_n|>0\}.
$$

对 $n\in\mathcal N$，exact-prefix pair set 是

$$
E_n=P_n\times R_n,
\qquad
m_n=|E_n|=|P_n||R_n|.
$$

每个 $(i,j)\in E_n$ 表示：

$$
\text{同一个 }c_n,\qquad
s_{ni}\text{ 的标签为 }+1,\qquad
s_{nj}\text{ 的标签为 }-1.
$$

因此 pair 只改变 candidate step，不改变题目或推理历史。这一点让相对比较具有明确含义；跨 context 的任意配对不满足这个控制条件。

### 1.3 Pair 不是新的独立标注

若一个 node 有 $2$ 个正项和 $8$ 个负项，则 $m_n=16$，但原始数据仍只有 $10$ 个 candidate ratings。16 个 pairs 是这 10 个绝对标签所蕴含的比较关系，不是 16 次独立的人类 forced-choice 标注。

这个事实不妨碍使用 pairwise loss，但决定了最后必须称为 **composite likelihood / composite risk**，不能称为全部独立 pair observations 的 full likelihood。

---

## 2. 共享 score 与 pointwise probability

模型对 candidate 输入 $x_{ni}=(c_n,s_{ni})$ 输出

$$
r_{ni}=r_\theta(c_n,s_{ni}).
$$

V0 将它定义为 absolute-validity log-odds：

$$
p_{ni}
:=
\Pr_\theta(Z_{ni}=1\mid c_n,s_{ni})
=
\sigma(r_{ni}),
$$

$$
\log\frac{p_{ni}}{1-p_{ni}}=r_{ni},
\qquad
\sigma(t)=\frac{1}{1+e^{-t}}.
$$

使用同一个 scalar score 有两个后果：

1. pointwise 分量可把 $\sigma(r_{ni})$ 解释为二值化后的绝对正类概率；
2. pairwise 分量只能依赖同一 context 内的 score difference $r_{ni}-r_{nj}$。

第二点不是额外指定的另一个 head；下面会从同一个 Bernoulli model 推出。

---

## 3. Pointwise loss 的推导

### 3.1 单个 candidate 的 observation model

对 $z\in\{0,1\}$，Bernoulli conditional probability 为

$$
\Pr_\theta(Z=z\mid c,s)
=
p_\theta(c,s)^z
[1-p_\theta(c,s)]^{1-z},
$$

其中

$$
p_\theta(c,s)=\sigma(r_\theta(c,s)).
$$

等价地，用 $y=2z-1\in\{-1,+1\}$ 可写成

$$
\Pr_\theta(Y=y\mid c,s)
=
\sigma\!\left(y\,r_\theta(c,s)\right).
$$

### 3.2 取负对数得到 logistic / BCE loss

单个观察的 negative log-likelihood 为

$$
\begin{aligned}
\ell_{\rm pt}(r,z)
&=
-\log \Pr_\theta(Z=z\mid c,s)\\
&=
-z\log\sigma(r)
-(1-z)\log[1-\sigma(r)]\\
&=
\operatorname{softplus}(r)-zr.
\end{aligned}
$$

用 $y\in\{-1,+1\}$ 表示时，

$$
\boxed{
\ell_{\rm pt}(r,y)
=
-\log\sigma(yr)
=
\operatorname{softplus}(-yr).
}
$$

这就是 `binary_cross_entropy_with_logits` 的数学形式。

### 3.3 Node-balanced pointwise objective

若直接对全部 candidates 扁平平均，candidate 多的 node 会获得更大权重。为使 outer training unit 与 pairwise 保持一致，定义层级抽样：

$$
N\sim\operatorname{Unif}(\mathcal N),
\qquad
I\mid N=n\sim\operatorname{Unif}(\mathcal I_n).
$$

于是 pointwise empirical risk 是

$$
\boxed{
L_{\rm pt}(\theta)
=
\frac1{|\mathcal N|}
\sum_{n\in\mathcal N}
\underbrace{
\frac1{b_n}
\sum_{i\in\mathcal I_n}
\ell_{\rm pt}(r_{ni},z_{ni})
}_{\ell_{\rm pt}(n)}.
}
$$

它表示“先等概率选择一个 node，再等概率选择其中一个二值 candidate”时的期望 NLL。严格 loss 对照中的 pointwise baseline、pairwise-only 与 hybrid 都使用同一个 $\mathcal N$。

---

## 4. Pairwise loss 从同一个 Bernoulli model 推出

### 4.1 Pairwise 数据对象：discordant pair

考虑同一 node 中两个 candidate $i$ 和 $j$。定义标签不同事件

$$
D_{nij}
=
\{Z_{ni}\ne Z_{nj}\}.
$$

再定义 orientation variable

$$
T_{nij}
=
\begin{cases}
1, & Z_{ni}=1,\ Z_{nj}=0,\\
0, & Z_{ni}=0,\ Z_{nj}=1.
\end{cases}
$$

训练实现会把 preferred candidate 放在前面，所以对 $(i,j)\in P_n\times R_n$，观察到的 orientation 恒为 $T_{nij}=1$。但其概率必须从模型计算，而不是被设成 1。

### 4.2 从 absolute probabilities 推出 relative probability

作如下 working assumption：

> 给定完整输入 $(c_n,s_{ni},s_{nj})$ 后，两个 candidate 的 binary label outcomes 条件独立。

于是

$$
\Pr_\theta(Z_{ni}=1,Z_{nj}=0\mid c_n,s_{ni},s_{nj})
=
p_{ni}(1-p_{nj}),
$$

$$
\Pr_\theta(Z_{ni}=0,Z_{nj}=1\mid c_n,s_{ni},s_{nj})
=
(1-p_{ni})p_{nj}.
$$

在已知两者标签不同的条件下，candidate $i$ 为正、candidate $j$ 为负的概率为

$$
\begin{aligned}
q_{nij}
&:=
\Pr_\theta(T_{nij}=1\mid D_{nij},c_n,s_{ni},s_{nj})\\
&=
\frac{p_{ni}(1-p_{nj})}
{p_{ni}(1-p_{nj})+(1-p_{ni})p_{nj}}.
\end{aligned}
$$

利用

$$
\frac{p_{ni}}{1-p_{ni}}=e^{r_{ni}},
\qquad
\frac{p_{nj}}{1-p_{nj}}=e^{r_{nj}},
$$

得到

$$
\begin{aligned}
q_{nij}
&=
\frac{e^{r_{ni}}}{e^{r_{ni}}+e^{r_{nj}}}\\
&=
\sigma(r_{ni}-r_{nj}).
\end{aligned}
$$

因此

$$
\boxed{
\Pr_\theta(T_{nij}=1\mid D_{nij},c_n,s_{ni},s_{nj})
=
\sigma(r_{ni}-r_{nj}).
}
$$

这就是 Bradley--Terry / RankNet logistic pair probability。这里它不是凭空加入的第二套概率模型，而是同一个 Bernoulli pointwise model 在 discordant-pair 条件下得到的 conditional probability。

若不愿作两个 labels 条件独立的 working assumption，也可以直接把上式作为标准 Bradley--Terry observation model；但此时“pointwise 与 pairwise 完全来自同一生成模型”不再是定理，而是共享 score 的建模选择。报告应采用其中一种解释并明确假设。V0 采用前一种统一推导。

### 4.3 取负对数得到 pairwise loss

对一个按 preferred/rejected 排序的 pair，观察到 $T_{nij}=1$，因此

$$
\begin{aligned}
\ell_{\rm pair}(r_{ni},r_{nj})
&=
-\log q_{nij}\\
&=
-\log\sigma(r_{ni}-r_{nj})\\
&=
\operatorname{softplus}(r_{nj}-r_{ni}).
\end{aligned}
$$

所以

$$
\boxed{
\ell_{\rm pair}(r^+,r^-)
=
\operatorname{softplus}(r^- - r^+).
}
$$

### 4.4 Node-balanced pairwise objective

定义层级抽样：

$$
N\sim\operatorname{Unif}(\mathcal N),
\qquad
(I,J)\mid N=n
\sim
\operatorname{Unif}(P_n\times R_n).
$$

对应的 pairwise empirical risk 为

$$
\boxed{
L_{\rm pair}(\theta)
=
\frac1{|\mathcal N|}
\sum_{n\in\mathcal N}
\underbrace{
\frac1{m_n}
\sum_{i\in P_n}
\sum_{j\in R_n}
\ell_{\rm pair}(r_{ni},r_{nj})
}_{\ell_{\rm pair}(n)}.
}
$$

$1/m_n$ 的作用是让每个 node 的 pairwise 总权重相同。若 $m_n$ 太大，从 $P_n\times R_n$ 均匀抽 $K$ 对并取均值，可得到 $\ell_{\rm pair}(n)$ 的无偏 Monte Carlo estimator。

---

## 5. 为什么 $L_{\rm pt}$ 与 $L_{\rm pair}$ 可以相加

### 5.1 两项是同一数据的 marginal 与 conditional views

Pointwise component 使用

$$
\Pr_\theta(Z_{ni}=z_{ni}\mid c_n,s_{ni}),
$$

即单个 label 的 marginal conditional likelihood。

Pairwise component 使用

$$
\Pr_\theta(T_{nij}=1\mid D_{nij},c_n,s_{ni},s_{nj}),
$$

即同一批 labels 在“pair 中恰有一个正例”的条件下，哪一个 candidate 为正的 conditional likelihood。

因此两项：

- 针对不同但兼容的数据视角；
- 使用同一个参数 $\theta$ 和同一个 score $r_\theta$；
- 在相同 exact-prefix node 上计算；
- 不要求 candidate loss 与 pair loss 一一对应或数值相等。

它们可组合的统计框架是 weighted composite likelihood。

### 5.2 Weighted composite likelihood

经典 composite likelihood 定义允许把同一观测数据的 marginal 或 conditional likelihood components 以非负权重相乘：

$$
\operatorname{CL}(\theta)
=
\prod_k L_k(\theta)^{w_k},
\qquad
w_k\ge0.
$$

本项目对每个 node 定义 pointwise component

$$
C_{{\rm pt},n}(\theta)
=
\left[
\prod_{i\in\mathcal I_n}
\Pr_\theta(Z_{ni}=z_{ni}\mid c_n,s_{ni})
\right]^{1/b_n},
$$

以及 pairwise conditional component

$$
C_{{\rm pair},n}(\theta)
=
\left[
\prod_{i\in P_n}
\prod_{j\in R_n}
\Pr_\theta(T_{nij}=1\mid D_{nij},c_n,s_{ni},s_{nj})
\right]^{1/m_n}.
$$

然后定义 node-balanced weighted composite likelihood

$$
\boxed{
\operatorname{CL}_\lambda(\theta)
=
\prod_{n\in\mathcal N}
\left[
C_{{\rm pt},n}(\theta)
C_{{\rm pair},n}(\theta)^\lambda
\right]^{1/|\mathcal N|}.
}
$$

这里的 $1/b_n$、$1/m_n$ 和 $1/|\mathcal N|$ 都是 composite weights；$\lambda$ 是 pairwise component 相对于 pointwise component 的额外权重。

### 5.3 取负对数，严格得到 additive loss

对上式取负对数：

$$
\begin{aligned}
-\log\operatorname{CL}_\lambda(\theta)
&=
\frac1{|\mathcal N|}
\sum_{n\in\mathcal N}
\left[
-\log C_{{\rm pt},n}(\theta)
-\lambda\log C_{{\rm pair},n}(\theta)
\right]\\
&=
L_{\rm pt}(\theta)
+
\lambda L_{\rm pair}(\theta).
\end{aligned}
$$

因此

$$
\boxed{
\arg\max_\theta \operatorname{CL}_\lambda(\theta)
=
\arg\min_\theta
\left[
L_{\rm pt}(\theta)
+
\lambda L_{\rm pair}(\theta)
\right].
}
$$

这就是两项相加的严格来源：**weighted likelihood components 相乘，取负 log 后变成加权和。**

### 5.4 为什么它不是 full joint likelihood

Pairwise orientations 是由同一组 point labels 派生的；同一个 candidate 还会出现在多个 pairs 中。因此这些 components 彼此相关，不能写成

$$
\Pr(\text{all point labels and all pairs}\mid\theta)
=
\prod \Pr(\text{each component}\mid\theta)
$$

并声称右边是真实独立分解。

Composite likelihood 正是为“多个 marginal / conditional components 可重叠、但仍希望联合估计同一参数”的情形提供的构造。它是合法的 estimation objective，但不自动具有 full-likelihood 的全部效率与标准误性质。

所以报告中的正确表述是：

> We minimize the negative log of a node-balanced weighted composite likelihood that combines Bernoulli marginal components with discordant-pair conditional components.

而不是：

> The point labels and all derived pairs are independent observations under one joint likelihood.

---

## 6. 两项怎样做到可比较

### 6.1 相同 outer unit

两项都先在 node 内平均，再在同一个 $\mathcal N$ 上平均：

$$
L_{\rm pt}
=
\frac1{|\mathcal N|}
\sum_n \ell_{\rm pt}(n),
\qquad
L_{\rm pair}
=
\frac1{|\mathcal N|}
\sum_n \ell_{\rm pair}(n).
$$

因此一个 $1\times1$ node 与一个 $3\times8$ node 都只贡献一个 node-level point risk 和一个 node-level pair risk，不会因为 pair 数不同而隐式改变 $\lambda$。

### 6.2 相同 scorer 与相同方向

两项都优化同一个 $r_\theta$：

$$
\ell_{\rm pt}\text{ 要求 }r^+\uparrow,\ r^-\downarrow,
$$

$$
\ell_{\rm pair}\text{ 要求 }r^+-r^-\uparrow.
$$

二者的监督方向一致。pointwise 固定 absolute location，pairwise 强化 within-context separation。

### 6.3 相同数值基准，但梯度不必相同

若所有 scores 初始化为零，则

$$
\ell_{\rm pt}(0,z)=\log2,
\qquad
\ell_{\rm pair}(0,0)=\log2.
$$

经过 node 内平均后，

$$
L_{\rm pt}(\theta_0)
=
L_{\rm pair}(\theta_0)
=
\log2.
$$

所以两项都是 dimensionless log-loss，且在零分模型处具有相同基准。这使 $\lambda=1$ 成为透明的参考点。

但“初始 loss 相同”不等于“参数梯度范数相同”，也不意味着 $\lambda=1$ 理论最优。candidate composition、pair graph 和模型 Jacobian 都会影响

$$
\|\nabla_\theta L_{\rm pt}\|,
\qquad
\|\nabla_\theta L_{\rm pair}\|.
$$

因此 $\lambda$ 仍必须在 validation set 上选择。

### 6.4 与 convex scalarization 的关系

令

$$
\alpha=\frac{\lambda}{1+\lambda}\in[0,1).
$$

则

$$
L_{\rm pt}+\lambda L_{\rm pair}
=
(1+\lambda)
\left[
(1-\alpha)L_{\rm pt}
+
\alpha L_{\rm pair}
\right].
$$

如果只考虑 data-fit objective，乘以正数 $1+\lambda$ 不改变 minimizer。因此 $\lambda$ 等价于在两个 node-level risks 之间做 convex scalarization：

$$
\arg\min_\theta
[L_{\rm pt}+\lambda L_{\rm pair}]
=
\arg\min_\theta
[(1-\alpha)L_{\rm pt}+\alpha L_{\rm pair}].
$$

若训练中另加固定强度的 regularization，整体缩放会改变 data-fit 与 regularizer 的相对权重；实现时因此仍应固定原始 loss 形式、optimizer 和 weight decay，并把 $\lambda$ 作为唯一改变的 loss-mixing hyperparameter。

---

## 7. 两种 loss 各自提供什么信息

### 7.1 Pointwise：absolute anchoring

Pointwise 单项梯度为

$$
\frac{\partial \ell_{\rm pt}}{\partial r}
=
\sigma(r)-z.
$$

它直接把正例推向较高 absolute logit，把负例推向较低 absolute logit，从而为跨 context 的 probability calibration 和 threshold selection 提供锚定。

### 7.2 Pairwise：local separation

令 $\Delta=r^+-r^-$，则

$$
\frac{\partial \ell_{\rm pair}}{\partial r^+}
=
\sigma(\Delta)-1,
\qquad
\frac{\partial \ell_{\rm pair}}{\partial r^-}
=
1-\sigma(\Delta).
$$

它直接增大同一 context 内正负候选的 margin，并在 margin 已经很大时减小梯度。

### 7.3 Pairwise-only 的不可辨识性

对任意只依赖 context 的函数 $b(c)$，

$$
r'(c,s)=r(c,s)+b(c)
$$

满足

$$
r'(c,s_i)-r'(c,s_j)
=
r(c,s_i)-r(c,s_j).
$$

所以 $L_{\rm pair}$ 不识别 context-specific additive offset。pairwise-only 可以学习 within-context ranking，但 unified threshold 的 absolute location 不由 pairwise objective 决定；这正是 hybrid 保留 pointwise component 的数学理由。

### 7.4 Hybrid 没有创造新的独立标注信息

由于 $E_n=P_n\times R_n$ 完全由 point labels 构造，pairwise component 不会增加独立的人类标注量。在 Bernoulli model 完全正确、样本无限且优化充分的理想情形下，正确的 pointwise probabilities 已经决定

$$
\Pr_\theta(T_{nij}=1\mid D_{nij},\cdots)
=
\sigma(r_{ni}-r_{nj}),
$$

因此 pairwise component 不提供新的 population identification。

Hybrid 可能产生差异的原因是有限样本、模型错设、共享表示的容量限制和优化过程：它通过 composite weights 更强调 exact-prefix 的局部区分。项目可以检验这种 reweighting / inductive bias 是否改善 ranking、first-error localization 或低数据量表现，但不能预先声称它等价于获得了更多独立监督。

---

## 8. $\lambda$ 的含义与选择

$\lambda$ 不是 probability，也不是 Bradley--Terry temperature。它是 weighted composite likelihood 中 pairwise conditional components 的 power weight：

$$
\operatorname{CL}_\lambda
\propto
C_{\rm pt}\,C_{\rm pair}^{\lambda}.
$$

- $\lambda=0$：只使用 pointwise components；
- $0<\lambda<1$：pairwise conditional information 是辅助项；
- $\lambda=1$：node 内两类 averaged log components 等权；
- pure pairwise：单独最小化 $L_{\rm pair}$，用于 baseline；不要用一个极大的 $\lambda$ 模拟。

建议 validation grid：

$$
\lambda\in\{0,0.1,0.3,0.5,1.0\}.
$$

选择规则必须预先固定。若 first-error within-1 是主指标，可先最大化该指标；并列时选 Brier score 更小者；仍并列时选更小的 $\lambda$。test set 不参与选择。

如果希望研究 loss mixing 本身，可额外记录 validation 上

$$
\|\nabla_\theta L_{\rm pt}\|,
\qquad
\lambda\|\nabla_\theta L_{\rm pair}\|,
$$

作为解释性诊断；不要据 test result 反向调整 $\lambda$。

---

## 9. `0` 标签与原论文三分类 baseline

Lightman et al. 的原始 PRM 预测三类 label token，因此最接近原论文的 pointwise reproduction 是

$$
p_\theta(Y\in\{-1,0,+1\}\mid c,s),
\qquad
\ell_{\rm 3cls}
=
-\log p_\theta(Y=y\mid c,s).
$$

它使用三维 categorical output，不与 V0 的 scalar Bernoulli/BT head 直接共享参数化，应作为独立 reproduction baseline 报告。

若希望在 scalar hybrid 中加入 `0`，不能未经说明地把它当作 $0.5$ 正确率。更严格的 V1 是 cumulative-link ordinal model：

$$
\begin{aligned}
\Pr(Y=-1\mid c,s)
&=\sigma(\kappa_- - r),\\
\Pr(Y=0\mid c,s)
&=\sigma(\kappa_+ - r)-\sigma(\kappa_- - r),\\
\Pr(Y=+1\mid c,s)
&=1-\sigma(\kappa_+ - r),
\end{aligned}
$$

其中 $\kappa_-<\kappa_+$。只有在标注语义支持严格的 $+1>0>-1$ 时，才应进一步构造涉及 `0` 的 pairwise conditional components。

---

## 10. 实现必须与推导一致

### 10.1 当前实现状态检查

截至 2026-07-30，主体训练代码已经实现本文的 node-balanced
composite objective：

- `materialize_nodes_v0` 写出统一的 exact-prefix node cohort；
- pointwise、pairwise、hybrid 共用同一个 node DataLoader；
- 一个 batch 中每个 candidate 只 forward 一次；
- `nodewise_pointwise_loss` 和 `nodewise_pairwise_loss` 都先在 node
  内求 mean，再在 nodes 间求 mean；
- hybrid 的两项损失来自完全相同的一批 nodes。

旧的 flat pointwise/pair JSONL 仍被保留，供审计、兼容和可选的
secondary evaluation 使用，但不再进入严格主训练链路。代码与数据
一致性测试已通过；真实 Qwen-LoRA GPU smoke/pilot 仍需在 UniCluster
上运行后，才可把数值结果写入报告。

### 10.2 Target implementation

一个 batch 的单位应是 nodes：

1. 从同一个 $\mathcal N$ 采样 nodes；
2. 每个 candidate 只 forward 一次，得到 $r_{ni}$；
3. 在 node 内计算 candidate mean 和 pair mean；
4. 再对 nodes 取 mean；
5. 最后计算 $L_{\rm pt}+\lambda L_{\rm pair}$。

```python
def hybrid_loss(scores_by_node, labels_by_node, lam):
    point_terms, pair_terms = [], []

    for r, y in zip(scores_by_node, labels_by_node):
        # y contains only {-1, +1}; every batch node belongs to N.
        z = (y == 1).float()
        point_terms.append(
            F.binary_cross_entropy_with_logits(r, z, reduction="mean")
        )

        r_pos = r[y == 1]
        r_neg = r[y == -1]
        pair_diff = r_pos[:, None] - r_neg[None, :]
        pair_terms.append(F.softplus(-pair_diff).mean())

    loss_pt = torch.stack(point_terms).mean()
    loss_pair = torch.stack(pair_terms).mean()
    return loss_pt + lam * loss_pair
```

不可先把 $E_n$ 展成扁平 pair dataset 后直接平均，因为那会：

- 重复 forward 同一个 candidate；
- 使 $m_n$ 大的 node 获得更大梯度权重；
- 破坏 §5 中定义的 node-balanced composite likelihood。

严格对照还要求固定相同的 problem-disjoint split、node list、context packing、initialization、optimizer、update 数和 regularization。

---

## 11. 可直接放入报告的方法表述

> For each exact-prefix annotated node, we model the retained binary step label with a Bernoulli distribution whose log-odds is the scalar reward score. For a discordant candidate pair, conditioning this Bernoulli model on exactly one candidate being positive yields the Bradley--Terry probability $\sigma(r_i-r_j)$ for the observed orientation. We average candidate and pair negative log-likelihoods within each node and then average across the same node cohort. Their weighted sum is therefore the negative log of a node-balanced weighted composite likelihood, rather than the log-likelihood of independent point and pair annotations:
>
> $$
> L_\lambda
> =
> L_{\rm pt}
> +
> \lambda L_{\rm pair}.
> $$
>
> Pointwise components identify absolute step-validity logits, whereas pairwise conditional components sharpen within-context separation. The composite weight $\lambda$ is selected on validation data.

---

## 12. 参考来源

- [Varin, Reid & Firth (2011), *An Overview of Composite Likelihood Methods*](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf)：weighted product of marginal / conditional likelihood components 的正式定义，见 Eq. (2.1)。
- [Lightman et al., *Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050)：PRM800K、process supervision 与原始三类 label-token likelihood。
- [Bradley & Terry (1952), *Rank Analysis of Incomplete Block Designs*](https://doi.org/10.1093/biomet/39.3-4.324)：paired-comparison probability model。
- [Burges et al. (2005), *Learning to Rank Using Gradient Descent*](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)：RankNet 的 pairwise logistic probability 与 cross-entropy。
- [Christiano et al. (2017), *Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741)：由人类 pairwise comparisons 学 reward 的经典设置。
- [Rafailov et al., *Direct Preference Optimization*](https://arxiv.org/abs/2305.18290)：显式 reward-model 的 Bradley--Terry likelihood 与 reward difference 的不可辨识性。

论文原文核查笔记见 `notes/2026-07-25_loss_derivation_primary_sources_zh.md`。
