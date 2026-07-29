# Pointwise / Pairwise PRM loss：一手资料核查与本项目的严格写法

> 范围：本笔记只核查「监督信号 \(\to\) 概率观测模型 \(\to\) likelihood \(\to\) loss」这一条推导链，并给出可以直接放进项目报告的表述。结论依据原论文或作者/机构的官方页面；它不声称原论文已经做过本项目的 pairwise / hybrid 实验。

## 先给结论：应当怎样写

1. **先分清原始标注和本项目构造出的监督。** Lightman et al. 的 PRM800K 原始信号是每个步骤的三类标签 \(\{+,0,-\}\)，论文以「预测标签 token 的对数似然」训练 PRM；附录更明确写成预测 positive / negative / neutral 的三类概率。因此，论文原样对应的是 **categorical cross-entropy**，不是 binary BCE。
2. 本项目 V0 丢弃 \(0\)，令 \(z=\mathbb 1[y=+]\)，再假设 \(z\mid(c,s)\sim\operatorname{Bernoulli}(\sigma(r_\theta(c,s)))\)。**BCE 是这个二值化后的建模选择自然导出的 NLL**；不要写成「Lightman et al. 使用 BCE」。
3. Bradley--Terry (BT) 的 likelihood 对应一条**实际给出的二元比较** \(s_w\succ s_l\)。由 BT choice probability 得到 \(-\log\sigma(r_w-r_l)\)。这正是 DPO reward-modelling 部分和 RankNet 的单对损失。
4. 本项目的 \(P_n\times R_n\) pair 并不是逐对询问人类所得，而是从同一 node 的绝对标签**确定性地派生**。它们共享 candidate 与原始标签，故不能将全部 pair 项的乘积称为「PRM800K 原始标注的独立联合 likelihood」。正确名称是 **node-balanced pairwise composite risk / surrogate**；每个 summand 的形式是 BT negative log-likelihood。
5. pointwise + pairwise 同时用同一批绝对标签时，hybrid 也不是一个新的「联合 NLL」。它是有明确统计目标的**加权 composite objective**；\(\lambda\) 是待验证/调参的权重，不是从 BT 或 Bernoulli 自动推出来的概率参数。

上述区分会让正文的推导更像 MLE 课程中标准的写法：先声明观测变量与条件分布，再把 likelihood 取负对数；最后才讨论重加权、派生 pair 和 hybrid 这些实验设计选择。

## 一手来源与能支持什么

| 来源 | 可直接核对的内容 | 对本项目的用途 |
|---|---|---|
| Varin, Reid & Firth, [*An Overview of Composite Likelihood Methods*](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf), Eq. (2.1) | composite likelihood 是 marginal / conditional likelihood components 的非负加权乘积 | 严格说明 pointwise marginal component 与 derived-pair conditional component 为什么可在 log space 中加权相加 |
| Lightman et al., [*Let's Verify Step by Step*](https://arxiv.org/pdf/2305.20050), §2.6, Appendix B/D/F | 人工给每步 positive / negative / neutral；PRM 最大化每步末尾标签 token 的 log-likelihood；Appendix F.1 写为预测三类标签的概率；phase 1 才收集同一步的多个 alternative completions（约 5%） | 原始 pointwise 观测究竟是什么；为什么二值化与派生偏好必须写成项目选择 |
| [PRM800K 官方仓库](https://github.com/openai/prm800k) | 原始记录、标签和标注说明 | 数据格式的官方核对入口 |
| Bradley & Terry, [*Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*](https://doi.org/10.1093/biomet/39.3-4.324), 1952 | 成对比较模型的原始文献 | BT 的历史原始引用；全文访问可能受期刊权限限制 |
| Christiano et al., [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/pdf/1706.03741), §2.2.2--2.2.3 | 人工比较 trajectory segments；以软最大 choice probability 和 human labels 的 cross-entropy 拟合 reward | 「真实比较事件 \(\to\) BT/softmax CE」的早期 RLHF 一手例子 |
| Burges et al., [*Learning to Rank Using Gradient Descent*](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf), Eq. (1)--(3) | RankNet 对一对样本设 logistic preference probability，并使用 pairwise cross-entropy | 与本项目 score-difference / logistic loss 完全同形的 pairwise ranking 一手推导 |
| Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/pdf/2305.18290), Eq. (1)--(2), §5.1 | BT probability、用已排序 preference pairs 做 reward MLE；reward 加任意 context-only 函数的不可辨识性 | 现代 LLM reward-model 的标准写法；pairwise 平移不变性 |

## 1. 从 PRM800K 的真实观测开始

令一个带标注的候选步骤为 \(x=(c,s)\)，其中 \(c\) 是题目和已有推理前缀，\(s\) 是待判断的下一步。Lightman et al. 让标注者给每个步骤赋予

$$
y\in\{+,0,-\}.
$$

其 Appendix D 的语义是：neutral 表示在语境中合适、合理、正确但只含容易验证的计算；positive 在 neutral 的基础上还推进了解题；其余为 negative。论文 §2.6 说明训练时对该步骤末尾的标签 token 最大化 log-likelihood，Appendix F.1 则明确说明预测三类标签的概率。论文还说明只有 phase 1 收集了每一步的多个 alternative completions，约占 PRM800K 的 5%；这不是一个由逐对偏好问卷构成的数据集。

因此，最贴近原论文的 observation model 是

$$
p_\theta(Y=y\mid x),\qquad y\in\{+,0,-\},
$$

并以条件 categorical likelihood

$$
p_\theta(\mathcal D_{\rm 3cls})
=\prod_{t=1}^{T}p_\theta(Y_t=y_t\mid x_t)
$$

得到

$$
\mathcal L_{\rm 3cls}
=-\sum_{t=1}^{T}\log p_\theta(Y_t=y_t\mid x_t).
$$

这是三类 cross-entropy。它是理解本项目 V0 的干净起点：本项目不是「复述原论文 loss」，而是在同一原始标注上比较两种新的二元/相对监督目标。

### 可直接用于正文的事实性表述

> PRM800K 的原始监督是步骤级的 positive, neutral, negative 三类标签；Lightman et al. 将 PRM 训练为预测该标签 token 的条件概率，并最大化其对数似然。为使 pointwise、pairwise 和 hybrid 使用同一个 scalar score，本项目 V0 仅保留 \(+\) 与 \(-\) 标签并重新定义二值观测模型；这不是对原论文训练目标的逐字复现。[Lightman et al.](https://arxiv.org/pdf/2305.20050)

## 2. Pointwise：二值化后的 Bernoulli likelihood

### 2.1 明确写出本项目新增的二值化

V0 只对 \(y\in\{+,-\}\) 的候选保留观测，并定义

$$
z=\mathbb 1[y=+]\in\{0,1\},\qquad
r_\theta(x)\in\mathbb R,\qquad
p_\theta(x)=\sigma(r_\theta(x)).
$$

这里的核心**建模假设**是

$$
Z\mid x\sim\operatorname{Bernoulli}(p_\theta(x)).
$$

单个已观察标签的条件概率是

$$
p_\theta(z\mid x)
=p_\theta(x)^z[1-p_\theta(x)]^{1-z}.
$$

若把记录的二值标注事件视作在给定输入后的条件独立样本，则工作 likelihood 为

$$
p_\theta(\mathcal D_{\rm pt})
=\prod_{(x_i,z_i)\in\mathcal D_{\rm pt}}
p_\theta(x_i)^{z_i}[1-p_\theta(x_i)]^{1-z_i}.
$$

取负 log 并按记录数平均，得到 BCE：

$$
\ell_{\rm pt}(r,z)
=-\bigl[z\log\sigma(r)+(1-z)\log(1-\sigma(r))\bigr].
$$

由于 \(\log\sigma(r)=r-\operatorname{softplus}(r)\)，数值稳定的同一表达为

$$
\boxed{\ \ell_{\rm pt}(r,z)=\operatorname{softplus}(r)-zr\ }.
$$

这就是 `binary_cross_entropy_with_logits` 所计算的 quantity；应输入 \(r\) 而不是先手动 sigmoid。

### 2.2 为什么此处可以称 NLL，而必须写出条件

上述 BCE 是所声明的 Bernoulli 条件模型下的负对数似然。严格措辞应保留两个条件：

- **二值化假设：** \(0\) 标签被排除，故目标是 \(+\) 相对于 \(-\) 的概率，而不是原始三类分布；
- **工作独立性假设：** 写成乘积 likelihood 时，将记录的候选标签事件按 \(x\) 条件独立处理。若标注者、题目或同一轨迹带来额外相关性，这只是常见的条件/工作 likelihood，而不是对所有数据采集机制的完整生成模型。

在模型足够表达、训练/测试分布一致等理想条件下，Bernoulli log loss 的 population minimizer 是 \(\Pr(Z=1\mid x)\)。这解释了它为何给出绝对的「正类概率」解释；但有限样本神经网络的 calibrated probability 仍应在 validation set 上检查或校准，不能仅由 BCE 最小化自动保证。

## 3. Pairwise：一条真实偏好观察时的 Bradley--Terry likelihood

### 3.1 正确的观测单位

考虑同一 context 下由人工直接比较得到的一条事件

$$
d=(c,s_a,s_b,o),\qquad o\in\{0,1\},
$$

其中 \(o=1\) 表示标注者选择 \(s_a\succ s_b\)。令 \(r_a=r_\theta(c,s_a)\)、\(r_b=r_\theta(c,s_b)\)。BT 模型为

$$
q_\theta(a\succ b\mid c)
=\frac{\exp(r_a)}{\exp(r_a)+\exp(r_b)}
=\sigma(r_a-r_b).
$$

这正是 DPO 的 reward-modelling Eq. (1)。Christiano et al. 对 trajectory-segment 比较使用同一类 softmax choice model，并最小化预测分布与 human labels 的 cross-entropy；RankNet Eq. (1)--(3) 则直接给出 logistic pair posterior 和其 cross-entropy cost。

给定一条比较，Bernoulli likelihood 和 NLL 为

$$
p_\theta(o\mid c,s_a,s_b)
=q_\theta^o(1-q_\theta)^{1-o},
$$

$$
\ell_{\rm BT}(r_a,r_b,o)
=-o\log\sigma(r_a-r_b)
-(1-o)\log\sigma(r_b-r_a).
$$

将 pair 按偏好方向重排为 \((s_w,s_l)\)，即观测 \(o=1\)，就有

$$
\boxed{\ \ell_{\rm BT}(r_w,r_l)
=-\log\sigma(r_w-r_l)
=\operatorname{softplus}(r_l-r_w)\ }.
$$

若 \(\mathcal D_{\rm direct}\) 真的是收集到的独立/条件独立比较事件，便可写

$$
\mathcal L_{\rm BT}^{\rm MLE}
=-\frac1{|\mathcal D_{\rm direct}|}
\sum_{(c,s_w,s_l)\in\mathcal D_{\rm direct}}
\log\sigma\!\bigl(r_\theta(c,s_w)-r_\theta(c,s_l)\bigr).
$$

这正是 Rafailov et al. DPO 论文 Eq. (2) 在「显式 reward model」阶段给出的 reward MLE；本项目只借用这一步，**不把 verifier 训练称作 DPO**。

### 3.2 RankNet 的补充：pairwise CE 还可写成软目标

Burges et al. 写出一般的 pair target \(\bar P_{ij}\in[0,1]\)：

$$
C_{ij}
=-\bar P_{ij}\log P_{ij}
-(1-\bar P_{ij})\log(1-P_{ij}),
\qquad
P_{ij}=\sigma(r_i-r_j).
$$

正/负 pair 使用 \(\bar P_{ij}=1\)，即退化为上节的 \(-\log\sigma(r_i-r_j)\)。\(\bar P=1/2\) 在 RankNet 的数学形式中可代表 tie/软 target；但不要因此把 PRM800K 的 neutral 自动解释成「人类对任意两个候选选择概率正好为 \(1/2\)」。原数据给的是单步三类 rating，并非这种 pairwise choice observation。

## 4. 本项目的 Cartesian pairs：为何不是独立 joint likelihood

令一个原始 annotated node 为

$$
n=(c_n,A_n),\qquad
A_n=\{(s_{ni},z_{ni})\}_{i=1}^{a_n},
$$

并令

$$
P_n=\{i:z_{ni}=1\},\quad
R_n=\{j:z_{nj}=0\},\quad
m_n=|P_n||R_n|.
$$

本项目从同一组绝对标签派生出 \(P_n\times R_n\) 中所有 \(+\!>\!-\) 关系。若 \(|P_n|=2,|R_n|=8\)，会有 16 个训练项；然而原始数据只提供了 10 个 candidate 的绝对 ratings，并没有由标注者完成 16 次独立的 A-vs-B 选择。

因此，条件在原始标签 \(A_n\) 上时，16 个「winner」是确定的函数：它们共享每一个候选和它的标签。把它们写成

$$
\prod_{i\in P_n}\prod_{j\in R_n}
\sigma(r_{ni}-r_{nj})
$$

并称为「原始 PRM800K 标注的 joint likelihood」会额外且错误地把派生关系当作独立比较观测。标准 BT MLE 的条件是原始数据集本来就由比较事件组成；DPO 明确把数据定义为 \((x,y_w,y_l)\) 的 preference dataset，Christiano et al. 也实际向标注者提出两段轨迹的比较。这里不满足这一点。

### 本项目可用的、诚实的 pairwise objective

对 node 内 pair 项取均值、再对 node 取均值：

$$
\ell_{\rm pair}(n)
=\frac{1}{m_n}
\sum_{i\in P_n}\sum_{j\in R_n}
\operatorname{softplus}\!\bigl(r_{nj}-r_{ni}\bigr),
$$

$$
\boxed{\quad
L_{\rm pair}^{\rm node}
=\frac1{|\mathcal N|}\sum_{n\in\mathcal N}\ell_{\rm pair}(n).
\quad}
$$

最适合报告的命名是：

> a node-balanced pairwise composite risk (or surrogate), whose individual terms have the Bradley--Terry negative-log-likelihood form.

中文可写为：

> 我们由同一 prefix 下的绝对步骤标签构造正负候选对，并最小化 node-balanced 的 pairwise composite risk。每个 pair 项采用 Bradley--Terry 负对数似然的形式；由于这些 pair 由同一组原始标注派生、并不对应独立收集的比较事件，本文不将所有 Cartesian pairs 的乘积解释为原始数据的联合似然。

这不是说该 loss 不能使用，而是精确说明它优化的经验风险是什么。

## 5. 平均/标准化：它们定义了「每个谁等权」

### 5.1 普通 MLE 平均与 node-balanced 平均不是同一 estimand

V0 过滤 `0` 后，令 $\mathcal I_n$ 为保留 candidate 的 index set，$b_n=|\mathcal I_n|$。若每条保留 candidate label 都是同等抽样单位，未加权经验 NLL 是

$$
\frac1{\sum_n b_n}\sum_n\sum_{i\in\mathcal I_n}
\ell_{\rm pt}(r_{ni},z_{ni}).
$$

这给 candidate 多的 node 更大总权重。若实验问题希望每个 annotated node 同等重要，可改为

$$
\ell_{\rm pt}(n)
=\frac1{b_n}\sum_{i\in\mathcal I_n}\ell_{\rm pt}(r_{ni},z_{ni}),
\qquad
\boxed{\quad
L_{\rm pt}^{\rm node}
=\frac1{|\mathcal N|}\sum_{n\in\mathcal N}\ell_{\rm pt}(n).
\quad}
$$

这个式子是完全合理的**重加权 empirical risk**，但一般不是「原始 candidate-record sampling distribution 下的 MLE」。它等价于先均匀抽一个 node，再在该 node 内均匀抽一个 candidate 的期望：

$$
L_{\rm pt}^{\rm node}
=\mathbb E_{n\sim\operatorname{Unif}(\mathcal N)}
\mathbb E_{i\sim\operatorname{Unif}(\mathcal I_n)}
[\ell_{\rm pt}(r_{ni},z_{ni})].
$$

同理，\(L_{\rm pair}^{\rm node}\) 等价于「先均匀抽 node，再均匀抽该 node 的一个派生正负 pair」。这给出 node 内 \(1/m_n\) 的严格理由：它不是为了让公式好看，而是明确指定每个 node 的总训练权重为 1。

### 5.2 大 node 只抽 \(K\) 对时

若从 \(P_n\times R_n\) 均匀独立抽样 \(K\) 个 pair（有放回），可使用

$$
\widehat\ell_{\rm pair}(n)
=\frac1K\sum_{k=1}^K
\operatorname{softplus}(r_{n,j_k}-r_{n,i_k}),
\qquad
(i_k,j_k)\sim\operatorname{Unif}(P_n\times R_n).
$$

则

$$
\mathbb E[\widehat\ell_{\rm pair}(n)]=\ell_{\rm pair}(n).
$$

这是对已定义 composite risk 的无偏 Monte-Carlo estimator，并不把它变回独立的原始人类比较 likelihood。报告应记录 \(K\)、with/without replacement 和随机种子。

## 6. Pairwise score 的可辨识性，以及它真正支持的结论

BT probability 只依赖差值。令 \(b(c)\) 是任意只依赖 context 的函数，且

$$
r'_\theta(c,s)=r_\theta(c,s)+b(c).
$$

那么

$$
r'_{\theta}(c,s_a)-r'_{\theta}(c,s_b)
=r_{\theta}(c,s_a)-r_{\theta}(c,s_b),
$$

所以所有 BT pair probabilities 和 pairwise objective 都不变。DPO §5.1 的 Definition 1 / Lemma 1 正式陈述了这一点：相差一个只依赖 \(x\) 的函数的 reward 诱导相同的 Bradley--Terry / Plackett--Luce preference distribution。

可以在报告中写：

> Pairwise supervision identifies within-context preference differences, not a context-comparable absolute offset. Therefore a pairwise-only objective does not by itself identify a global threshold for the binary step-validity event.

须避免两种过强说法：

- 不要说「pairwise-only 模型绝对不可能在实践中得到可用阈值」；共享神经网络的参数化、正则化和数据分布可能选出某个偏移；
- 更严格、也足够强的说法是「该偏移**不由 pairwise likelihood 识别**，所以若任务需要可解释的 \(\Pr(z=1\mid c,s)\) 或全局阈值，需要额外的绝对监督/校准约束」。

Pointwise BCE 恰提供了这种绝对标签约束：它不是与 pairwise 相同的 loss，而是对一个不同、但互补的观测模型建模。

## 7. Hybrid 的正确概率语言

在相同 node 上，pointwise 项使用原始 \(z_{ni}\)，而 pairwise 项又是该 \(z_{ni}\) 派生出的比较。因此

$$
L_{\rm hyb}=L_{\rm pt}^{\rm node}+\lambda L_{\rm pair}^{\rm node},
\qquad \lambda\ge 0,
$$

应被表述为 weighted composite / multi-objective empirical risk，而不是

$$
-\log p_\theta(\text{raw pointwise labels and all derived pairs})
$$

的单一联合 likelihood。后者会把同一标注信息重复当作独立观测。

更严格地，令 $C_{{\rm pt},n}$ 是 node $n$ 内 pointwise likelihood contributions 的 geometric mean，$C_{{\rm pair},n}$ 是 derived discordant-pair conditional likelihood contributions 的 geometric mean。根据 Varin, Reid & Firth (2011) 的 weighted composite likelihood 定义，可写

$$
\operatorname{CL}_\lambda(\theta)
=
\prod_{n\in\mathcal N}
\left[
C_{{\rm pt},n}(\theta)
C_{{\rm pair},n}(\theta)^\lambda
\right]^{1/|\mathcal N|}.
$$

于是

$$
-\log\operatorname{CL}_\lambda(\theta)
=
L_{\rm pt}^{\rm node}(\theta)
+
\lambda L_{\rm pair}^{\rm node}(\theta).
$$

这给出了 additive hybrid loss 的直接数学来源：component likelihoods 在 probability space 中以权重相乘，取负对数后成为加权和。由于 components 重复使用同一 labels，它仍是 composite likelihood，而不是 full joint likelihood。

node-level normalisation 的价值是：\(L_{\rm pt}^{\rm node}\) 与 \(L_{\rm pair}^{\rm node}\) 都以「一个 node」为外层单位，故 \(\lambda\) 不会因某些 node 有大量 Cartesian pairs 而被隐式放大。它**不**使 \(\lambda\) 从理论上自动确定；\(\lambda\) 仍是「absolute validity 与 within-context ranking 两类风险的取舍」，应在 validation split 上选定，随后固定到 test evaluation。

适合放进论文的方法段落：

> We optimise a node-balanced hybrid composite objective. The pointwise term is the Bernoulli negative log-likelihood under our binarised-label model, whereas the pairwise term is a Bradley--Terry-form surrogate over comparisons derived from those labels. Since the two terms reuse the same annotations, \(\lambda\) is treated as a validation-selected multi-objective weight rather than as a likelihood-derived constant.

## 8. 建议放进项目正文的紧凑推导版本

以下版本刻意删去与主线无关的梯度、soft-label 变体和过早的实现细节，适合作为 report 的核心数学小节。

> 对每个标注 node \(n\)，设 \(c_n\) 是题目与已选前缀，\(A_n=\{(s_{ni},y_{ni})\}_{i=1}^{a_n}\) 是该 context 下的候选下一步及其 PRM800K 标签。原始数据为三类标签 \(y\in\{+,0,-\}\)；在 V0 中令 \(\mathcal I_n=\{i:y_{ni}\in\{+,-\}\}\)、\(b_n=|\mathcal I_n|\)，去除 \(0\) 并对 \(i\in\mathcal I_n\) 定义 \(z_{ni}=\mathbb 1[y_{ni}=+]\)。令 reward head 输出 \(r_{ni}=r_\theta(c_n,s_{ni})\)。
>
> **Pointwise model.** 我们假设 \(z_{ni}\mid(c_n,s_{ni})\sim\operatorname{Bernoulli}(\sigma(r_{ni}))\)。因此一个观察到的步骤标签的负对数似然为
> $$
> \ell_{\rm pt}(r_{ni},z_{ni})
> =\operatorname{softplus}(r_{ni})-z_{ni}r_{ni}.
> $$
> 为使每个 annotated node 等权，我们定义
> $$
> L_{\rm pt}=\frac1{|\mathcal N|}\sum_{n\in\mathcal N}
> \frac1{b_n}\sum_{i\in\mathcal I_n}\ell_{\rm pt}(r_{ni},z_{ni}).
> $$
>
> **Pairwise model.** 对同一 \(c_n\) 内满足 \(z_{ni}=1,z_{nj}=0\) 的候选，我们以 Bradley--Terry choice model 定义该排序的模型概率
> $$
> \Pr_\theta(s_{ni}\succ s_{nj}\mid c_n)
> =\frac{e^{r_{ni}}}{e^{r_{ni}}+e^{r_{nj}}}
> =\sigma(r_{ni}-r_{nj}).
> $$
> 它对应的单对负对数似然形式为
> $$
> \ell_{\rm BT}(i,j)=-\log\sigma(r_{ni}-r_{nj})
> =\operatorname{softplus}(r_{nj}-r_{ni}).
> $$
> 本数据中的 pairs 由绝对标签派生，故我们将其聚合定义为 node-balanced pairwise composite risk，而不把它称作独立 pair observations 的联合 likelihood：
> $$
> L_{\rm pair}=\frac1{|\mathcal N|}\sum_{n\in\mathcal N}
> \frac1{|P_n||R_n|}\sum_{i\in P_n}\sum_{j\in R_n}\ell_{\rm BT}(i,j).
> $$
>
> **Hybrid objective.** 最后使用
> $$
> L_{\rm hyb}=L_{\rm pt}+\lambda L_{\rm pair},
> $$
> 其中 \(\lambda\) 在 validation split 上选择。该目标将来自同一标注的绝对有效性与相对排序信号结合，因而是 weighted composite objective，不被解释为单一独立联合 likelihood。

## 9. 容易被老师追问的句子：推荐/不推荐写法

| 不推荐 | 推荐 |
|---|---|
| “PRM800K trains a binary BCE verifier.” | “Lightman et al. train a three-label PRM; our V0 converts the retained \(+/-\) labels into a Bernoulli pointwise model.” |
| “All Cartesian pairs form the BT likelihood.” | “Each derived pair contributes a BT-form NLL term; their node-balanced average is our pairwise composite risk.” |
| “The pairs are \(m_n\) independent human preferences.” | “They are \(m_n\) logical comparisons implied by \(a_n\) absolute labels, so they share observations.” |
| “Node average is the MLE.” | “Node average reweights the empirical risk so every annotated node has equal total weight; it is the MLE only under the corresponding node-uniform sampling target.” |
| “\(\lambda\) follows from probability theory.” | “\(\lambda\) is a validation-selected weight in a hybrid composite objective.” |
| “Pairwise cannot be calibrated.” | “BT differences do not identify a context-specific additive offset; any desired absolute probability/threshold needs pointwise supervision and empirical calibration.” |

## 参考文献（原始来源）

1. Cristiano Varin, Nancy Reid and David Firth (2011), [*An Overview of Composite Likelihood Methods*](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf), Eq. (2.1).
2. Hunter Lightman et al. (2023), [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050). 具体训练/标注细节见 PDF 的 §2.6、Appendix B、D、F；官方数据见 [openai/prm800k](https://github.com/openai/prm800k)。
3. Ralph A. Bradley and Milton E. Terry (1952), [*Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*](https://doi.org/10.1093/biomet/39.3-4.324), *Biometrika* 39(3/4), 324--345.
4. Paul F. Christiano et al. (2017), [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741), §2.2.
5. Christopher J. C. Burges et al. (2005), [*Learning to Rank Using Gradient Descent*](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf), Eq. (1)--(3).
6. Rafael Rafailov et al. (2023/2024), [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290), Eq. (1)--(2), §5.1.
