# PRM 项目答辩笔记（一）

用途：与老师讨论 proposal 前的技术答辩笔记。它区分原论文已经证明的内容、当前项目的假设，以及 V0 实际实现。

主论文：Lightman et al. (2023), *Let's Verify Step by Step*。

- 论文：https://arxiv.org/abs/2305.20050
- 数据集：https://github.com/openai/prm800k
- Canonical project plan：[idea/idea.md](../idea/idea.md)

## 1. 术语与整体角色

- **ORM**：Outcome-supervised Reward Model。输入完整 solution，预测最终解答是否正确。
- **PRM**：Process-supervised Reward Model。对 reasoning process 中的每一步打分。
- 本项目讨论的是 PRM，不要说 PLM。
- Generator 产生多个候选解答；verifier / reward model（ORM 或 PRM）对候选排序。
- 原论文的比较固定 generator，不用 reward model 的信号做 RL 更新 generator。

对同一题，固定 generator 产生 $N$ 条 candidate solutions：

~~~text
N 条 generator solutions
        ├─ ORM：每条一个 outcome score，选最高者
        └─ PRM：每步打分，聚合成一个 solution score，选最高者
                                      ↓
                 外部 final-answer grader 判被选 solution 是否正确
                                      ↓
                    跨题正确率 = best-of-N accuracy
~~~

PRM 与 ORM 的 raw scores 不直接比较；各自在同一题的 $N$ 条候选内排序。可比较的是两者选择结果的 best-of-N accuracy。

## 2. 原论文的大规模实验

### 2.1 两种模型的 target 与 score

| 模型 | 训练 target | 推理时的 solution score |
|---|---|---|
| ORM | 完整 solution 的 final answer 是否正确；实际主要由 automatic answer grading 给标签 | 最终 token 的 outcome score |
| PRM | 已标 reasoning step 的 positive / neutral / negative 标签 | step-level acceptable scores 的聚合 |

论文的最佳 PRM 聚合将 neutral 视为 acceptable，并取每步 acceptable probability 的乘积：

$$
S_{\mathrm{PRM}}(y)=\prod_{t=1}^{T}q_t.
$$

$q_t$ 是第 $t$ 步被视为 acceptable 的概率。它强烈惩罚一条链中的任何可疑 step，但论文承认它略偏向较短的 solution。

### 2.2 Figure 3 证明什么，不能证明什么

原论文 Section 3 明确说明：

- PRM 使用 PRM800K；phase 2 经 active learning，优先标注“当前 PRM 打分高、但 final answer 错”的 convincing wrong-answer solutions；
- ORM 以每题 100 条均匀采样的 solutions 训练；
- ORM training set 与 PRM800K 不重叠，且论文明确写明 ORM training set 大约大一个数量级。

所以可以说：

> In the paper's best large-scale pipelines, PRM outperforms ORM.

但不能只凭 Figure 3 就说：

> 在同一份数据上，只因 process label 更细，所以 PRM 必然更强。

前者是实际 pipeline 的性能结论；后者要求严格只改变一个变量的因果实验。

### 2.3 模型规模

large-scale 的 generator、ORM、PRM 都从 GPT-4 base model fine-tune 而来。small-scale 模型是设计相似、但预训练计算量约小 200 倍的模型。每一个 regime 内比较的 reward models 处于同一规模；large PRM 与 small PRM 不是直接作公平性能比较的对象。

## 3. small-scale synthetic supervision：受控实验

### 3.1 什么是 PRMlarge oracle

由于重新收集大量人工 step labels 很昂贵，作者用强大的 PRMlarge 充当 **labeling oracle**。

它不是测试时直接替代小模型的答案，而是一个临时监督器：

1. 输入一条 generator 已产生的 reasoning solution；
2. 对每一步输出 positive / neutral / negative 的概率；
3. 如果 negative probability 大于 20%，该步在 synthetic dataset 中被判为错误。

这样可以便宜地构造大量近似人工标注，以做 controlled ablation。

### 3.2 同一份 raw solutions，三种 supervision

假设固定 small-scale generator 对同一道题产生：

~~~text
A: step 1 ✓, step 2 ✓, step 3 ✓
B: step 1 ✓, step 2 ✗, step 3 ?
C: step 1 ✓, step 2 ✗, step 3 ?
~~~

PRMlarge 打过标签后，从**同一批 raw solutions**构造：

| 训练方式 | A | B | C |
|---|---|---|---|
| Process supervision | ✓, ✓, ✓ | ✓, ✗ 后停止 | ✓, ✗ 后停止 |
| PRMlarge-outcome supervision | correct | incorrect | incorrect |
| Final-answer outcome supervision | automatic final-answer grader 决定 | 同左 | 同左 |

对 PRMlarge-outcome 版本：一条 solution 当且仅当每一步都被 PRMlarge 判正确时，才标为 correct。也就是说，同一份 step information 被压缩成一个 outcome bit。

故 process 与 PRMlarge-outcome 的比较控制了：

- 相同问题、generator 与 sampled solutions；
- 相同强监督 oracle；
- 相同小模型规模与训练预算；
- 主要只改变：保留 step information，或将其压缩为 outcome。

论文还比较自动 final-answer outcome supervision，以观察 automatic outcome target 的影响。small-scale synthetic supervision 并不伪造数学题或新 solution；它伪造的是便宜且可大规模产生的监督标签。

注意：Figure 3 的 large-scale 实验有 majority voting baseline；Figure 4 的 small-scale controlled comparison 重点是不同 reward-model supervision，不是 majority voting。

## 4. PRM800K 数据究竟长什么样

### 4.1 四层结构

从大到小，数据不是“每题只有一条解答”，而是：

~~~text
PRM800K JSONL files
  └─ 一行 = 一个 solution sample / annotated reasoning tree
       ├─ question：problem text、ground-truth answer 等
       └─ label.steps：按时间顺序的 annotated step nodes
            └─ 每个 node 有 completions = 多个 candidate next steps
                 ├─ text
                 ├─ rating：+1 / 0 / -1
                 └─ chosen_completion：哪一个 candidate 被选入继续的 trajectory
~~~

同一个 math problem 可以有多个 solution samples；一个 sample 也可以在一个 node 内有多个 candidate completions。selected trajectory 是沿每个 node 的 chosen completion（或 phase-1 的 human completion）继续得到的一条实际路径。prefix 就是这条 selected trajectory 到当前 node 之前的文本。

因此，一个 node 天然已经满足：

~~~text
same problem + same selected previous steps + alternative next steps
~~~

它正是 exact-prefix pairing 所需的最小数据单位。

### 4.2 真实 V0 pair 的例子

当前 processed pair data 中有一道题：

~~~text
Problem: How many positive two-digit integers leave a remainder of 2 when divided by 8?

Prefix:
  form is 8n+2
  test n=1: 10 is two-digit
  test n=2: 18 is two-digit
  ...（前七步完全固定）

Positive candidate:
  Keep plugging numbers until we get a three-digit number.

Negative candidate:
  All numbers of the form 8n+2 are two-digit integers.
~~~

两条 candidate 的题目和全部 prefix 一字不变，只有当前 next step 改变。前者是有效策略，后者作了错误的全称断言。V0 将它写成一个 $+1>-1$ pair record。

这也说明：当前实现不是先拿同题 20 条独立完整 solution 做文本匹配，再猜测哪些 prefix 相同；它直接读取原数据 node 内已有的 alternatives 和 ratings。

### 4.3 candidate 怎么配对：不是一一对应

设一个 node 的 labels 为：

~~~text
Positive candidates: p1, p2
Negative candidates: n1, n2, n3, n4, n5, n6, n7, n8
~~~

V0 使用 Cartesian product，得到：

$$
\{(p_i,n_j):i\in\{1,2\},j\in\{1,\ldots,8\}\}.
$$

所以这个 node 产生 $2\times8=16$ 个 pair，而不是 2 个或 8 个一一对应 pairs。每个 pair 都使用同一个 context；当前 V0 每个 pair 的 weight 都是 1.0。它会强化“在这个 state 下任何正候选都应排在任何负候选之前”，但也意味着 candidate 多的 node 会得到更大训练权重。这正是后文的 multiplicity risk。

### 4.4 三种 processed artifacts

| Artifact | 一行代表什么 | 用途 |
|---|---|---|
| data/processed/pointwise | 一个 candidate step + binary label | 训练 absolute correctness score |
| data/processed/pairs_v0 | 同一 node 的一个 positive/negative candidate pair | 训练 local ranking |
| data/processed/trajectories | selected trajectory、每步 rating、known first-error index | first-error evaluation |

同一个 underlying human label 可以同时出现在 pointwise artifact 中，也参与一个或多个 pair records；这没有造成 train/test leakage，因为 split 是按 problem 做的。

### 4.5 annotated node 是怎样收集出来的

这里没有 UNSNode；准确名称是 **annotated step node**。它可以用 phase-1 的简化收集流程理解：

~~~text
current selected prefix
        ↓
模型在该 prefix 下提出若干 candidate next steps
        ↓
标注者分别给每个 candidate 打 +1 / 0 / -1
        ↓
选择一个 candidate 继续（若全错，phase 1 可由人补一个正确 step）
        ↓
得到新的 selected prefix，进入下一个 node
~~~

因此，一个 node 的 completions 不是事后从 20 条独立完整 solutions 模糊匹配出来的；它们在收集时就被呈现为同一 current prefix 下的 alternatives。phase 2 更接近“先生成完整轨迹、沿轨迹标注、遇到 first negative 停止”，所以 pair-rich nodes 的来源和分布并不均匀。

## 5. exact-prefix pairs

### 5.1 prefix 的定义

对第 $t$ 个 annotated node：

$$
c_t=(\mathrm{problem},s_1,\ldots,s_{t-1}).
$$

prefix 是当前 candidate step 之前已确定的 reasoning steps。若候选在第一步就不同，prefix 可以为空。

一个 exact-prefix pair 不是同一条 trajectory 的前后两个 steps，而是同一 reasoning node 的两个**替代 next steps**：

~~~text
Problem:  Solve 3x + 2 = 11
Prefix:   3x + 2 = 11

Candidate A: 3x = 9     rating = +1
Candidate B: 3x = 13    rating = -1

Preference: A > B
~~~

V0 只使用最干净的 $+1 > -1$ pairs。

### 5.2 为什么 same problem 不够

只固定同一道题、却允许两个 candidate 来自不同 previous steps，依然存在混淆：

- 一个 candidate 可能基于已错误的 algebra，另一个基于正确 prefix；
- prefix 长度、已化简程度与语言形式不同；
- 模型可能利用 context 难度，而非判断 next step 的好坏。

exact prefix 固定 reasoning state $c$，让目标成为：

> At exactly the same reasoning state, which next action is better?

两条 trajectory 一旦在前文分叉，即使后续文字偶然相同，也不应跨 trajectory 配对；它们不在同一个 reasoning state。

### 5.3 当前 V0 怎样获得 prefix 与 pair

当前实现逐条读取 PRM800K record 的 annotated step nodes：

1. 从该 record 的 selected trajectory 累积 prefix；
2. 在当前 node 读取所有 labelled candidate completions；
3. 对该 node 内每个 positive candidate 与每个 negative candidate 取 Cartesian product；
4. 保存同一个 problem、同一个 prefix、不同的 positive / negative next steps。

因此当前实现只在同一 node 内配对，不跨题、不跨 node、不把 sequential steps 互配。

## 6. pointwise、pairwise 与 hybrid 的分工

共用 reward function：

$$
r_\phi(c,s)\in\mathbb{R},
$$

输入为 problem + prefix + candidate step。V0 采用 scalar reward head，忽略 neutral，$+1\mapsto1$、$-1\mapsto0$。

### 6.1 Pointwise：absolute calibration

pointwise 不要求一个 node 有两个 candidates，因此也使用 singleton nodes：

$$
\mathcal{L}_{\mathrm{point}}
=\operatorname{BCEWithLogits}(r_\phi(c,s),y).
$$

它让高 score 有“单步可接受”、低 score 有“单步错误”的绝对语义。

### 6.2 Pairwise：local ranking

pairwise 使用同 context 的 $s^+$ 与 $s^-$：

$$
\mathcal{L}_{\mathrm{pair}}
=-\log\sigma\left(r_\phi(c,s^+)-r_\phi(c,s^-)\right).
$$

它显式要求好 candidate 的 reward 高于坏 candidate；它并不使用 first error 之后那些未标注 steps。

### 6.3 Hybrid：两件事同时要

$$
\mathcal{L}_{\mathrm{hybrid}}
=\mathcal{L}_{\mathrm{point}}
+\lambda\mathcal{L}_{\mathrm{pair}},
\quad\lambda\in\{0.1,0.3,0.5,1.0\}.
$$

- pointwise 提供 calibration，支持“什么分数算错”的阈值；
- pairwise 强化同一 state 内的相对排序；
- pairwise-only 可能排序准确，却缺乏稳定的 absolute threshold；hybrid 是主方法的理由。

pairwise loss 不做 step-probability product，因此不会自动加剧原论文 product aggregation 的短解偏置。V0 的 optional Best-of-N 计划以 minimum step score 聚合，仍应单独检查 aggregation 与解答长度的敏感性。

### 6.4 只有一个 candidate 的 node 在 hybrid 中怎样训练

singleton node 没有 pair，不代表不能用于 hybrid model。它只贡献 pointwise update：

$$
(c,s,y)\longrightarrow\mathcal{L}_{\mathrm{point}}.
$$

pair-capable node 的 candidates 则可同时贡献 pointwise examples 和 pairwise records。一个 hybrid update 的概念形式是：

~~~text
任意 nodes 的 pointwise batch  ──→ L_point
pair-capable nodes 的 pairwise batch ──→ L_pair
同一个 encoder + reward head           ──→ L_point + λ L_pair
~~~

两种 batch 不需要逐行一一对应；它们训练的是同一个共享 reward function。singleton examples 给模型大量覆盖面和 absolute calibration，pairwise examples 给模型一部分高质量 state 内的 relative ranking signal。

这在统计上是合理的 multi-objective training，但有真实实验风险：pairwise subset 的分布可能特殊，而 hybrid 还会额外看到 pairwise records。若要做严格因果 claim，应加入 restricted-pointwise baseline：只用能形成 pairs 的 underlying nodes 训练 pointwise，并匹配 node、label 与 update budget，再比较是否仍有 hybrid 增益。

### 6.5 为什么可能跨数学题泛化，又为什么没有保证

模型不会为每一道题保存一张 candidate 对照表。所有例子共同更新同一个参数化函数 $r_\phi(c,s)$；它希望从 problem、prefix、candidate 的文本关系中学习可复用模式，例如：

- 等式移项是否保持等价；
- 算术是否与上一行数值一致；
- candidate 是否从 prefix 的正确状态自然推进；
- candidate 是否作出被 prefix 反驳的过强结论。

训练/validation/test 按 unique problem text 切分，因此 test problems 不在 train 中。若 test performance 好，说明该 scorer 对同一 MATH-style distribution 中的未见题目有经验性泛化；它不构成对任意数学题、更难证明题或其他领域的保证。pairwise loss 是否提升这种泛化正是本项目要检验的假设，而非已知定理。

## 7. first-error localization 怎样做

只对具有已知 first-error index 的 selected trajectories 评估。对真实 trajectory 的每一步，以之前真实 steps 为 prefix，产生：

$$
r_1,r_2,\ldots,r_T.
$$

在 validation set 选阈值 $\tau$，最大化 first-error within $\pm1$；固定阈值后才评估 test：

$$
\hat t=\min\{t:r_t<\tau\}.
$$

没有 score 低于阈值时，当前实现记作 $T+1$，即未检测到 error。报告 exact match、within $\pm1$、mean absolute error 与 detection rate。

pairwise accuracy 问“同 context 下是否排对 alternatives”；first-error metric 问“真实 trajectory 上是否及时发现首错”。两者不同。

## 8. 风险、annotation budget 与防守说法

### 8.1 pairs 没有创造人工信息

若一个 node 有 2 个 positive、2 个 negative candidates，4 个原始 human labels 可产生 $2\times2=4$ pairs。若有 10 个正、10 个负，20 个 labels 可产生 100 pairs。

pair 数不是 annotation budget。**Annotation budget** 是获得底层 human labels 所花的标注数量、annotated nodes 或人工时间。pair 是把已有 labels 重新组织出的 correlated training records。

所以在没有严格控制前，不能说“236k pairs = 236k 新的人类偏好”。

### 8.2 pair-rich nodes 为何可能不代表普通 reasoning states

一条 trajectory 可以有很多 steps，但在某个 state 只可能有一个被标 candidate；该 state 仍可用于 pointwise，却无法形成 exact-prefix pair。PRM800K phase 1 专门收集过 alternative completions；phase 2 则多为预生成整条 solution，遇到 first negative step 停止。因此多个 alternatives 的 nodes 是选择性子集。

风险：

1. pairwise gain 可能只泛化到 explicit-alternative states，未必泛化到最常见的 singleton states；
2. 同一 node 的 Cartesian pairs 高度相关；candidate 多的 node 会在 pairwise loss 中被重复放大；
3. 控制相同 pair-record 数不等于控制相同人工标注成本。

当前 V0 总数据：

| 内容 | 数量 |
|---|---:|
| pointwise $+1/-1$ examples | 935,143 |
| exact-prefix $+1>-1$ pairs | 236,539 |
| known-first-error trajectories | 85,316 |

当前 configs 让 pointwise 与 pairwise 各最多抽 50,000 个训练 **records**。这控制的是训练记录/更新预算，**不是**严格 annotation-budget control。

若报告要主张 label efficiency，应额外按相同 problems 或 annotated context nodes 抽样，并同时报告 unique labels、unique contexts 与 pairs；必要时对每个 context 限制 pair 数或重加权。

### 8.3 对老师的安全回答

> Same problem is not enough: a next step can be valid only relative to the exact reasoning state created by all previous steps. We therefore construct pairs only from alternative candidates at the same annotated node. Pointwise supervision learns whether individual steps are acceptable or incorrect, while pairwise supervision explicitly requires the preferred candidate to outrank the rejected candidate under the same context. The pairs do not create new human labels; they reuse ordinal structure already implicit in the labels. The main risk is selection and multiplicity: pair-rich nodes may not represent singleton reasoning states, and pairs from one node are correlated. This is why our main method is hybrid and why label-efficiency claims need an underlying-annotation-budget control.
