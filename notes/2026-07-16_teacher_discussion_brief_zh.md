

我想做的是 **Process Reward Model (PRM)** 实验。PRM 可以理解成一个给推理过程当裁判的模型：它不负责解题，而是看到一道题、前面已经写出的步骤、以及当前的下一步之后，判断这一步靠不靠谱。

原论文的做法比较直接：把每一个 step 单独拿出来，给它一个标签。正确的 step 是正例，错误的 step 是负例，模型学会给正确 step 更高的分数。

我想改的地方不在于换一个更大的模型，也不在于训练 generator；而是在于**怎样使用已有的 step 标注**。

PRM800K 里有一类数据很适合利用：在同一道题、同一个已有推理前缀下，标注者看到了几个不同的「下一步候选」，并分别给出了 $+1$、$0$、$-1$ 的评价。例如：

```text
题目和前面步骤完全相同

候选 A：3x = 9       -> +1
候选 B：3x = 13      -> -1
```

这里其实不仅告诉我们「A 是对的、B 是错的」，还明确告诉我们：**在这个完全相同的推理状态下，A 应该排在 B 前面。**
在同一道题、同一个 previous prefix 下：
  正确下一步 > 中性下一步 > 错误下一步

我的 idea 就是：除了把 A、B 各自当成分类样本之外，也把它们作为一对比较样本来训练。然后看这种「局部比较信息」到底能不能让 PRM 更会判断推理步骤。

把 PRM800K 中 `+1 / 0 / -1` 的 step-level human labels 转换成 **ordinal step preferences**，并训练一个 **hybrid pointwise + pairwise Process Reward Model (PRM)**。

> 用 PRM800K 中同一个 reasoning prefix 下的备选步骤，给 PRM 增加“哪个下一步更好”的训练信号；比较它和普通逐步分类相比，对排序、错误定位和分数可靠性有什么影响。

---

## 数据到底怎么用

原始数据不是只有一条条完整的 solution。它可以理解成很多 annotated step node：

```text
problem + 已选中的前面步骤（prefix）
    ├── candidate step 1, rating = +1
    ├── candidate step 2, rating =  0
    └── candidate step 3, rating = -1
```

我会从每个 node 生成两种训练数据。

第一种是普通的 **pointwise data**：

```text
(problem, prefix, candidate step, label)
```

例如 A 是正例、B 是负例。它回答的是：

> 不和别人比较时，这个 step 本身看起来对不对？

第二种是 **pairwise data**：

```text
(problem, prefix, better candidate, worse candidate)
```

例如把 $+1$ 的 A 和 $-1$ 的 B 配成一对。它回答的是：

> 在同一个推理状态里，模型能不能把更好的下一步排在前面？

这个配对不需要我自己从不同答案里猜 prefix 是否相同。PRM800K 的一个 node 本来就是在相同 prefix 下出现的多个候选，所以 pair 是干净的。

第一版只用最清楚的 $+1$ 和 $-1$：

```text
+1 > -1
```

`0` 的意思比较微妙：它通常表示没有明显数学错误，但也没有很好地推进推理。所以我会先把它排除，确保第一版的问题足够干净。等 V0 得到结果之后，再做一个小实验看 $+1 > 0 > -1$ 这种三档信息是否真的有帮助。

#### 一个需要控制的细节

同一个 node 里如果有 2 个正候选和 8 个负候选，会自动得到 $2\times8=16$ 个 pair。它们共享同一个 prefix，不能简单地当作 16 个完全独立的人工判断。

因此我准备在主实验中让每个 node 的总权重一样：保留所有 pair、但把它们的 loss 除以这个 node 的 pair 数。这样可以避免候选特别多的 node 主导训练，也让「用了同样多的人类标注」这个比较更公平。

数据会先按 **problem** 分成 train / validation / test，再构造 sample 和 pair。不能按 step 随机切分，否则同一道题的相似 prefix 会同时出现在训练和测试中，结果会虚高。

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
| exact-prefix preference pairs 总数 | 379,652 |
| `+1 > -1` pairs | 236,539 |
| `+1 > 0` pairs | 54,766 |
| `0 > -1` pairs | 88,347 |

结论：

> PRM800K 中 exact-prefix preference pairs 的数量足够。本项目可以真实地使用“同一上下文下的 step preference”，而不需要依赖跨题配对、弱配对、Monte Carlo rollout 或 final-answer-based labeling。


---

## 模型使用相对小模型

主实验用一个公开的、能在普通 GPU 上训练的 Transformer encoder：`roberta-base`。它大约 1.25 亿参数。或者 Qwen2.5-Math-1.5B

输入是：

```text
[题目] + [前面已经选中的推理步骤] + [当前候选步骤]
```

模型最后输出一个标量 $r(c,s)$：

- $c$ 是题目和 prefix；
- $s$ 是当前 candidate step；
- $r(c,s)$ 越高，表示模型越相信这个 step 是好的。

结构就是：

```text
文本输入 -> RoBERTa -> 一个 linear head -> 一个 score
```

这里会下载已有的 `roberta-base` 权重，然后对 encoder 和最后的 head 一起做 full fine-tuning。换句话说，它不是从零预训练，也不是 LoRA；但模型规模足够小，适合把实验重点放在训练目标的差异上。

我不打算在 V0 使用 teacher model。PRM800K 已经有人类的 step labels；教师模型更适合「我自己新生成了大量推理、却没有人类标签」的场景。现在直接使用人类标签，问题更简单也更干净。

---

## 三种训练方式具体差在哪里

为了让比较公平，三种模型使用完全相同的 backbone、同一份 problem split、相同的训练步数；唯一主要改变是 loss。

### 1. 普通 pointwise PRM：基线

把 $+1$ 当作 $z=1$，$-1$ 当作 $z=0$。模型将 score 经过 sigmoid 变成一个「这一步为正」的概率：

$$
p=\sigma(r(c,s)).
$$

训练损失是普通 binary cross-entropy：

$$
\mathcal{L}_{\mathrm{point}}
=-\big[z\log p+(1-z)\log(1-p)\big].
$$

直觉上，这要求模型把正确 step 的分数推高、错误 step 的分数推低。它的优点是分数有绝对意义：理论上 $0.8$ 和 $0.2$ 可以作为“比较有把握”与“比较可疑”的信号。

### 2. Pairwise PRM：只学比较

对同一个 context $c$ 中的正负候选 $(s^+,s^-)$，训练目标是：

$$
r(c,s^+) > r(c,s^-).
$$

具体使用的 loss 是：

$$
\mathcal{L}_{\mathrm{pair}}
=-\log \sigma\big(r(c,s^+)-r(c,s^-)\big).
$$

如果正候选只比负候选高一点，loss 还比较大；如果高很多，loss 就小。

它并不要求某一步一定高于固定阈值，也不要求不同题目之间的 score 可以直接比较。它只关心：**同一个 prefix 下，能不能把更好的下一步排在前面。**

### 3. Hybrid PRM：同时保留两种信息

最后训练一个模型，同时看到 pointwise 和 pairwise loss：

$$
\mathcal{L}_{\mathrm{hybrid}}
=\mathcal{L}_{\mathrm{point}}+\lambda\mathcal{L}_{\mathrm{pair}}.
$$

$\lambda$ 是 pairwise loss 的权重。我会在 validation set 上试几个小的值，例如 $0.1,0.3,0.5,1.0$，然后只选择其中一个进入最终 test。

这个设计背后的判断很简单：

- pointwise 可能更擅长说“这个 step 是否有问题”；
- pairwise 可能更擅长在多个可选下一步里选更好的那个；
- hybrid 想看能否同时得到这两种能力，而不是只提升其中一个。

---

## 我打算怎样判断它有没有用

我不想只报一个 accuracy，因为三种训练方式学到的能力不完全一样。至少会从三个角度看。

### A. 单步判断

给一个 step，看模型能否判断它是正还是负：

- accuracy / macro-F1
- AUROC

这回答的是：模型有没有基本的 step correctness 能力？

### B. 同 prefix 的候选排序

在从未见过的 exact-prefix pairs 上，看模型是否满足：

$$
r(c,s^+) > r(c,s^-).
$$

这里用 pairwise accuracy。它最直接检验我的改动：模型是否更会在同一个推理状态下做选择。

### C. 找到一条解答最早出错的地方

对一条多步推理，依次给每一步打分，然后找第一个低于阈值的位置。和人工已知的 first error 比较：

- 是否刚好找对；
- 是否落在真实错误位置前后一步；
- 预测位置和真实位置的平均差距。

这是我最关心的过程层面指标。因为一个 PRM 最终有用，不只是给单步贴标签，而是应该在一长串 reasoning 里尽早指出哪里开始不可靠。

另外会报告 calibration（例如 Brier score / ECE）。这是为了避免 pairwise 模型虽然排得更对，但分数完全不能作为“这里是不是错了”的信号。

---

## 实施计划

项目里已经完成了数据审计、problem-level split、pointwise/pair 数据构造、模型和训练/evaluation 脚本。现在缺的是实际跑完训练并分析结果。

我会按下面顺序做：

1. 在有 PyTorch 和 Transformers 的 GPU 环境跑三个 smoke test，先确认数据、tokenizer、forward/backward 都通。
2. 训练 pointwise baseline，先确认它能给出合理的 step-level 表现。
3. 训练 pairwise model。
4. 训练几组 hybrid，validation 上选择 $λ$ 和 first-error 阈值。
5. 固定选择后只在 test 上评估一次，并对比三类指标。
6. 如果 V0 结果清楚，再做 neutral label 和 pair 数归一化的扩展实验。

主实验规模会先控制在：`roberta-base`、最大长度 512、每种训练信号最多 50k 条记录、3 epochs。这样不会因为算力把项目拖成“只跑得动一个模型”，也能把主要比较做完整。

---

## 我希望和老师确认的事情

我想请老师主要帮我判断这四点：

1. 这个问题是否够清楚：不是换模型，而是测试同一份 process labels 里的“单点标签”和“局部比较”分别有什么作用？
2. V0 是否应该只先做 $+1/-1$，把 neutral 作为第二阶段。
3. pair 数按 node 归一化是否应该作为主设置，还是保留所有 Cartesian pairs 才更合理？
4. 是否接受 `roberta-base` full fine-tuning 作为主实验模型；如果有额外时间或算力，再考虑更大的 math LLM？


> 我现在想把 PRM800K 里本来就有的局部选择信息用起来。原来的训练把每一步分别判对错；但数据里经常是同一个已有推理下有几个不同的下一步，并且人已经告诉我们哪个更好。我想比较：如果训练时既让模型知道每一步对不对，也让它学会在同一个推理状态下把更好的下一步排前面，最后它能不能更准确地给步骤打分、也更早定位整条推理的第一个错误。
