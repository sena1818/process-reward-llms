# 2026-07-17 老师反馈：项目决策与严格实现方案

> 说明：以下把口述中的「20148」按 **2048 tokens** 理解；若老师实际说的是别的长度，需以老师原话为准。这里的「中间标签」指 PRM800K 的 `0`（neutral）标签。

## 1. 已确认的项目决策

老师认可项目的基本问题：不改变 generator，而是研究同一份 PRM800K 人工过程标注中，**绝对的 step label** 与 **相同 prefix 下的相对偏好**各自带来什么信息。

因此项目应从「提出一种新的 pairwise PRM 方法」调整为更严格、也更可信的研究问题：

> 在固定的人类标注 node、problem-disjoint split、backbone、上下文长度和训练预算下，pointwise 的绝对校准与 exact-prefix pairwise 的局部排序分别贡献什么；hybrid 是否能改善 first-error localization，而不是只提高局部排序？

### V0（主实验）

- 只使用 `+1` 与 `-1`，**不训练也不评估 `0` 作为监督标签**。
- 只使用 exact-prefix `+1 > -1` 比较；两个候选必须来自同一个 `problem + previous prefix` node。
- `0` 仍保留在原始数据和 trajectory 中，但只作为之后的 V1 ordinal ablation。

这是合理的主实验，而非“丢失数据”：`0` 的语义是“未明显错误但没有有效推进”，比 `+1/-1` 更含混。先用语义最干净的数据回答核心问题，再单独研究 `+1 > 0 > -1` 是否有增益。

### V1（可选扩展）

- pointwise：把 `-1,0,+1` 视为有序的三档标签；
- pairwise：加入 `+1>0`、`0>-1`；
- 不应把该扩展和 V0 混在同一主结论中。

## 2. 上下文长度：2048 是 LLM 实验的默认值，4096 是受控扩展

### 不能直接把当前 RoBERTa 的 `512` 改为 `2048`

当前 V0 backbone 是 `roberta-base`。其官方 config 的 `max_position_embeddings=514`，所以它最多只能处理约 512 token；设置 `max_length: 2048` 不会让模型获得 2048 长度能力，反而会在位置 embedding 处失败。

因此应把模型分成两个清楚的 tier：

| tier | backbone | 长度 | 作用 |
|---|---|---:|---|
| sanity / 低成本基线 | RoBERTa-base（全参） | 512 | 验证数据与 loss 管线；不是长上下文主结论 |
| 主 LLM-PRM | 长上下文 causal LM + scalar reward head | 2048 | 三个 objective 的公平主比较 |
| 长度敏感性 | 与主 LLM 完全相同 | 4096 | 只检验 2048 截断是否影响结论 |

`2048` 与 `4096` 的比较必须在**同一个 backbone、同一种 LoRA/QLoRA 设置、同一个 node cohort**上进行；不能将 “RoBERTa-512” 和 “Qwen-2048” 的差异归因于 loss。

### 截断不能只靠 tokenizer 的默认行为

现有 `datasets.py` 使用 `truncation="only_first"`：它保留 candidate，但会从 context 的末尾截断。对 PRM 而言，这常恰好删掉最相关的**最近几步 reasoning**，不适合作为 2048/4096 主方案。

应该实现 `head-tail` / `recent-prefix` 截断：

1. 分别 tokenize `problem`、所有 `prefix` steps、`candidate`，不立即截断；
2. 给 candidate 保留固定上限 `C`（例如 512；先由长度审计确认）；
3. 给 problem 保留小的前段预算 `Q`（例如 384 或 512）；
4. 剩余预算用于 prefix，保留**最后**的 tokens / 最近的完整 steps；
5. 插入明确分隔符，例如 `Problem:`、`Previous steps:`、`Candidate step:`；
6. 对每条 record 写入或统计：原始 token 数、最终 token 数、problem/prefix/candidate 各自是否被截断。

先在 train split 上用目标 tokenizer 做一次无截断长度审计，报告 `p50/p90/p95/p99` 以及 `L=512/2048/4096` 的截断率。默认用 2048；仅当 2048 仍截断非小比例样本（例如超过 5%）或长链 subset 明显退化时，才执行 4096 敏感性实验。

## 3. 32 GB 显存：full fine-tuning 1.5B 不应作为主方案

对普通 mixed-precision AdamW，参数状态的常用下界估计为约 `18 bytes / parameter`（fp16+fp32 weights、fp32 gradients、Adam 的两份 fp32 states），还**不含** activation、临时 tensor、CUDA allocator 和 checkpoint 开销。

| 模型规模 | 仅参数/梯度/Adam 状态的估计 | 32 GB 下判断 |
|---:|---:|---|
| 0.6B | 10.8 GB | full FT 可作为可控方案；2048 通常有余量 |
| 0.8B | 14.4 GB | full FT 的合理上限；4096 仍需 micro-batch 1、checkpointing 后实测 |
| 1.0B | 18.0 GB | 2048 可能可跑，但不应承诺 4096 或较大 batch |
| 1.5B | 27.0 GB | 只剩约 5 GB；2048/4096 会非常脆弱，不能作为可靠主实验 |

因此老师的判断是对的：**标准 32 GB 单卡上，1.5B full FT 并非绝对数学上不可能，但对 2048/4096 的 PRM 训练不是稳妥计划。** 只有采用 8-bit optimizer、CPU offload、极小 batch 或其它特殊技巧时才可能勉强跑通；这些会增加工程变量、降低吞吐，也不利于完成多 seed / 多 loss 的公平实验。

本地机器当前是 Apple M4，未检测到 NVIDIA CUDA GPU；上表是对未来 32 GB NVIDIA 租卡/服务器的估计，最终必须做 1k-example warm-up 测量。

### full FT 是否有必要？

对本项目的主要因果问题，答案是：**没有必要把 1.5B LLM full FT 当作必须条件。** 研究变量是 supervision objective，不是“是否更新全部基座参数”。只要 pointwise、pairwise、hybrid 使用完全相同的预训练模型、相同 LoRA modules/rank、相同 reward head 和相同训练节点，LoRA/QLoRA 是有效且公平的微调方式。

保留一个全参训练的 RoBERTa/DeBERTa 基线仍有价值：它证明结论不完全依赖 LoRA。但不值得为了“全参”牺牲上下文长度、重复次数和严格消融。

### LoRA / QLoRA 的模型选择

- 如果目标是**稳定、数学专用、2048 主实验**：`Qwen/Qwen2.5-Math-1.5B` base + **bf16 LoRA** 是首选。1.5B 在 32 GB 下没有必要先引入 4-bit 量化误差；Qwen 官方也说明 base model 比 Instruct 更适合作为 fine-tuning 起点。
- 如果目标是「用 32 GB 做更大模型」：可把 `Qwen2.5-Math-7B` + 4-bit QLoRA 作为**单独的 scale extension**，起步为 length 2048、physical batch 1、gradient checkpointing；先跑 1k/10k pilot，不能预先承诺 4096。
- 如果老师坚持 full FT：选择约 0.6B--0.8B 长上下文模型，例如 `Qwen/Qwen3-0.6B`（不是 math-specialized）或同等级开放模型；不要宣称 Qwen2.5-Math 存在 0.8B 版本，官方 Math 系列的最小规模是 1.5B。

LoRA 推荐起点：bf16 base、gradient checkpointing、rank 16 / alpha 32 / dropout 0.05、`target_modules="all-linear"`，另保存并训练 scalar `score` head。仅当 4096 的 pilot OOM 或升级到 7B 时，再切换 4-bit NF4 + double quant 的 QLoRA；这样量化是节省显存的工程手段，而不是额外的研究变量。

## 4. 严格的统一数学定义

### 4.1 数据单位必须是 annotated node，而不是 Cartesian pair

令一个原始标注 node 为

$$
n=(c_n,A_n),\qquad c_n=(q_n,\mathrm{prefix}_n),
$$

其中 $A_n=\{(s_{ni},z_{ni})\}_{i=1}^{a_n}$ 是**同一个 context**的全部候选，V0 中 $z_{ni}\in\{0,1\}$ 分别表示 `$-1` / `$+1`。定义

$$
P_n=\{i:z_{ni}=1\},\quad R_n=\{j:z_{nj}=0\},\quad
m_n=|P_n||R_n|.
$$

训练 cohort 为

$$
\mathcal N=\{n:\ |P_n|>0,\ |R_n|>0\}.
$$

这意味着三种训练方式面对的是同一批 `problem + prefix` 节点和同一批人工 candidate ratings；pairwise 只是从这些**已有** ratings 推导出比较，不能把 $m_n$ 个 Cartesian pairs 当成 $m_n$ 次独立人工标注。

score function 统一为

$$
r_\theta(c,s)\in\mathbb R.
$$

### 4.2 Node-balanced pointwise objective

不要对扁平化后的 candidate records 直接平均，因为 candidate 多的 node 会获得更大权重。主 pointwise objective 应为

$$
\mathcal L_{\mathrm{pt}}
=\frac{1}{|\mathcal N|}\sum_{n\in\mathcal N}
  \frac{1}{|A_n|}\sum_{i\in A_n}
  \ell_{\mathrm{BCE}}(r_{ni},z_{ni}),
$$

$$
\ell_{\mathrm{BCE}}(r,z)=\operatorname{softplus}(r)-zr,
\qquad r_{ni}=r_\theta(c_n,s_{ni}).
$$

其中 $σ(r_{ni})$ 可解释为该 step 为 `$+1` 的概率。V0 的 class imbalance（全局正例更多）不能通过删除候选来“平衡”；可在 pointwise BCE 中加入只在 train split 估计的 class weight，并把 unweighted BCE 作为 ablation。

### 4.3 Node-balanced pairwise objective

对每一个 node 内所有正负候选比较：

$$
\mathcal L_{\mathrm{pair}}
=\frac{1}{|\mathcal N|}\sum_{n\in\mathcal N}
 \frac{1}{m_n}\sum_{i\in P_n}\sum_{j\in R_n}
 \operatorname{softplus}\left[-(r_{ni}-r_{nj})\right].
$$

这就是 Bradley--Terry / RankNet logistic loss：

$$
-\log\sigma(r_{ni}-r_{nj}).
$$

它保证每一个原始 node 的总权重为 1，不会因为一个 node 有 $2\times8=16$ 个 Cartesian pairs 就比一个 $1\times1$ node 重要 16 倍。

在 $Δ=r^+-r^-$ 处，

$$
\frac{\partial \ell_{\mathrm{pair}}}{\partial r^+}=\sigma(\Delta)-1,
\qquad
\frac{\partial \ell_{\mathrm{pair}}}{\partial r^-}=1-\sigma(\Delta).
$$

所以当两者同分（$Δ=0$）时，梯度分别为 $-1/2,+1/2$，直接把 preferred candidate 推高、rejected candidate 拉低。若 $r^+\gg r^-$，两个梯度趋近 0。

### 4.4 为什么 pointwise 与 pairwise 不同、但可以公平比较

若对每个 context 加任意常数 $b(c_n)$，

$$
r(c_n,s)\mapsto r(c_n,s)+b(c_n),
$$

则 pairwise 的差 $r^+-r^-$ 完全不变。因此 pairwise-only 只能保证**同一 context 内的排序**；它没有足够信息让不同题目之间的 score 可用同一阈值比较。BCE 则锚定了绝对正/负标签，提供校准与 first-error threshold 所需的绝对尺度。

Hybrid 的严格定义是

$$
\mathcal L_{\mathrm{hybrid}}
=\mathcal L_{\mathrm{pt}}+\lambda\mathcal L_{\mathrm{pair}},
$$

其中两项均在相同 node cohort 上按 node 平均。初始所有 score 为 0 时，BCE 和 pairwise loss 都是 $\log 2\approx0.693$，所以 $λ$ 有可解释的相对量级；从 `{0.1, 0.3, 0.5, 1.0}` 中只能用 validation 选择一次，test 不参与选择。

不要比较三种方法的 raw training loss 数字（目标不同）；应在同一 test node cohort 上比较预先注册的 step classification、node-macro pairwise accuracy、calibration 和 first-error 指标。

## 5. 严格公平的实验协议

### 固定条件

1. 首先按 `problem_id` 切 train/validation/test，之后才 materialize nodes；绝不按 step 或 pair 随机切分。
2. 训练、validation、test 均保留 node grouping；不能让同 node 的 pairs 分到不同 split。
3. 三种 objective 固定 backbone、tokenizer、max length、recent-prefix truncation、reward head、LoRA config、seed、optimizer、scheduler 和训练 node cohort。
4. 每一个训练预算 $b\in\{10\%,30\%,50\%,100\%\}$ 先从 train 中抽取同一组 node，再由这组 nodes 同时生成 pointwise/pairwise/hybrid 所需的信号。
5. 横轴报告 **annotated nodes 与 candidate ratings 数量**，不报告“pair 数量”作为标注预算。pair 是由相同人类 ratings 派生，不是新的 annotation。
6. 至少 3 个随机 seeds；置信区间按 `problem_id` cluster bootstrap，而不是把相关 pairs 假装独立样本。

### 计算公平

现有 `pairs_v0` 将 Cartesian pair 展平，并为每个 pair 重复 encoder forward。这会让拥有很多 pairs 的 node 获得更多梯度和更多计算，也使 pairwise 方法多次重算同一个 candidate。主实验不应继续使用这一形式。

正确做法：一个 batch 的单位是 node。每个 node 的全部 candidates 各 forward **一次**，随后在 score 上构造所有 node-internal pair losses。这样：

- pointwise、pairwise、hybrid 都看见同一批候选；
- pairwise 的所有组合只增加很小的 score-difference 计算，不重复 Transformer forward；
- 公平控制的 compute 单位是 candidate forwards / training nodes，而非 Cartesian pairs。

若某个 node 候选过多，固定 `K_max` 后在每一个 epoch 内从其 pairs 均匀抽 `K_max` 对，并保持至少一个正与一个负；这仍是 node-average loss 的无偏 Monte-Carlo 估计。采样 seed 和每个 node 的原始/保留候选数必须落盘。

### 评估

在固定 test nodes 上报告：

- step classification：AUROC、macro-F1、Brier、ECE；
- ranking：先在每个 node 内平均 pairwise accuracy，再对 nodes 平均（node-macro）；可附 flat-pair 指标，但不能作为主结果；
- first-error：阈值只在 validation trajectories 上选定，test 上固定评估 exact / within-1 / MAE / detection rate；
- 可选 downstream：固定 generator、相同候选池和相同 reranking rule 的 Best-of-N。

pairwise-only 的 calibration 可额外做 validation-only temperature scaling，但不能把它当作 hybrid 已有绝对锚定的替代；应分别报告 pre/post calibration。

## 6. 代码实施顺序

当前 pipeline 已完成 problem split、V0 pointwise/pair files、RoBERTa reward head 和基础训练。下一阶段不应只改一个 `max_length`，而应按下面顺序重构。

1. `scripts/profile_token_lengths.py`（新增）
   - 用主 LLM tokenizer 计算 node 输入长度；
   - 输出各 split 的分位数和 `512/2048/4096` 截断率；
   - 用此报告选择 2048 或启动 4096 ablation。
2. `src/prm_pref/data/materialize.py`（新增 node materialization）
   - 写 `nodes_v0/{train,val,test}.jsonl`；一行一个 `context_id`，包含该 context 的所有 `+1/-1` candidates、ratings 与候选数；
   - 保留已有 flat files 仅作回归检查，不作为主训练输入。
3. `src/prm_pref/data/datasets.py`（新增 `NodeCollator`）
   - batch 内 flatten candidates，并同时返回 `node_id`、label、node offsets；
   - 使用 recent-prefix truncation，记录 truncation telemetry。
4. `src/prm_pref/training/losses.py`（新增 node-wise losses）
   - `nodewise_pointwise_loss(scores, labels, node_ids)`：先 node 内 candidate mean，再对 node mean；
   - `nodewise_pairwise_loss(scores, labels, node_ids)`：node 内 Cartesian pairs 平均，再对 node mean；
   - `hybrid_loss` 只组合来自同一 node batch 的两项。
5. `src/prm_pref/training/runner.py`
   - 三种 mode 均读取同一个 node loader；
   - 日志输出 node 数、candidate forwards、pairs（仅统计，不当作独立样本）与 truncation rate；
   - pointwise / pairwise / hybrid 对每一个 seed 都使用同一 node sample list。
6. LLM backend（新增 causal reward model + LoRA config）
   - base causal LM 的最后一个非 padding token hidden state 接 scalar `score` head；
   - `score` head 与 LoRA adapter 同时保存；
   - 先 1k examples smoke，再 10k pilot，记录 peak VRAM、tokens/s、截断率；通过后再跑完整 2048 主矩阵。
7. V1 ordinal neutral（只在 V0 结果稳定后）
   - 使用 cumulative-link ordinal objective：$P(Y\ge k\mid r)=\sigma(r-\tau_k)$，$k=1,2$，其中阈值 $\tau_1<\tau_2$；
   - pairwise 只在同 node 内加入 `$+1>0`、`0>-1`，并将权重选择视为 validation ablation，而非先验事实。

## 7. 推荐的执行选择

最稳妥的主线是：

1. 用现有 RoBERTa-512 跑 smoke / sanity baseline；
2. 完成 node-balanced 重构和 tokenizer-length profile；
3. 用 `Qwen2.5-Math-1.5B` base + bf16 LoRA、2048、同一 node cohort 跑 pointwise / pairwise / hybrid；
4. 跑 3 seeds 与 10/30/50/100% 的 annotation-node budget；
5. 若 2048 截断率或长链结果显示有必要，**只选一个代表性设置**跑 4096；
6. 若还有 32 GB 预算，再将 7B QLoRA 作为 scale check；不要为 1.5B full FT 牺牲严格实验设计。

这样最终结论可以是关于 ranking、absolute calibration、长度和人工标注效率的结论，而不是“某个模型恰好跑得更大”。

## 8. 依据（官方资料）

- [Hugging Face: GPU memory usage](https://huggingface.co/docs/transformers/model_memory_anatomy)（mixed precision Adam 的 weights / gradients / optimizer states，以及 activation 随 sequence length 增长）
- [RoBERTa-base config](https://huggingface.co/FacebookAI/roberta-base/blob/main/config.json)（`max_position_embeddings=514`）
- [Qwen2.5-Math-1.5B model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)（base vs Instruct 与 Math 系列的可用规模）
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)（0.6B full-FT 备选与 32,768 context capacity）
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)（NF4、`prepare_model_for_kbit_training`、QLoRA 的 `all-linear`）
