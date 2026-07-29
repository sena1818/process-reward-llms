# PRM 的 LLM 微调与 24/48 GB GPU 方案

结论先行：本项目训练的是 **verifier / process reward model (PRM)**，不是从零训练、更不是用 RL 训练一个解题 generator。对带人工标签的 PRM800K 做的是监督式微调：输入 `problem + prefix + candidate step`，输出一个标量 reward（或 `+ / 0 / -` 的概率）。LLM 版本应是 V0 encoder 实验之后的**受控扩展**，所有 pointwise、pairwise、hybrid 仍须使用同一基座、相同 split 与相同训练预算。

本项目的原始论文是 2023 年的 *Let's Verify Step by Step*，large-scale 模型从 **GPT-4 base model** 微调，时间上不可能是 GPT-4o；作者没有公开参数量、optimizer 或采用全参/LoRA/QLoRA 的细节，不能把其中任一项说成论文的复现设置。论文把 PRM 训练为对带已标注步骤的 solution prefix 预测三种 step label，并在附录写明使用低学习率、训练两 epoch；generator 在该研究中固定，并没有由 PRM 信号做 RL 更新。[原论文 PDF](https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf)

## 1. 这三种做法分别是什么

| 做法 | 更新什么 | 对本项目的建议 |
|---|---|---|
| 全参监督微调（full FT） | 基座所有参数 + 新 scalar head | 适合 `roberta-base` / `deberta-v3-base`；不建议把 1.5B/4B/7B LLM 作为默认方案。 |
| LoRA | 冻结基座；在选定线性层学习低秩矩阵，另训练 scalar head | LLM-PRM 的首选；仍然是 fine-tuning，只是更新量很小。 |
| QLoRA | 冻结的基座以 4-bit 保存；反传到 LoRA adapter 和 scalar head | 24/48 GB 下训练 LLM-PRM 的最稳妥默认。不是「不改变模型」：adapter 会改变最终函数，只是不直接写回基座权重。 |

LoRA 的定义正是冻结预训练权重、在 Transformer 层注入可训练的低秩分解矩阵；原论文报告其可大幅减少可训练参数与训练显存。[LoRA 原论文](https://arxiv.org/abs/2106.09685) QLoRA 则让梯度穿过冻结的 4-bit 基座流向 LoRA；其论文演示过单张 48 GB 卡微调 65B，但这**不是**对本项目的实际显存承诺。[QLoRA 原论文](https://arxiv.org/abs/2305.14314)

实现时，HF 官方也明确：量化模型通常不直接继续全参训练；用 PEFT adapter 才能在量化基座上训练。4-bit 实践配置用 `NF4`、double quant、bf16 compute，并在量化后调用 `prepare_model_for_kbit_training()`；若作 QLoRA-style 训练，可取 `target_modules="all-linear"`。[PEFT quantization guide](https://huggingface.co/docs/peft/main/developer_guides/quantization) 若用 causal LM 加 scalar reward head，TRL 特别要求将 `score` 放入 `modules_to_save`，否则该 head 不会被正确保存/训练。[TRL RewardTrainer](https://huggingface.co/docs/trl/reward_trainer)

这不是“再喂一些无标签数据去改分布”（continued pre-training）。只有在先额外使用大量**无标签数学语料**继续预训练时才属于那个阶段；本项目已有 `+1/0/-1` 和 preference pairs，应直接做 supervised PRM fine-tuning。

## 2. 数据量与目前真正要跑的量

PRM800K 官方发布称包含 80 万 step-level correctness labels；论文附录同时说明，未过滤原始释放版本含 1,085,590 个 step labels、101,599 条 solutions，而过滤后的实际 PRM 训练约为 80 万 labels / 7.5 万 solutions。因此不要把「80 万」误解为 80 万个独立 pair。[官方数据集仓库](https://github.com/openai/prm800k) [原论文 PDF，Appendix B–D](https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf)

本仓库经过 problem-level split 和 V0 过滤后的实际 records 是：

| artifact | train | val | test |
|---|---:|---:|---:|
| pointwise `+1/-1` | 768,020 | 88,547 | 78,576 |
| exact-prefix `+1>-1` pairs | 192,748 | 23,252 | 20,539 |
| first-error trajectories | 82,028 | 9,562 | 8,450 |

现有 configs 并不会一开始跑全量：每个来源最多抽 50,000 条，3 epochs、max length 512。故一个 run 的理论训练前/反向传播序列数为：pointwise `50k × 3 = 150k`；pairwise `50k × 3 × 2 = 300k`；hybrid 同时做一个 pointwise 与一对 pair，为 `450k`。四个 lambda 的 hybrid sweep 会是约 180 万条 model-sequences；加 pointwise/pairwise 基线约 225 万条，尚未计验证。全量训练会约扩大 4–15 倍，取决于 objective。

## 3. 选择什么模型

### 必做主表：encoder PRM，不需要 LoRA

继续把 `roberta-base + scalar head` 作为复现实验基线，并在算力允许时以 `microsoft/deberta-v3-base + scalar head` 做 stronger encoder replication；二者都全参监督微调。它们训练的是 reward scorer，完全符合本项目问题，并能用较少算力完成 pointwise / pairwise / four-lambda hybrid 的公平消融。

### 24 GB 的 LLM extension：推荐 Qwen2.5-Math-1.5B **base** + QLoRA

推荐模型 id：[`Qwen/Qwen2.5-Math-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)，而非 Instruct。原因是数据集来自 MATH，模型官方定位为中英文数学 CoT/TIR；更关键地，Qwen 自己明确说 base model 是 fine-tuning 的更好起点，Instruct 是聊天模型。该系列提供 1.5B/7B/72B 基座，且模型卡为 Apache-2.0。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)

建议：4-bit NF4 QLoRA，`r=16`、`alpha=32`、dropout `0.05`、`target_modules="all-linear"`，bf16，gradient checkpointing，`max_length=512` 起步，先用 50k budget。使用 `AutoModelForSequenceClassification(num_labels=1)` 或 causal backbone 的 pooled-final-token + 新 `score` head；所有 pointwise/pairwise/hybrid 共用这个同一个 score function。TRL 的 Bradley–Terry loss 正是 `-log sigmoid(r(chosen)-r(rejected))`，可直接用于 pure pairwise；本项目的 hybrid 仍以自定义训练循环更自然。[TRL reward loss](https://huggingface.co/docs/trl/reward_trainer)

### 48 GB：把预算花在完整实验，7B 只是一个单独扩展

先用同一 1.5B QLoRA 跑全量/重复 seed/label-efficiency，科研价值通常大于直接换 7B。若这些已完成，才增加 [`Qwen/Qwen2.5-Math-7B`](https://huggingface.co/Qwen/Qwen2.5-Math-7B) 的 QLoRA scale ablation；不做 7B full FT。若希望加入一个更新的非数学专用 reasoning 对照，可在 24 GB 用 [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) QLoRA、在 48 GB 尝试 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) QLoRA；它们应标为「general reasoning scale/control」，不能和 Math-specialized 结果混成“仅由 pairwise loss 导致”。

不要用 Qwen 的 72B mathematical reward model 作为训练基座：规模超出本项目，也引入了外部 reward-training data 的混杂；最多把它当不可比的零样本参考。

## 4. 显存与时间：可执行的估算

混合精度 + AdamW 的朴素全参训练，官方经验式约为 **18 bytes/parameter + activations**；activation 又随 batch、sequence length、depth、hidden size 增长。因此 1.5B 的全参 AdamW 光参数状态约 27 GB、7B 约 126 GB，尚未含 activation，说明 24 GB 不适合作为 LLM full FT，48 GB 也不应拿 7B full FT 冒险。[HF GPU-memory anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy)

| GPU | 合理任务 | 不建议 |
|---|---|---|
| 24 GB | 全参 RoBERTa/DeBERTa；1.5B QLoRA 很稳；4B QLoRA 先 smoke 后做 50k budget | 1.5B full FT 作为默认；7B full FT |
| 48 GB | 1.5B QLoRA 的全量/多 seed；7B QLoRA（512、checkpointing、small physical batch） | 7B full FT；一开始就大模型 sweep |

对「单张消费级/工作站 GPU、512 token、flash attention/driver/实际平均长度不同」只能给范围，不能在租卡前保证准确小时数。以每 run 50k、3 epoch 为例，24 GB 的 1.5B QLoRA 通常可按约 **2–6 h pointwise、4–12 h pairwise、6–18 h hybrid** 预留；48 GB 7B QLoRA 常约为其 **2–4 倍**，且可能因 OOM 降 batch 而更慢。encoder V0 在 24 GB 通常是小时级，整个六-run sweep 预留半天到一天较安全。全量 1.5B hybrid 单个 lambda 是约 3.46M model-sequences（对 50k hybrid 的 450k 约 7.7 倍），所以把上述 50k 计时乘约 8，再加验证与 checkpoint 的开销。

最可靠的租卡方法是：先跑 200 warm-up + 500 timed updates，记录 `model-sequences/s`、`torch.cuda.max_memory_allocated()`；再用 `总 model-sequences / 实测吞吐 × 1.2` 预订时间。先做 1k-example smoke、再 10k pilot，只有显存有至少 10–15% 余量才升到 50k 或全量。

## 5. 建议的决策顺序

1. 不租卡也先完成/验证 encoder V0 主表；这是 project 的主 claim。
2. 若租 24 GB，只加一个 `Qwen2.5-Math-1.5B base + QLoRA` LLM-PRM，先跑 50k 的 pointwise、pairwise、一个 `lambda=0.5` hybrid；再决定是否进行 sweep。
3. 若有 48 GB 和充足时间，优先把 1.5B 的结果补成 full-data、3 seeds 或 label-efficiency 曲线；最后才把 7B QLoRA 作为 scale ablation。

这样答辩时可以准确说：**我们微调的是一个 step verifier 的 scalar reward function；主实验是低成本全参 encoder，LLM extension 采用 QLoRA 的参数高效监督微调，而非从零训练或 policy RL。**
