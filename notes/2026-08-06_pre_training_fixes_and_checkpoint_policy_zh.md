# 2026-08-06 训练前的代码修正与 checkpoint 选择决策

用途：记录在提交 UniCluster 正式作业之前对训练/评测代码做的四处修改，以及
checkpoint 选择规则的决策依据。报告的 Experimental Setup 一节应引用本文第 2 节。

## 1. 结论摘要

| 项目 | 修改前 | 修改后 |
|---|---|---|
| LoRA / reward head 精度 | adapter 随 base 为 bf16，head 为 fp32 | 所有可训练参数提升为 fp32，冻结 base 保持 bf16 |
| 本机可运行性 | 只有 `cuda + autocast` 下不报 dtype 错 | CPU / MPS / CUDA 均可运行，可用小模型做 smoke |
| checkpoint 选择 | 评测读 `best.pt`（各自 objective 的 validation loss） | 评测默认读 `last.pt`（固定最终 epoch），`best.pt` 仅作诊断 |
| 评测数据路径 | staged root 存在时强制使用，缺目录即崩 | staged 目录不存在时回退到 workspace 副本 |
| `max_trajectories` 截断 | 取文件前 N 条，而文件是强排序的 | 改为与其它 cap 一致的 seeded 随机抽样 |

## 2. Checkpoint 选择规则（需要写进报告）

### 2.1 修改前的问题

原实现每个 epoch 末在 validation 上计算该 run 自己的训练目标，取最低者存为
`best.pt`，评测读取 `best.pt`。问题在于「该 run 自己的训练目标」对三种 objective
不是同一个量：

| run | validation loss |
|---|---|
| pointwise | $L_{\rm pt}$ |
| pairwise | $L_{\rm pair}$ |
| hybrid $\lambda$ | $L_{\rm pt}+\lambda L_{\rm pair}$ |

因此不同 run 可能停在不同 epoch，而这个差异由**不同的选择准则**产生。主表中的
差距就同时包含三个来源：loss 形状的差异、实际训练轮数的差异、以及某个 epoch 在
validation 上的随机波动。本项目的全部论证建立在「唯一变量是 supervision
objective」之上，这个不受控变量与之直接冲突。

### 2.2 已采纳的规则

> 所有 run 固定训练 3 个 epoch，不做早停；评测一律使用最终 epoch 的权重
> （`last.pt`）。`best.pt` 仍然保存，但只作为诊断，不进入任何报告数值。

对应英文表述：

> All runs are trained for a fixed number of epochs without early stopping,
> and every objective is evaluated at the same final epoch. Selecting the
> epoch by each run's own validation objective would apply a different
> criterion to pointwise, pairwise, and each hybrid lambda.

### 2.3 为什么不用「统一的 validation 指标」

另一个方案是三种 objective 都按同一个 validation 指标（如 node-macro pairwise
accuracy 或 step macro-F1）选 epoch。它在原理上可行，但有两个实际代价：

1. **成本**：需要对每个 epoch 的 checkpoint 跑一次完整评测（节点打分 + 轨迹打分
   + 阈值校准）。每个 run 多 2 次评测，seed=42 的六个主 run 就是十几个额外的
   H100 GPU-hours，全部花在选 epoch 上。
2. **双重选择**：若用 validation first-error within-1 选 epoch，又用同一指标选
   $\lambda$，则在同一个 validation 集合上做了两层选择，过拟合 validation 的风险
   上升，而 Phase B 的停止规则本来就依赖这个指标的可信度。

固定 epoch 数没有这两个问题，规则也更容易在论文里一句话说清。

### 2.4 实现

- `runner.py`：`last.pt` 现在与 `best.pt` 携带相同的评测侧元数据
  （`training_config`、`validation_loss`、`planned_epochs`），因此可以直接被
  evaluator 加载；`run.json` 增加 `checkpoint_policy` 字段说明两个 checkpoint 的
  角色。
- `evaluator.py`：新增 `evaluation.checkpoint`，取值 `last`（默认）或 `best`；
  实际使用的 checkpoint 及其选择规则写入 `metrics.json`，汇总里增加
  `checkpoint_policy_is_uniform`，防止一批结果混用两种规则而不被发现。
- `07_eval_all.py`：自动发现 run 时按所选 checkpoint 名字 glob。

若将来确实需要早停结果作对照，把 `checkpoint: best` 写进 eval config 即可，
但两种规则的结果**不能混进同一张表**。

## 3. 精度：可训练参数保持 fp32

`CausalLoRARewardModel` 以 `load_dtype`（默认 bf16）加载冻结 base，随后把所有
`requires_grad` 的参数提升为 fp32。理由：bf16 只有 8 位尾数，学习率 1e-4 量级的
AdamW 更新相对权重可能被直接舍入掉；而冻结的 15 亿参数不接受更新，保持 bf16 不
影响正确性，显存收益也全部保留。

PEFT 的 `lora.Linear.forward` 会把输入 cast 到 adapter 的 dtype、再把结果 cast
回 base 的 dtype，因此混合 dtype 是安全的；`forward` 里额外把 pooled hidden state
cast 到 reward head 的 dtype，使得在没有 autocast 的设备（CPU/MPS）上同样成立。

这项修改的副作用是**本机可以跑通完整的 causal-LoRA 代码路径**，见第 5 节。

## 4. 评测数据路径回退

集群作业把高频读取的数据复制到 `$TMPDIR`。原实现只要 `PRM_DATA_ROOT` 存在，就把
`nodes_v0`、`trajectories`、`pointwise` 三个目录一并指向本地，但
`04_qwen_eval.sbatch` 只复制了前两个。在 `report_full_pointwise: false` 时不会暴露，
一旦打开该开关，评测作业会在训练结束后才因 `FileNotFoundError` 失败。

现在 evaluator 对每个目录单独检查 staged 副本是否存在，不存在则回退到 workspace
路径；同时 `04_qwen_eval.sbatch` 也会复制 `pointwise`。

## 4.1 `max_trajectories` 的抽样口径

本机 smoke 评测意外暴露的问题：`score_trajectories` 原来通过读取文件**前 N 行**
实现 `max_trajectories`，而 trajectory 文件是强排序的。实测：

| 取前 N 条 | val 中有已知 first error 的数量 | test |
|---:|---:|---:|
| 8 | 0 / 8 | 0 / 8 |
| 64 | 0 / 64 | 0 / 64 |
| 128 | 28 / 128 | 26 / 128 |
| 400 | 83 / 400 | 77 / 400 |
| 全部 | 8,198 / 9,562（86%） | 7,158 / 8,450（85%） |

也就是说，一次「快速部分评测」拿到的子集，其含错率比整个 split 低 4 倍以上；
取前 64 条时评测器会直接因为「validation trajectories 里没有已知 first error」
抛错。同一份 eval config 里的 `max_nodes` 与 `max_pointwise_examples` 走
`IndexedJsonlDataset`，本来就是 seeded 随机抽样，因此三个 cap 的行为并不一致。

现在 `_read_jsonl` 在设置了上限时改用同样的 seeded 抽样（未设上限时仍然流式读取
全部记录，行为不变）；`evaluator` 按 split 传入不同 seed。修复后取 12 条约得到
10 条含错，与 split 的 86% 一致。

正式 eval config 里 `max_trajectories: null`，因此**主实验结果不受此问题影响**；
它影响的是任何一次调试用的部分评测。

## 5. 本机环境与 smoke

conda 环境 `llm`（Python 3.11），安装 `requirements.txt` 加 `pytest`：

```bash
conda create -y -n llm python=3.11
conda activate llm
python -m pip install -r requirements.txt pytest
```

训练脚本新增两个只用于 smoke 的开关：

- `--model`：覆盖 `model.name_or_path`，用小模型验证代码路径；
- `--device`：覆盖 `training.device`。

两者都会写进 `resolved_config.json` 和 resume 签名，所以被替换过 backbone 的 run
永远可以被识别出来，不会误当作正式结果。

`with_smoke_profile` 在目标设备不是 CUDA 时自动降为 fp32 并关闭 gradient
checkpointing：smoke 检验的是代码正确性而不是精度行为，而 CPU 上的 bf16 matmul
会走未加速的慢路径（实测同一个 5 步 smoke，bf16 超过 10 分钟未完成，fp32 约 5
分钟）。在 CUDA 上仍保持配置里的精度，因此集群 smoke 依然覆盖生产 autocast 路径。

本机验证命令：

```bash
export HF_HOME="$PWD/.hf-cache"
python scripts/04_train_pointwise.py --config experiments/qwen_lora_main/train_pointwise.yaml \
    --smoke --model Qwen/Qwen2.5-0.5B --device cpu
python scripts/05_train_pairwise.py  --config experiments/qwen_lora_main/train_pairwise.yaml \
    --smoke --model Qwen/Qwen2.5-0.5B --device cpu
python scripts/06_train_hybrid.py    --config experiments/qwen_lora_main/train_hybrid.yaml \
    --smoke --model Qwen/Qwen2.5-0.5B --device cpu --lambda-pair 0.3
python scripts/07_eval_all.py --config configs/eval_smoke.yaml
```

2026-08-06 实测：三种 mode 均正常前向/反向，hybrid 的 train loss 由 6.08 降到
3.25，`best.pt` / `last.pt` / `history.json` / `run.json` 全部生成，adapter 与
reward head 的 dtype 确认为 fp32。CPU 吞吐约 0.17 candidates/s，仅够做正确性验证。

## 6. 仍未处理的事项

- first-error 的虚警对照（在 `first_error_index is None` 的轨迹上统计误报率）
  尚未实现；test split 有 1,292 条这样的轨迹目前完全不参与评测。
- `report_full_pointwise` 在正式 eval config 中仍为 `false`；是否打开取决于是否
  要在报告中给出自然分布下的 step 指标。
- 正式 run 的时间估计仍依赖 token profile 与 pilot 实测。
