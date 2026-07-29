# 训练代码审查、实验分层与 UniCluster 执行指南

更新日期：2026-07-30

## 1. 结论

原来的总体流程“本地整理代码 → 推送仓库 → UniCluster 拉取 → smoke
test → fine-tuning”方向正确，但不能从拉取代码直接跳到正式训练。当前
推荐的严格顺序是：

```text
冻结数据规则
  → CPU 重建与一致性校验
  → token 长度审计
  → encoder GPU smoke
  → Qwen-LoRA pilot 测吞吐/显存
  → seed=42 的六个主比较
  → 只用 validation 选择 lambda
  → 两个额外 seed 的确认实验
  → test + problem-cluster bootstrap
  → 资源允许时再做 label-efficiency / 4096 ablation
```

不要在 login node 上运行数据重建、token profiling、训练或完整评估。

## 2. 本次代码审查发现了什么

| 项目 | 审查前状态 | 当前处理 |
|---|---|---|
| 主训练 cohort | pointwise 和 pairwise 来自不同的 flat 数据集 | 三种 objective 统一使用 `nodes_v0` |
| loss 聚合 | flat candidate/pair batch mean | 先在 node 内 mean，再对 nodes mean |
| candidate 计算预算 | hybrid 使用两个 loader，candidate 会重复 forward | 同一个 node batch，每个 candidate 只 forward 一次 |
| 标签冲突 | 同一 node 中有相同文本同时标成 `+1/-1`，会产生无意义 self-pair | 严格 cohort 删除冲突文本并记录计数 |
| 长上下文 | 普通右截断可能丢掉 candidate 或最近 reasoning | Qwen 使用 problem head + recent-prefix suffix + candidate 的 2048 packing |
| Qwen 主模型 | 只有 encoder 训练路径 | 增加 Qwen2.5-Math-1.5B base、bf16 LoRA、scalar head |
| checkpoint | 只有每 epoch 的 best model | 增加 `last.pt`，保存 optimizer、scheduler、scaler、RNG 和配置签名 |
| 运行覆盖保护 | 重名运行可能静默覆盖 | 默认拒绝非空目录；显式使用 `--resume` 或 `--overwrite` |
| 可复现性 | 未记录完整运行环境 | 保存 resolved config、HF revision、包版本、GPU、Slurm 和数据统计 |
| 评估 | flat pair accuracy，缺少校准指标与置信区间 | strict node-macro pair accuracy、AUROC、Brier、ECE、first-error、problem-cluster bootstrap |
| 多 seed / label budget | 没有可靠入口 | 增加 `--seed`、`--max-train-nodes`；同 seed 的预算样本严格嵌套 |
| 集群执行 | 没有可直接提交的作业层 | 增加 CPU、smoke、pilot、train array、eval array、summary 作业 |

本地严格 cohort 已重建并通过一致性校验：

| split | nodes | candidates | derived pairs |
|---|---:|---:|---:|
| train | 47,852 | 210,160 | 192,415 |
| validation | 5,751 | 25,442 | 23,236 |
| test | 4,825 | 22,043 | 20,519 |
| total | 58,428 | 257,645 | 236,170 |

清理掉的冲突文本为 train/val/test = `111/7/6`。旧 flat pairs 的
236,539 与清理后严格 cohort 的 236,170 不应混为同一个统计口径。

## 3. 代码结构：主体与实验层

```text
step_preference_prm/
├── src/prm_pref/                      # 主体代码：可复用、受测试保护
│   ├── data/
│   │   ├── load_prm800k.py            # 原始 schema 适配与冲突清理
│   │   ├── materialize.py             # splits / flat artifacts / strict nodes
│   │   ├── datasets.py                # JSONL 随机访问与 node collator
│   │   └── input_packing.py           # encoder / causal input adapter
│   ├── models/
│   │   └── encoder_reward_model.py    # encoder 与 Qwen-LoRA scalar scorer
│   ├── training/
│   │   ├── configuration.py           # 配置校验、profile、resume 签名
│   │   ├── losses.py                  # node-balanced BCE / BT / hybrid
│   │   └── runner.py                  # 训练生命周期与 checkpoint
│   ├── eval/
│   │   ├── inference.py               # node / trajectory scoring
│   │   └── evaluator.py               # 阈值校准、指标、汇总
│   └── utils/metrics.py               # AUROC/Brier/ECE/bootstrap 等
├── scripts/                           # 稳定 CLI 入口
├── configs/                           # encoder sanity 配置
├── experiments/                       # 实验层：可替换的研究决策
│   ├── qwen_lora_main/                # 正式 Qwen 主实验配置
│   └── unicluster/                    # Slurm 资源与调度脚本
├── tests/                             # 数据、packing、loss、metric 测试
├── data/                              # 不进 Git；可由 raw 数据重建
└── outputs/                           # 不进 Git；模型、日志、评估结果
```

主体代码的关键接口是 `InputPacker`。训练和评估只依赖
`problem/prefix/candidate → token batch`，encoder 与 causal model 的
输入差异被隔离在 adapter 内。这样 2048/4096 packing 或未来更换模型时，
不需要改 loss、trainer 和 evaluator。

严格训练单位是一条 annotated exact-prefix node：

```text
node
  ├── problem
  ├── exact selected-step prefix
  └── candidates: [+1, -1, ...]
         ↓ 每个 candidate forward 一次
      scalar scores
         ├── node pointwise mean BCE
         └── node pairwise mean softplus(-(r+ - r-))
                    ↓
        outer mean over nodes
```

## 4. 主体配置与训练口径

正式配置位于 `experiments/qwen_lora_main/`：

- backbone：`Qwen/Qwen2.5-Math-1.5B` base，而不是 Instruct；
- adaptation：bf16 LoRA，rank 16，alpha 32，dropout 0.05；
- head：最后一个非 padding token 的 hidden state → scalar reward；
- context：2048；
- physical batch：2 nodes；
- gradient accumulation：16；
- effective batch：32 nodes；
- epochs：3；
- optimizer：AdamW，learning rate `1e-4`；
- data：完整 47,852 个严格训练 nodes；
- 单卡训练，不启用 DDP。

LoRA checkpoint 只保存 adapter 与 scalar head，恢复或评估时仍需要同一
base model revision。第一次下载模型时解析出的 Hugging Face revision 会
写入 resolved config 和 checkpoint。

## 5. 必做实验与停止规则

### Phase A：只验证代码，不做科研结论

1. CPU data materialization + validation。
2. 全训练集 token-length profile。
3. RoBERTa pointwise、pairwise、hybrid 三个 smoke。
4. Qwen hybrid `lambda=0.3` pilot。

任何一项失败都不要提交正式 array。Pilot 至少确认：

- loss 有限且能够 backward；
- `best.pt`、`last.pt`、`history.json` 均生成；
- `nvidia-smi` 没有 OOM；
- candidate truncation rate 可接受；
- 吞吐可以给出正式运行 ETA；
- `--resume` 能从下一 epoch 恢复。

### Phase B：seed=42 主比较

在完全相同的 backbone、node cohort、candidate-forward budget 下运行：

| array index | objective | lambda |
|---:|---|---:|
| 0 | pointwise | — |
| 1 | pairwise | — |
| 2 | hybrid | 0.1 |
| 3 | hybrid | 0.3 |
| 4 | hybrid | 0.5 |
| 5 | hybrid | 1.0 |

只允许根据 validation first-error within ±1 选 lambda；validation
node-macro pairwise accuracy 只用于打破平局。选完才读取 test 结果。

### Phase C：多 seed 确认

固定 Phase B 选出的 lambda，不重新为每个 seed 调 lambda。推荐 seeds：
`7, 42, 123`。对 pointwise、pairwise、selected-hybrid 各做三次；seed=42
的运行直接复用，因此需要新增 6 个训练运行。

最终表同时报告：

- 三个 seed 的 mean ± sample standard deviation；
- test problem-cluster bootstrap 95% CI；
- step macro-F1、AUROC、Brier、ECE-10；
- node-macro pairwise accuracy；
- first-error exact、within ±1、MAE。

### Phase D：资源足够后才做

Label-efficiency 使用训练 node 数，而不是 flat labels：

| budget | nodes |
|---:|---:|
| 10% | 4,785 |
| 30% | 14,356 |
| 50% | 23,926 |
| 100% | 47,852 |

先用 seed=42 跑 pointwise 与 selected-hybrid 的 10/30/50%；100% 复用主
实验。只有准备在报告中作 label-efficiency 统计结论时，才给所有预算补齐
三个 seed。

以下不是当前必做：

- 4096 context：只有 2048 profile 显示 prefix/candidate 截断明显，且
  误差分析表明长链受损时再做；
- 7B / QLoRA：属于模型规模扩展，不进入主比较；
- neutral label `0`：属于 V1 ordinal ablation；
- Best-of-N：目前缺少固定 solver generations 和 answer checker，代码未
  假装实现它。

## 6. UniCluster 一次性准备

仓库只跟踪源码、配置、测试和文档。`data/`、`outputs/`、HF cache、
venv 和 checkpoint 已被 `.gitignore` 排除，因此集群 clone/pull 后仍需
单独下载或同步 raw data。

在项目目录中执行；推荐项目、venv、HF cache 位于可持久化 workspace，
而不是临时目录：

```bash
module purge
module load jupyter/ai

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

mkdir -p logs

export PRM_PROJECT_ROOT="$(pwd -P)"
export PRM_VENV="${PRM_PROJECT_ROOT}/.venv"
export PRM_HF_HOME="${PRM_PROJECT_ROOT}/.hf-cache"
```

每次重新登录、提交作业前都要重新设置最后三个变量。若账号属于多个计费
project，在 `sbatch` 命令上增加 `--account=<你的项目>`；不要把未知
account 硬编码进共享脚本。

raw data 和 processed data 已被 `.gitignore` 排除，`git pull` 只会带来
代码。第一次在集群上可重新下载约 455 MB 的官方 PRM800K：

```bash
python scripts/download_prm800k.py --all
```

也可以从本机把 `data/raw/prm800k/*.jsonl` 同步到同一路径。四个 raw
JSONL 到齐后再提交 data materialization 作业。

确认队列和账号：

```bash
sinfo_t_idle
sacctmgr show assoc user="${USER}" format=User,Account,Partition
```

## 7. 推荐提交顺序

### 7.1 数据与 profile

```bash
cd "${PRM_PROJECT_ROOT}"

data_job="$(sbatch --parsable experiments/unicluster/00_materialize_cpu.sbatch)"
profile_job="$(sbatch --parsable \
  --dependency="afterok:${data_job}" \
  experiments/unicluster/01_token_profile_cpu.sbatch)"
smoke_job="$(sbatch --parsable \
  --dependency="afterok:${data_job}" \
  experiments/unicluster/01_encoder_smoke.sbatch)"
```

检查：

```bash
squeue -j "${data_job},${profile_job},${smoke_job}"
python scripts/validate_v0_data.py
```

长度报告位于
`outputs/profiles/qwen_lora_token_lengths.json`。如果 candidate truncation
接近 0、prefix truncation 只集中在极长链，保持 2048；不要因为少量长尾
直接把所有实验翻倍到 4096。

### 7.2 Qwen pilot

```bash
pilot_job="$(sbatch --parsable \
  --dependency="afterok:${profile_job}:${smoke_job}" \
  experiments/unicluster/02_qwen_pilot.sbatch)"
```

Pilot 输出被隔离在 `outputs/qwen_lora_runs_pilot/`，不会混入正式结果。

### 7.3 seed=42 正式训练、并行评估与汇总

先人工看完 pilot 的显存、吞吐、truncation 和 loss，再提交：

```bash
main_job="$(sbatch --parsable \
  experiments/unicluster/03_qwen_main_array.sbatch)"

eval_job="$(sbatch --parsable \
  --dependency="afterok:${main_job}" \
  experiments/unicluster/04_qwen_eval.sbatch)"

summary_job="$(sbatch --parsable \
  --dependency="afterok:${eval_job}" \
  experiments/unicluster/05_qwen_summarize_cpu.sbatch)"
```

训练与评估 array 都限制为最多两个并发单卡作业。不要请求整台 4-GPU
node：当前 trainer 是单进程单 GPU，给它 4 张卡不会加速。

### 7.4 断点恢复

如果 array 因 walltime 或节点问题结束，使用相同 index、seed 和 budget：

```bash
sbatch \
  --export=ALL,PRM_RESUME=1 \
  experiments/unicluster/03_qwen_main_array.sbatch
```

恢复粒度是完整 epoch。已经完成的 run 会立即退出，未完成的 run 从
`last.pt` 的下一 epoch 继续；修改训练配置后 resume 会因签名不一致而拒绝。

### 7.5 额外 seeds

假设 validation 选择 `lambda=0.3`，对应 array index 3：

```bash
seed7_train="$(sbatch --parsable \
  --array=0,1,3%2 \
  --export=ALL,PRM_SEED=7 \
  experiments/unicluster/03_qwen_main_array.sbatch)"

seed7_eval="$(sbatch --parsable \
  --dependency="afterok:${seed7_train}" \
  --array=0,1,3%2 \
  --export=ALL,PRM_SEED=7 \
  experiments/unicluster/04_qwen_eval.sbatch)"
```

对 seed 123 重复。两个 seed 的 eval 完成后，再提交一次
`05_qwen_summarize_cpu.sbatch`。如果选出的 lambda 不是 0.3，index 映射
按 Phase B 表替换。

### 7.6 Label-efficiency 示例

仍假设 selected lambda 是 0.3，只运行 pointwise 和 hybrid：

```bash
budget_job="$(sbatch --parsable \
  --array=0,3%2 \
  --export=ALL,PRM_MAX_TRAIN_NODES=4785 \
  experiments/unicluster/03_qwen_main_array.sbatch)"
```

14,356 和 23,926 同理。相同 seed 下的较小预算是较大预算的严格子集。

## 8. 资源建议

UniCluster 当前最适合本项目的是 `gpu_h100`：

| 阶段 | partition | GPU | CPU | 脚本 walltime | 说明 |
|---|---|---:|---:|---:|---|
| 数据重建 | `cpu` | 0 | 4 | 2 h | streaming JSONL，主要是 I/O |
| token profile | `cpu` | 0 | 8 | 2 h | 只加载 tokenizer |
| encoder smoke | `dev_gpu_h100` | 1 | 8 | 30 min | development queue 只用于调试 |
| Qwen pilot | `gpu_h100` | 1 | 16 | 4 h | 生产 shape、限制 steps |
| 每个正式 run | `gpu_h100` | 1 | 16 | 36 h | 初始保守上限，pilot 后缩短 |
| 每个 eval run | `gpu_h100` | 1 | 16 | 24 h | array 并行，含 trajectory scoring |
| summary | `cpu` | 0 | 2 | 30 min | 只聚合已有 metrics |

官方队列页面列出的 H100 node 每台有 4 张 GPU，regular `gpu_h100`
walltime 上限 72 h；但 GPU 必须显式用 `--gres=gpu:1` 请求。脚本已经这样
设置。H100 文档示例显示约 95,830 MiB 显存，因此 1.5B bf16 LoRA 不需要
先引入 4-bit quantization。

如果 H100 排队很久，可以在提交时覆盖为 A100：

```bash
sbatch --partition=gpu_a100_il \
  experiments/unicluster/02_qwen_pilot.sbatch
```

先在 A100 重新 pilot，再决定 walltime；不要直接沿用 H100 ETA。

### 用 pilot 估算，而不是猜训练时间

读取：

```text
outputs/qwen_lora_runs_pilot/
  qwen_lora_hybrid_v0_lam0.3_pilot/history.json
```

第一行中的 `throughput.candidates_per_second` 已把 validation 时间包含在
分母里，是偏保守的估计。完整一轮训练加 validation 约处理：

```text
3 × (210,160 train candidates + 25,442 val candidates)
= 706,806 candidate sequences
```

因此单个正式 run 的初步小时数：

```text
estimated_hours ≈ 706806 / candidates_per_second / 3600
```

再乘 `1.2` 作为 checkpoint、启动与波动余量。只有 pilot 实测值才应写入
资源申请或计划；在实测前，可把单 run 粗略看作 5–15 H100 GPU-hours，
六个 seed=42 主比较约 30–90 H100 GPU-hours。这只是排期区间，不是性能
承诺。

## 9. 存储与日志

严格 nodes 约 115 MB；全部 processed artifacts 约 1.4 GB；原始 PRM800K
约 455 MB。训练脚本会把高频读取的 `nodes_v0` 复制到作业本地
`$TMPDIR`，评估脚本也会复制 trajectories。模型、checkpoint 和最终结果
仍直接写回持久化 workspace。

`$TMPDIR` 会在作业结束时删除，不能把唯一 checkpoint 写在那里。当前
脚本只把可重建输入放入 `$TMPDIR`。

每个正式 run 至少应保留：

```text
best.pt
last.pt
tokenizer/
resolved_config.json
environment.json
history.json
run.json
metrics.json
```

全局结果为 `outputs/qwen_lora_runs/v0_summary.json`。

## 10. 当前仍未完成但不阻塞主训练的部分

1. 本机没有安装 Torch/Transformers，因而这里不能替代 UniCluster 的
   真实 GPU smoke、显存峰值和吞吐验证。
2. 4096 context 没有预先创建正式配置，因为是否运行必须由长度审计决定。
3. Best-of-N reranking 需要固定 solver generations、答案解析器和 outcome
   benchmark，目前不属于训练主体。
4. neutral `0` 的 ordinal loss 仍是单独 V1，不应混入 V0。
5. 当前 trainer 是单 GPU；若以后升级 7B 或确实需要 DDP，应作为新的
   scale experiment，而不是静默改变主实验。

## 11. 官方参考

- [bwUniCluster 3.0 Batch Queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues)
- [bwUniCluster 3.0 Slurm](https://wiki.bwhpc.de/e/BwUniCluster3.0/Slurm)
- [bwUniCluster 3.0 Filesystem Details](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture/Filesystem_Details)
- [Jupyter/AI module and virtual environments](https://wiki.bwhpc.de/e/Jupyter_at_SCC)
- [Qwen2.5-Math-1.5B model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [PEFT checkpoint format](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint)
