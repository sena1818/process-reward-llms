# bwUniCluster 3.0：PRM 项目的每日操作与 Slurm 提交手册

> 项目仓库：`https://github.com/sena1818/process-reward-llms.git`
>
> 目标项目目录：`projects/step_preference_prm`
>
> 适用环境：bwUniCluster 3.0（`uc3.scc.kit.edu`）
>
> 最后核对：2026-08-08，依据 bwHPC 官方 Wiki 与当前项目的 `experiments/unicluster/` 脚本。

这份文件是“每天实际照着操作”的版本。完整的 Linux、模型、loss 与评估教学请读：

- [`2026-08-07_linux_slurm_prm_learning_manual_zh.md`](2026-08-07_linux_slurm_prm_learning_manual_zh.md)

---

## 0. 一页主流程：只按这个顺序推进

| 阶段 | 要做什么 | 目的 | 通过标志 | 现在的状态 |
|---|---|---|---|---|
| 1. 环境 | clone、module、`.venv`、依赖、tests | 确保代码和软件栈正确 | `17 passed` | 已完成 |
| 2. 数据 | 下载 PRM800K，提交 CPU data job | 生成训练样本，避免在 GPU 上做预处理 | `status: ok` | 已完成：job `6221573` |
| 3. 长度 profile | 提交 CPU tokenizer 统计 job | 确认 2048 context 是否会严重截断 | 生成 token-length JSON | 已通过：job `6221609` |
| 4. GPU smoke | Qwen smoke + encoder smoke，各 1 GPU、最多 30 min | 验证 CUDA、模型、LoRA、bf16、checkpoint | 两个 smoke 都成功 | Qwen 已通过；encoder 发现代码 bug |
| 5. Pilot | 1 H100、4 h | 测显存、吞吐、恢复和正式 walltime | 无 OOM；输出可用 | 暂缓，等 encoder 问题处理 |
| 6. 主实验 | H100 array，seed 42 | 比较 pointwise / pairwise / hybrid | 6 个 run 完整 | 等 pilot |
| 7. 评估汇总 | evaluation + summary | 产出表格、指标与结论 | `metrics.json`、summary | 等主实验 |

**现在不要提交 pilot 或正式 array；按这三件事做：**

1. 从 GitHub 拉取 encoder 截断修复；
2. 重跑 encoder smoke，确认它成功；
3. encoder smoke 重跑成功后，才提交 4 小时 H100 pilot。

> 注意：不要因 `PD (Priority)` 或 `PD (Resources)` 重复提交相同任务；不要在 login node 运行训练；不要在 smoke/pilot 成功前提交正式训练。

---

## 1. 当前状态与已验证结果

截至 2026-08-08，已经完成：

- bwUniCluster 3.0 服务注册；
- 成功 SSH 登录 UC3；
- 集群用户名为 `hd_cn362`；
- 已创建 workspace `prm_step_preference`，有效期 60 天；
- workspace 路径由 `ws_find prm_step_preference` 返回。
- 已通过 GitHub SSH 克隆仓库；`origin` 为
  `git@github.com:sena1818/process-reward-llms.git`，`main` 与远端同步；
- 已加载 `jupyter/ai`（CUDA 12.8），建立 `.venv` 并安装项目依赖；
- Python 为 3.11.11，PyTorch 为 `2.10.0+cu128`，`transformers` 为 4.57.6，
  `peft` 为 0.20.0；
- `python -m pytest tests -q` 已通过：`17 passed`；
- PRM800K 四个原始 JSONL 已下载，并由下载脚本完成 SHA-256 校验；
- 已确认唯一可用 Slurm account 为 `hd`；
- 首次 CPU 数据构建任务 `6221573` 已在节点 `uc3n003` 以 4 CPU、8 GB RAM
  成功完成（`COMPLETED / 0:0`，耗时 59 秒）；最终 data validation 输出
  `status: ok`。已得到 58,428 个严格 nodes、236,539 个 pair 和 935,143 个
  pointwise 样本。
- Qwen H100 smoke `6221594` 已在 `uc3n088` 成功完成（`COMPLETED / 0:0`，2 分 34 秒）。
  三个 smoke run 都写出了结果，日志以 `Qwen smoke OK` 结束；这证明 H100、CUDA、
  Qwen、LoRA、bf16、训练和 checkpoint 路径均正常。
- CPU token-length profile `6221609` 已在 `uc3n007` 成功完成（`COMPLETED / 0:0`，
  2 分 4 秒）。
- encoder smoke `6222039` 在 H100 `uc3n088` 上发现可复现的代码问题并失败
  （`FAILED / 1:0`）：通用 smoke profile 将 max length 固定为 128；encoder 的
  `truncation="only_first"` 策略不能截断一个本身已超过 128 token 的 candidate，
  因而 Transformers 抛出 `Truncation error`。这不是 GPU、CUDA、网络或 Slurm 问题。
- 后续修复：encoder packer 现在会预先分配 pair token budget，优先保留 candidate，
  仅在 candidate 本身过长时安全截断它，并记录 telemetry；新增回归测试覆盖该情形。
  修复推送后仍必须在 UC3 重跑 smoke，不能只凭本地单元测试视为集群验证完成。

进入 workspace 的固定写法是：

```bash
cd "$(ws_find prm_step_preference)"
```

这样无需手打完整 `/pfs/...` 路径；workspace 位置变化时命令仍然正确。

---

## 2. 必须先记住的模型

```text
你的 Mac
  │ SSH
  ▼
UC3 login node
  ├─ 编辑、Git、创建环境、传数据、提交/查看 job
  └─ 不运行训练或大规模数据处理
          │ sbatch / salloc
          ▼
Slurm 调度器
          │ 在资源可用时分配
          ▼
CPU/GPU compute node
  └─ 真正运行 Python、训练、推理和评估
```

`sbatch` 不是“立刻运行 Python”，而是一个资源申请和作业提交。Slurm 在合适的 CPU/GPU、内存和 walltime 可用时才启动任务。login node 上的计算密集任务可能被直接终止。

---

## 3. 每次登录 UC3：固定执行的命令

### 3.1 从 Mac 登录

在不在 Heidelberg 校园网时，先连接 Heidelberg VPN。然后：

```bash
ssh hd_cn362@uc3.scc.kit.edu
```

首次或 SSH key 状态失效时，按提示输入：

1. TOTP 一次性验证码；
2. bwUniCluster 的 service password。

登录后先确认：

```bash
hostname
pwd
whoami
```

预期：`whoami` 是 `hd_cn362`，初始目录在 `$HOME` 下。`hostname` 显示 login node 是正常的；这不是你实际训练会使用的 GPU 节点。

### 3.2 进入项目并恢复软件环境

每个新 SSH session 都运行以下命令：

```bash
cd "$(ws_find prm_step_preference)/Process_Reward_LLMs/projects/step_preference_prm"

module purge
module load jupyter/ai
source .venv/bin/activate

export PRM_PROJECT_ROOT="$(pwd -P)"
export PRM_VENV="${PRM_PROJECT_ROOT}/.venv"
export PRM_HF_HOME="${PRM_PROJECT_ROOT}/.hf-cache"

git status --short
squeue
```

| 命令 | 作用 |
|---|---|
| `cd "$(ws_find ...)"` | 进入真实项目位置；避免手打路径和拼写错误。 |
| `module purge` | 清空默认或遗留 module，避免混用软件栈。 |
| `module load jupyter/ai` | 加载 UC3 的 AI/PyTorch 软件栈。 |
| `source .venv/bin/activate` | 激活项目专用 Python 包环境。 |
| `PRM_PROJECT_ROOT` | 供 sbatch 脚本定位项目根目录。 |
| `PRM_VENV` | 供 sbatch 脚本激活相同 Python 环境。 |
| `PRM_HF_HOME` | 让 Hugging Face cache 位于 workspace，而非 HOME。 |
| `git status --short` | 提交前检查有没有未提交的代码改动。 |
| `squeue` | 看已有 Running/Pending 作业，防止重复提交。 |

每次重新 SSH 登录都要执行这些 `export`；环境变量只存在于当前 shell。项目的 `common.sh` 会故意检查这三个变量是否存在。

---

## 4. 第一次在 UC3 配置项目

如果 workspace 里还没有仓库，推荐使用已配置好的 GitHub SSH key：

```bash
cd "$(ws_find prm_step_preference)"
git clone git@github.com:sena1818/process-reward-llms.git Process_Reward_LLMs
cd Process_Reward_LLMs/projects/step_preference_prm

git remote -v
git branch --show-current
```

预期：`origin` 指向 `git@github.com:sena1818/process-reward-llms.git`，分支为 `main`。

如果 HTTPS clone 提示输入 `Password`，不要输入 GitHub 登录密码。GitHub 对 HTTPS Git
操作只接受 Personal Access Token；已配置 SSH key 时应始终使用上面的 SSH 地址。

### 4.1 创建 Python 环境

```bash
module purge
module load jupyter/ai

python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

mkdir -p logs
```

`--system-site-packages` 是有意设置：它复用 UC3 `jupyter/ai` module 提供的、与 CUDA 匹配的 PyTorch。不要轻易用普通 `pip install torch` 覆盖它，否则可能安装 CPU-only 版本。

检查安装：

```bash
which python
python --version
python -c "import torch, transformers, peft; print(torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__)"
```

login node 上 `torch.cuda.is_available()` 可能为 `False`；这是正常的。GPU 可见性必须在 Slurm 分配的 GPU node 中检查。

### 4.2 获取 raw data

Git 不包含 `data/`、`outputs/`、checkpoint 或 Hugging Face cache。下载 PRM800K：

```bash
python scripts/download_prm800k.py --all
ls -lh data/raw/prm800k/
```

看到四个 JSONL split 后才继续。不要把本机生成的 `data/processed` 当成自动同步到 UC3；它们需要在 UC3 上重新构建。

### 4.3 最小检查

这是短检查，不是训练：

```bash
python -m pytest tests -q
```

测试通过只说明代码中的数据/loss/config/metric 不变量正常；它不代表大模型已经在 UC3 GPU 上跑通。

---

## 5. UC3 队列与“申请多长 GPU 时间”

### 5.1 本项目应使用的队列

| 阶段 | Queue | 申请时间 | 原因 |
|---|---|---:|---|
| data materialization | `cpu` | 2 h | JSONL 读取、数据构建，GPU 没必要。 |
| token profile | `cpu` | 2 h | 只跑 tokenizer 和统计。 |
| GPU smoke / debug | `dev_gpu_h100` | 30 min | 只验证 Qwen、LoRA、bf16、checkpoint 路径。 |
| Qwen pilot | `gpu_h100` | 4 h | 生产模型/长度下测吞吐和显存。 |
| 正式训练 | `gpu_h100` | 初始 36 h | 六个受控主实验；pilot 后再判断是否调整。 |
| 评估 | `gpu_h100` | 24 h | 大量 forward-only candidate/trajectory 打分。 |

UC3 官方规则：`dev_gpu_h100` 只用于 debugging 或 performance optimization，最大 30 分钟；每用户只允许一个运行中的 development job，最多排队 3 个。`gpu_h100` 是 regular queue，单 GPU 的最大 walltime 为 72 小时。正式项目不要把 development queue 当成常规训练队列。

### 5.2 调度优先级：按我个人，还是按学校？

答案是：**两层都会影响，不能简单理解成只按个人或只按学校。**

```text
你的 job 的优先级
  ├─ 你所在大学在 UC3 的投资/份额
  ├─ 该大学所有成员近期已使用的资源
  ├─ job 本身的资源形状：GPU/CPU、内存、申请时长、分区
  ├─ 队列中其他 job 与可用于 backfill 的时间空隙
  └─ 集群的 QoS / 调度策略
```

UC3 官方明确说明：fair-share 会考虑各大学的投资份额，以及该大学成员已经使用的资源。
因此，同一学校的整体近期使用量较高时，你的作业优先级可能下降；你自己的使用也包含在
“该大学成员”的总使用量中。另一方面，UC3 还实施每用户同时运行资源的硬限制：同一用户
跨所有 running job 最多使用 1920 个物理 CPU core。这个限制防止单个用户长期独占资源，
并不意味着 GPU 或排队优先级只按个人计算。

对当前作业，`scontrol show job 6221573` 显示 `Account=hd`、`QOS=normal`、
`Priority=5988`。这些字段说明 Slurm 正按你的 account 和普通 QoS 调度；它们不是可以
手工“调高”的分数。最有效的做法是如实申请资源、先跑短 smoke/pilot、避免重复提交，
让小而短的 job 有机会利用 backfill 空隙。

### 5.3 H100 还是 A100？

对这个项目要区分“**验证代码**”和“**形成正式实验数据**”两种目标：

| 情况 | 选择 | 原因 |
|---|---|---|
| 30 分钟 smoke | H100 优先；A100 也可用 | Qwen-1.5B + LoRA、bf16 与 CUDA 都能在两类 NVIDIA GPU 上运行；smoke 的目的是验证路径，不比较速度。 |
| 4 小时 pilot、正式训练与评估 | H100 | 当前项目的 sbatch 文件、显存/吞吐预估和正式可比性都以 H100 为目标。 |
| H100 development queue 没资源，但 A100 short 有空闲 | 可以将**一次性 smoke** 改投 A100 short | 先验证代码是合理的；日志必须记录实际 GPU 型号，不能把该吞吐直接当作 H100 pilot 估计。 |

仓库里的 `01b_qwen_smoke.sbatch` 默认写的是 `dev_gpu_h100`，这正是推荐的
H100 smoke 路线。若选择 A100，只在提交命令上覆盖 partition，不修改版本控制的
脚本：

```bash
sbatch --account=hd --export=ALL --partition=gpu_a100_short \
  experiments/unicluster/01b_qwen_smoke.sbatch
```

在切到 `*_short` queue 前，先阅读当前限制；这些 partition 名称和限制可能随 UC3
配置变化：

```bash
scontrol show partition gpu_h100_short
scontrol show partition gpu_a100_short
```

如有疑问，首选等待 `dev_gpu_h100` 或使用已有的 `gpu_h100_short`；不要因为当前
空闲就把正式 4 小时 pilot 投到 A100 short。

### 5.4 为什么不要一开始申请 72 小时

`--time` 是“最长允许时间”，不是预测值。它到了就会杀掉作业；但申请远超实际需求的时间通常不利于调度。正确做法：

1. 用 30 min Qwen smoke 排除配置错误；
2. 用 4 h pilot 得到真实的 `candidates_per_second`；
3. 用 pilot 估算正式 run；
4. 给正式 run 加合理余量，当前项目先采用 36 h；
5. 如果 walltime 不够，在**同一配置**下用 `--resume` 恢复，而不是重头训练。

### 5.5 GPU、CPU、RAM 是三种不同资源

项目的典型 GPU 配置：

```bash
#SBATCH --partition=gpu_h100   # 选择 H100 regular queue
#SBATCH --gres=gpu:1           # 申请一张 GPU
#SBATCH --cpus-per-task=16     # 为该 Python 进程申请 CPU 线程
#SBATCH --mem=96G              # 主机 RAM，不是 GPU 显存
#SBATCH --time=04:00:00        # 最长运行 4 小时
```

`--mem=96G` 是 RAM；H100 本身的显存不是通过这个字段申请。当前项目已把这些资源写进 `.sbatch` 文件，因此先不要随意改。单进程单 GPU trainer 申请 4 张 GPU 不会自动变快，因为代码没有 DDP/multi-GPU 训练。

---

## 6. 可选：交互式 GPU 调试（当前不需要）

`salloc` 适合你需要手动运行 `nvidia-smi`、逐条试命令或复现报错时使用。当前已经提交
可复现的 Qwen batch smoke（`6221594`），因此**不要再额外申请交互式 GPU**，避免重复占用。

只有 batch smoke 失败、且需要人工定位 GPU 环境问题时，再申请一张开发 GPU：

```bash
salloc \
  --partition=dev_gpu_h100 \
  --gres=gpu:1 \
  --time=00:30:00
```

等待 Slurm 分配资源。拿到 shell 后执行：

```bash
hostname
nvidia-smi

cd "$(ws_find prm_step_preference)/Process_Reward_LLMs/projects/step_preference_prm"
module purge
module load jupyter/ai
source .venv/bin/activate

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

你应看到：

- hostname 不再是 login node；
- `nvidia-smi` 能显示 NVIDIA H100；
- Python 输出 `True`；
- `torch.cuda.get_device_name(0)` 显示 H100。

完成后：

```bash
exit
```

这会主动释放 GPU。`salloc` 适合交互式调试；`sbatch` 适合可重复、可留日志的项目任务。

---

## 7. 正式作业的完整提交顺序

以下全部在项目目录执行，并且第 3 节中的 module、venv、三个 `PRM_*` 环境变量已经设置。

### Step 1：CPU 数据构建（仅首次或需要重建数据时）

```bash
data_job="$(sbatch --parsable --account=hd --export=ALL \
  experiments/unicluster/00_materialize_cpu.sbatch)"
echo "data_job=${data_job}"
```

作用：audit → problem split → pairs → pointwise → strict nodes → data validation。

检查：

```bash
squeue -j "${data_job}"
scontrol show job "${data_job}"
```

`PD (Priority)` 表示已进入队列、正在等待调度，不是错误。日志通常在 job 获得计算节点
后才创建；不要在 `PD` 时因 `tail` 报“不存在”而重新提交。只有状态为 `R` 或 `CG` 后才运行：

```bash
tail -n 50 -f "logs/prm-data-v0-${data_job}.out"
```

`CG` 表示 completing：主计算已经完成，Slurm 正在收尾、刷写日志和文件。作业从 `squeue`
消失后，再用以下命令确认最终结果：

```bash
sacct -j "${data_job}" --format=JobID,JobName,State,Elapsed,ExitCode
```

真实首次运行记录：`data_job=6221573`，2026-08-08 在 `cpu` 分区、节点 `uc3n003` 上
运行，资源为 4 CPU、8 GB RAM、最长 2 小时；实际 59 秒完成，最终为
`COMPLETED / 0:0`，并写出 `status: ok`。

脚本已经在最后自动执行 `python scripts/validate_v0_data.py`。确认日志中出现
`"status": "ok"` 即可；不必在 login node 再重复跑一次。

### Step 2：profile 与 smoke

**目的：** profile 决定正式 context/显存方案；smoke 证明训练路径真的能在 GPU 上运行。

首次从零提交时，可以使用以下带依赖的版本：

```bash
profile_job="$(sbatch --parsable --account=hd --export=ALL \
  --dependency="afterok:${data_job}" \
  experiments/unicluster/01_token_profile_cpu.sbatch)"

encoder_smoke_job="$(sbatch --parsable --account=hd --export=ALL \
  --dependency="afterok:${data_job}" \
  experiments/unicluster/01_encoder_smoke.sbatch)"

qwen_smoke_job="$(sbatch --parsable --account=hd --export=ALL \
  --dependency="afterok:${data_job}" \
  experiments/unicluster/01b_qwen_smoke.sbatch)"

echo "profile=${profile_job}, encoder_smoke=${encoder_smoke_job}, qwen_smoke=${qwen_smoke_job}"
```

`afterok` 的意思是：只有 `data_job` 成功结束，当前任务才可以开始。这样不会用到半成品数据。

**当前这次运行：** data job、profile `6221609` 与 Qwen smoke `6221594` 均已成功完成。
encoder smoke `6222039` 失败；在拉取修复前不要重交相同脚本，因为会得到同一个截断错误：

```bash
sacct -j 6222039 --format=JobID,JobName,State,Elapsed,ExitCode
tail -n 80 logs/prm-enc-smoke-6222039.out
tail -n 80 logs/prm-enc-smoke-6222039.err
```

拉取修复后再提交 encoder smoke；同样优先使用 H100 short：

```bash
encoder_smoke_job="$(sbatch --parsable --account=hd --export=ALL \
  --partition=gpu_h100_short \
  experiments/unicluster/01_encoder_smoke.sbatch)"
echo "encoder_smoke_job=${encoder_smoke_job}"
```

检查：

```bash
squeue
tail -f "logs/prm-qwen-smoke-${qwen_smoke_job}.out"
jq '.' outputs/profiles/qwen_lora_token_lengths.json | less
```

Qwen smoke 成功的最低标准：三个 mode 都有 `last.pt`、`best.pt`、`history.json`，日志没有 CUDA OOM、NaN/Inf loss 或 tokenizer error，并以 `Qwen smoke OK` 结束。

---

### Step 3：Qwen pilot

```bash
pilot_job="$(sbatch --parsable \
  --dependency="afterok:${profile_job}:${encoder_smoke_job}:${qwen_smoke_job}" \
  experiments/unicluster/02_qwen_pilot.sbatch)"

echo "pilot_job=${pilot_job}"
```

pilot 不是科研结果；它要回答：

- 2048-token、1.5B Qwen、bf16 LoRA 是否 OOM；
- 实际吞吐 `candidates_per_second` 是多少；
- `last.pt` 是否可恢复；
- 36 h 正式 walltime 是否合理。

查看：

```bash
tail -f "logs/prm-qwen-pilot-${pilot_job}.out"
jq '.[] | {epoch, throughput, validation}' \
  outputs/qwen_lora_runs_pilot/qwen_lora_hybrid_v0_lam0.3_pilot/history.json | less
```

### Step 4：seed=42 的主实验 array

确认 pilot 后才提交：

```bash
main_job="$(sbatch --parsable experiments/unicluster/03_qwen_main_array.sbatch)"
echo "main_job=${main_job}"
```

该 array 的 six tasks：

| array id | 运行内容 |
|---:|---|
| 0 | pointwise |
| 1 | pairwise |
| 2 | hybrid，λ=0.1 |
| 3 | hybrid，λ=0.3 |
| 4 | hybrid，λ=0.5 |
| 5 | hybrid，λ=1.0 |

`%2` 限制最多同时跑两个单-GPU job。这是为了避免一口气占六张 GPU，也减少 I/O 压力。

### Step 5：评估和汇总

```bash
eval_job="$(sbatch --parsable \
  --dependency="afterok:${main_job}" \
  experiments/unicluster/04_qwen_eval.sbatch)"

summary_job="$(sbatch --parsable \
  --dependency="afterok:${eval_job}" \
  experiments/unicluster/05_qwen_summarize_cpu.sbatch)"

echo "eval_job=${eval_job}, summary_job=${summary_job}"
```

输出位置：

```text
outputs/qwen_lora_runs/<run-name>/metrics.json
outputs/qwen_lora_runs/v0_summary.json
```

先根据 validation 选择 hybrid λ，之后才解释 test。不要按 test 指标选择 λ。

---

## 8. 每天如何看“GPU 空闲了吗？我的 job 为什么不跑？”

### 查看即时空闲资源

```bash
sinfo_t_idle
```

重点看这些分区：

```text
dev_gpu_h100
gpu_h100
gpu_a100_il
gpu_h100_il
```

例如 `gpu_h100 : 0 nodes idle` 意味着当前没有立即可分配的 H100 节点资源；不是你的环境出错。UC3 当前系统满负荷运行时，排队更久是正常调度状态。

### 看自己的作业

```bash
squeue
squeue -l
```

常见状态：

| 状态 | 含义 | 你的动作 |
|---|---|---|
| `PD (Resources)` | 等符合需求的资源 | 等待；不要重复 sbatch。 |
| `PD (Dependency)` | 前置 job 尚未成功 | 用 `scontrol show job <前置ID>` 查失败原因。 |
| `R` | 正在运行 | `tail -f logs/...out` 看进度。 |
| `CG` | 正在收尾写文件 | 等待；不要重复提交。 |
| `COMPLETED` | 退出码 0 | 检查 checkpoint、metrics 和日志。 |
| `FAILED` / `TIMEOUT` | 有错误或超时 | 保存日志，修复或使用 resume。 |

看单个 job 的具体请求、account、reason、node：

```bash
scontrol show job <JOB_ID>
```

看估计启动时间：

```bash
squeue --start -j <JOB_ID>
```

`squeue --start` 在调度器不能可靠预测时可能只有表头、不显示日期；这不是错误。不要用
`watch squeue` 或高频循环轮询，UC3 明确要求避免以这种方式给 Slurm daemon 增加 RPC 压力。
手动间隔一段时间再运行一次 `squeue -u "$USER"` 即可。

取消误提交作业：

```bash
scancel <JOB_ID>
```

不要因为 job Pending 就提交第二份相同任务；这通常只会制造重复计算和更难管理的输出目录。

---

## 9. 日志、输出与恢复

### 日志在哪里

每个 `.sbatch` 在顶部指定：

```bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
```

含义：

- `%x`：job name，例如 `prm-qwen-pilot`；
- `%j`：job ID；
- `.out`：普通输出；
- `.err`：Python traceback、CUDA error 等错误输出。

实时看日志：

```bash
tail -f logs/<job-name>-<job-id>.out
tail -f logs/<job-name>-<job-id>.err
```

### `TIMEOUT`、节点故障时如何恢复

只有当配置、seed、data budget 都不变时，才用：

```bash
sbatch \
  --export=ALL,PRM_RESUME=1 \
  experiments/unicluster/03_qwen_main_array.sbatch
```

程序从 `last.pt` 的下一个完整 epoch 恢复。不要对正式 run 使用 `--overwrite`；它会删除可审计的输出。配置被修改时，resume signature 报错是保护机制，不是需要绕开的障碍。

### `$TMPDIR` 的正确使用

`$TMPDIR` 是计算节点的本地 SSD，只在 job 运行期间存在。UC3 官方特别建议 AI 训练把“单个节点反复读取”的临时数据复制到 `$TMPDIR`，因为它比 Lustre 更适合小文件/重复读取。

项目的 `experiments/unicluster/common.sh` 已经把可重建的 `nodes_v0` stage 到 `$TMPDIR`。不要把唯一 checkpoint 写入 `$TMPDIR`；它们必须保留在 workspace 的 `outputs/`。

---

## 10. 本项目的停止条件

不要跳过这些 gate：

- data job 后：`python scripts/validate_v0_data.py` 成功；
- token profile 后：candidate truncation 接近 0；
- Qwen smoke 后：GPU 可见、loss 有限、checkpoint 完整；
- pilot 后：无 OOM、吞吐可接受、可以估算正式 walltime；
- 正式 array 后：六个 run 完整；
- eval 后：只用 validation 选 λ，随后读取 test；
- 多 seed 前：固定 λ，不为每个 seed 重选。

这就是科研上的可复现性：不是“程序没有报错”就够，而是每一步都只在前置假设已经被验证后才继续。

---

## 11. 每日最短清单

```bash
# 1. 登录（在 Mac）
ssh hd_cn362@uc3.scc.kit.edu

# 2. 每个新 UC3 session 的环境恢复
cd "$(ws_find prm_step_preference)/Process_Reward_LLMs/projects/step_preference_prm"
module purge
module load jupyter/ai
source .venv/bin/activate
export PRM_PROJECT_ROOT="$(pwd -P)"
export PRM_VENV="${PRM_PROJECT_ROOT}/.venv"
export PRM_HF_HOME="${PRM_PROJECT_ROOT}/.hf-cache"

# 3. 提交前先检查
git status --short
squeue
sinfo_t_idle

# 4. 提交 CPU 数据任务（仅首次重建数据时）
sbatch experiments/unicluster/00_materialize_cpu.sbatch

# 5. 提交后看日志
tail -f logs/<实际日志文件>.out
```

---

## 12. 官方参考

- [bwUniCluster 3.0 登录](https://wiki.bwhpc.de/e/BwUniCluster3.0/Login)
- [UC3 Running Jobs、队列与 `sinfo_t_idle`](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs)
- [UC3 Slurm 选项、interactive jobs 与 GPU 示例](https://wiki.bwhpc.de/e/BwUniCluster3.0/Slurm)
- [UC3 硬件、Lustre、workspace 与 `$TMPDIR`](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture)
- [UC3 软件模块](https://wiki.bwhpc.de/e/BwUniCluster3.0/Software_Modules)
- [Workspace 命令参考](https://wiki.bwhpc.de/e/Workspace)
