# V1 实验记录与预注册

> 这份文件有两个用途：**记录**已经执行过的每一次集群实验（job ID、日期、结论、
> 产物位置），以及**预注册** V1 确认性实验的协议。第 4 节的内容必须在 V1 sweep
> 提交之前冻结；跑完之后再改，它就不再是预注册。
>
> 操作细节（环境变量、workspace 布局、排队策略）见
> [`TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md`](TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md)。
> [`RUNBOOK_V0.md`](RUNBOOK_V0.md) 是更早的 encoder 版本，已不是主实验。

> **2026-09-30 提交状态更新**：18 个 V1 run 均有完整训练与评测归档。
> 复现入口见 [`SUBMISSION_README.md`](SUBMISSION_README.md)。本次只补充
> 执行状态和结果位置；第 4 节保留训练前冻结的协议。

## 1. 时间线

| 阶段 | 日期 | 结果 |
|---|---|---|
| V0 数据物化与 token profile | 2026-08-08 | 严格 cohort 58,428 nodes |
| V0 pilot | 2026-08-10 | 吞吐与显存确认 |
| **V0 主实验（6 run × 3 epoch）** | 2026-08-10 ~ 08-11 | 完成，写进 ACL 报告草稿 |
| V0 评测（epoch 3）与汇总 | 2026-08-11 ~ 08-12 | `v0_summary.json` |
| 探索性 epoch-1 评测与汇总 | 2026-08-13 | `v0_best_epoch1_exploratory_summary.json` |
| 全面代码/结果审计 | 2026-08-30 | 发现 epoch 3 过拟合，见第 3 节 |
| **V1 固定协议实验（18 run × 1 epoch）** | 2026-08-31 起 | 完成，18 份 metrics/history/environment 已归档 |

## 2. 已执行的集群作业（UC3）

集群 workspace：
`/pfs/work9/workspace/scratch/hd_cn362-prm_step_preference/Process_Reward_LLMs`

本地完整归档（1.7 GB，含 checkpoint、history、metrics、slurm 日志）：
`/Users/sena/Desktop/Heidelberg_SciComp/26SS/Process_Reward_LLMs_uc3_archive_2026-08-12`

| Job ID | 名称 | 日期 | 内容 | 状态 |
|---|---|---|---|---|
| 6221573 | prm-data-v0 | 08-08 | 物化 nodes/pointwise/trajectories | 完成 |
| 6221609 | prm-token-profile | 08-08 | 2048 token 覆盖率剖面 | 完成 |
| 6221594 | prm-qwen-smoke | 08-08 | H100 上的 causal-LoRA 冒烟 | 完成 |
| 6222039 / 6232117 | prm-enc-smoke | 08-08 / 08-10 | encoder 路径冒烟 | 完成 |
| 6232503 | prm-qwen-pilot | 08-10 | 吞吐与显存 | 完成 |
| **6235682** | prm-qwen-main | 08-10 ~ 08-11 | **V0 主实验，6 task** | 6/6 COMPLETED 0:0 |
| 6254986 | prm-qwen-eval | 08-11 | epoch-3（`last.pt`）评测 | 6/6 COMPLETED 0:0 |
| 6268292 | prm-qwen-summary | 08-12 | `v0_summary.json` | 完成 |
| **6268417** | prm-qwen-e1-eval | 08-13 | **epoch-1（`best.pt`）探索性评测** | 6/6 COMPLETED 0:0 |
| 6294975 | prm-qwen-e1-summary | 08-13 | epoch-1 汇总 | 完成 |
| 6743886 | prm-qwen-v1-pilot | 08-31 | V1 实现门禁（训练 + 评测） | 完成，三项检查全过 |
| **6743958** | prm-qwen-v1 | 08-31 | **V1 训练 seed 42，6 task** | 最终 6 个 run 已归档，含后续重投 |
| 6743959 | prm-qwen-v1-eval | 08-31 | seed 42 评测（依赖 6743958） | 6 份最终 metrics 已归档 |
| **6743960** | prm-qwen-v1 | 08-31 | **V1 训练 seed 7，6 task** | 最终 6 个 run 已归档，含后续重投 |
| 6743961 | prm-qwen-v1-eval | 08-31 | seed 7 评测（依赖 6743960） | 6 份最终 metrics 已归档 |
| **6743962** | prm-qwen-v1 | 08-31 | **V1 训练 seed 123，6 task** | 最终 6 个 run 已归档，含后续重投 |
| 6743963 | prm-qwen-v1-eval | 08-31 | seed 123 评测（依赖 6743962） | 6 份最终 metrics 已归档 |
| 6743964 | prm-qwen-v1-summary | 08-31 | 汇总（依赖三个 eval array） | 最终 18-run 汇总已归档 |

以上是原始提交 ID；后续重投的最终训练 ID 记录在各 run 的 environment
归档中。这里的完成状态依据最终产物，不等同于证明原始 array 每个 task
的 Slurm 退出状态。

代码版本：V0 主实验跑在 `145e343`；epoch-1 探索性评测跑在 `0fa739d`；V1 跑在
`0353318`（本文件的预注册提交为 `7a15cf7`，早于 6743958 起的 sweep）。归档里的 `cluster-git-state.txt` 记的是 `145e343`，是归档时刻的状态，
不覆盖 08-13 那次评测。

### 关键产物

- `submission/artifacts/v1_summary.json` —— 最终报告主表的 18-run 汇总
- `submission/artifacts/metrics/` —— 每个 V1 run 的详细评测结果
- `submission/artifacts/v1_scores.zip` —— validation/test 分数，用于复核重标定
- `qwen_lora_runs/v0_summary.json` —— 旧 V0 epoch-3 主表，仅作历史对照
- `qwen_lora_runs/v0_best_epoch1_exploratory_summary.json` —— epoch-1 对照
- `qwen_lora_runs/<run>/history.json` —— 每轮训练/验证损失
- `logs/prm-qwen-main-6235682_*.out` —— 每 25 步的训练损失（累计均值）
- `logs/prm-token-profile-6221609.out` —— token 剖面（p99 = 1277；截断 31/391/30）

## 3. 2026-08-30 审计：为什么要做 V1

### 3.1 唯一的阻断性问题

**6 个 V0 run 的验证损失全部在 epoch 1 最低，epoch 2、3 单调变差。**

| run | ep1 train/val | ep2 train/val | ep3 train/val |
|---|---|---|---|
| pointwise | 0.491 / **0.491** | 0.345 / 0.509 | 0.227 / 0.647 |
| pairwise | 0.352 / **0.346** | 0.166 / 0.385 | 0.053 / 0.638 |
| hybrid 0.1 | 0.525 / **0.525** | 0.361 / 0.538 | 0.229 / 0.714 |
| hybrid 0.3 | 0.594 / **0.586** | 0.394 / 0.601 | 0.239 / 0.779 |
| hybrid 0.5 | 0.664 / **0.657** | 0.426 / 0.674 | 0.251 / 0.884 |
| hybrid 1.0 | 0.838 / **0.827** | 0.510 / 0.856 | 0.284 / 1.139 |

epoch 1 时 train ≈ val，泛化间隙几乎为零；epoch 3 时 pairwise 的训练损失是验证
损失的 1/12。从 slurm 日志还原的 step 级曲线呈**阶梯状**：每轮之内前 30–40% 就
压平，跨轮重看同一批数据时断崖下降——这是记忆，不是学习。

主表用的是 `last.pt`（epoch 3），也就是每个 run 都越过最优点之后的 checkpoint。

### 3.2 学习率调度的次生问题

V0 的调度分母是 3 个 epoch 的总更新数（4488）。因此 epoch-1 checkpoint 取在
**lr = 7.09e-5，仍是峰值的 70.9%**，完全没有退火——它是训练途中的快照，不是收敛
点。这也解释了 epoch 1 内训练损失早早压平在 0.43–0.46：那是高学习率下的噪声
地板。

V1 把 `epochs` 设为 1 后，调度分母变成 1496，末步 lr = 7.1e-8，完整退火。所以
**V1 不是"复现那个 epoch-1 checkpoint"，而是一个训练条件更好的新实验。**

### 3.3 first-error 指标退化

test 集 7158 条有已知错误的轨迹里，7155 条错误就在最后一步。
7155 / 7158 = 99.9581%，与 6 个 run、两种 checkpoint 政策报告的
`first_error_within_1` 一位不差——它是数据常数，与模型无关。V1 用
`report_first_error: false` 把它从评测中移除，只在论文的 Limitations 保留说明。

### 3.4 已核实无误的部分

- 切分无泄漏：train/val/test 的 problem 文本，精确匹配与归一化匹配交集均为 0
- 阈值只在 validation 上选，test 只用一次
- bootstrap 按 problem cluster 重采样（984 个 cluster），不是按 pair 独立重采样
- 梯度累积权重是正确的 node 加权平均
- audit 数字自洽：1,040,685 = 680,279 + 105,542 + 254,864
- token 剖面与报告一致：p99 = 1277，截断 31 / 391 / 30
- 平凡 baseline 很弱（always-negative macro-F1 0.350，长度启发式 pair acc 0.533，
  AUROC 0.485），模型的 0.75 / 0.83 确实来自学习

## 4. V1 预注册（提交 sweep 前冻结）

### 4.1 协议

除 `epochs` 外，全部沿用 V0：

| 项 | 值 |
|---|---|
| Backbone | `Qwen/Qwen2.5-Math-1.5B`，bf16，base 冻结 |
| LoRA | rank 16、alpha 32、dropout 0.05、all-linear、bias none |
| 输入 | problem + recent prefix + candidate，2048 token |
| Batch | 2 nodes × accumulation 16 = 32 nodes |
| 优化器 | AdamW，lr 1e-4，weight decay 0.01，grad clip 1.0 |
| 调度 | 线性 warmup 6%（89 步）→ 线性衰减到 0，共 1496 次更新 |
| **Epochs** | **1**（V0 是 3） |
| Cohort | 47,852 train / 5,751 val nodes，与 V0 同一批 |
| Seeds | **42、7、123** |
| 目标 | pointwise、pairwise、hybrid λ ∈ {0.1, 0.3, 0.5, 1.0} |

共 18 个 run。预算：训练 18 × 2.02h + 诊断验证 18 × 0.59h + 评测 18 × 0.6h
≈ **59 GPU-h**（V0 的三 epoch 版本花掉约 44 GPU-h）。

### 4.2 采用 1 epoch 的依据——只能引用 validation

改成 1 epoch 的决定，**只依据第 3.1 节的验证损失**（6 个 run 的 validation loss
都在 epoch 1 最低）。这个论证是干净的。

epoch-1 的 **test** 结果（job 6268417）在做出这个决定时已被查看，因此它们在论文
中一律标记为 **exploratory / 用于确定协议**，不得作为确认性证据。V1 的 test
结果才是 confirmatory。

### 4.3 Checkpoint 政策

固定的 1-epoch 预算的最后一步（`last.pt`），对所有 objective 与所有 seed 一致。
`eval_every_optimizer_steps: 150` 产生的 9 个诊断验证点**只用于画曲线和确认
1 epoch 尚未过拟合，不参与任何 checkpoint 选择**——否则不同 run 会停在由各自
准则决定的不同步数，重新引入不受控变量。

### 4.4 λ 的处理

V1 评测使用 `selection_rule: report_all`：**报告完整的 λ 响应曲线，不做事后选择。**
论文的主张是关于这条曲线的形状，而不是"我们选出的 hybrid 打败了 baseline"。

若正文确实需要指定单个 λ，规则在此冻结：

> 按 3 个 seed 的 **平均 validation macro-F1** 选择；
> 平均 **validation node-macro pairwise accuracy** 破平。
> 任何 test 指标都不得参与。

### 4.5 明确不做的事

- 不调学习率、不换余弦调度、不改 LoRA rank —— 卖点是"只改监督目标"的受控对比，
  多引入一个维度只会稀释它，而且没有预算做 ablation
- 不加 neutral label（`0`）—— 属于 V1 ordinal 扩展，进 future work
- 不做 4096 context、7B backbone、Best-of-N、label-efficiency 曲线
- 不因为某个 λ 的 test 数字好看而改动 4.4 的规则

### 4.6 已知的口径变化

V0 摘要里"pairwise-only 局部排序最好"这一断言**在 epoch-1 的 validation 上不成立**
（pairwise 的 val pair accuracy 是六者中最低）。V1 结果出来之前，正文不得保留这
句话。V1 完成后按实际数据重写 Results。

## 5. 执行流程

### 5.1 门禁：pilot

```bash
cd "${PRM_PROJECT_ROOT}"
sbatch experiments/unicluster/05c_qwen_v1_pilot.sbatch
```

pilot 会训练一个 λ=0.3 的截断 run 并**评测它**，因此新的诊断验证路径和新的评测
路径（`report_first_error: false` + `save_scores: true`）都在正式 sweep 之前在
H100 上执行过一次。完成后检查：

2026-08-31 的 pilot（6743886）实测：`diagnostics: 2`、`final lr: 0.0`、两个
`strict_node_scores_*.jsonl` 均生成、summary 的 `selected_hybrid_lambda` 为
`null`。`gradient_norm.mean_pre_clip = 12.1`，见下方说明。

```bash
python -c "import json,glob;p=glob.glob('outputs/qwen_lora_v1_runs_pilot/*/history.json')[0];h=json.load(open(p))[0];print('diagnostics:',len(h['diagnostic_validations']),'| final lr:',h['learning_rate'],'| grad_norm mean:',h['gradient_norm']['mean_pre_clip'])"
ls outputs/qwen_lora_v1_runs_pilot/*/strict_node_scores_*.jsonl
```

`gradient_norm.mean_pre_clip` 如果远大于 `max_grad_norm: 1.0`，说明裁剪实际上是
优化器的一部分而不是安全网。V0 也是同样设置，受控对比不受影响，但论文应提一句。

### 5.2 全量 sweep

三个 seed，评测跟在各自训练之后，汇总等全部评测完成：

```bash
eval_jobs=()
for s in 42 7 123; do
    t=$(sbatch --parsable --export=ALL,PRM_SEED=$s \
        experiments/unicluster/06_qwen_v1_train_array.sbatch)
    e=$(sbatch --parsable --dependency=afterok:$t --export=ALL,PRM_SEED=$s \
        experiments/unicluster/07_qwen_v1_eval_array.sbatch)
    echo "seed $s: train=$t eval=$e"
    eval_jobs+=("$e")
done
sbatch --dependency=afterok:$(IFS=:; echo "${eval_jobs[*]}") \
    experiments/unicluster/08_qwen_v1_summarize_cpu.sbatch
```

### 5.3 失败重投

1-epoch 的 run 无法 resume（恢复粒度是整个 epoch），失败的 task 会在 run 目录里
留下配置和 tokenizer，重投会直接 `FileExistsError`。必须显式开启覆盖，且只对失败
的那个 index 开启：

```bash
PRM_OVERWRITE=1 sbatch --array=<失败的index> --export=ALL,PRM_SEED=<seed>,PRM_OVERWRITE=1 \
    experiments/unicluster/06_qwen_v1_train_array.sbatch
```

默认 `PRM_OVERWRITE` 不设置，正常提交永远不会删除已完成的 run。

## 6. 完成后的检查清单

```bash
python -c "import json;d=json.load(open('outputs/qwen_lora_v1_runs/v1_summary.json'));print(len(d['rows']),'rows');print(sorted({(r['mode'],r['lambda_pair'],r['seed']) for r in d['rows']}))"
```

- [ ] `v1_summary.json` 恰好 18 行（`--summarize-only` 用 glob，少一个 run 不会
      报错，会静默给出不完整的汇总）
- [ ] 每个 aggregate 的 `n_runs == 3`、`sample_std` 非 null
- [ ] 所有 run 的 `checkpoint_epoch == 1`、`planned_epochs == 1`
- [ ] `checkpoint_policy_is_uniform == true`
- [ ] `selected_hybrid_lambda == null`（`report_all` 的预期行为）
- [ ] 每个 run 的 `history.json` 有 9 个 `diagnostic_validations`
- [ ] 每个 run 目录有 `strict_node_scores_{val,test}.jsonl`
- [ ] 18 个 run 的 `environment.json` 里 `git_revision` 一致

## 7. 拉回本地与后续分析

```bash
rsync -aP --include='*/' --include='history.json' --include='run.json' \
  --include='metrics.json' --include='environment.json' \
  --include='resolved_config.json' --include='strict_node_scores_*.jsonl' \
  --include='v1_summary.json' --exclude='*' \
  hd_cn362@uc3.scc.kit.edu:/pfs/work9/workspace/scratch/hd_cn362-prm_step_preference/Process_Reward_LLMs/projects/step_preference_prm/outputs/qwen_lora_v1_runs/ \
  ./outputs/qwen_lora_v1_runs/
```

不要拉 checkpoint（每个 run 约 300 MB，18 个约 5.3 GB），分析用不到。

后续分析（都不需要 GPU）：

1. **λ 响应曲线**：x 轴 λ（0、0.1、0.3、0.5、1.0、∞ 断轴），左 y 轴 ECE-10，
   右 y 轴 node-macro pair accuracy，每点带 3-seed 误差棒。这是论文的主图。
2. **Platt / temperature 重标定**：用 `strict_node_scores_val.jsonl` 拟合
   σ(a·r + b)，应用到 test，报告重标定后的 Brier / ECE。判定 pairwise 的校准
   劣势是"尺度未被锚定、一次后处理即可修复"还是真实缺陷。
3. **训练与诊断验证曲线**：确认 1 epoch 内验证损失未回升。
4. **Reliability diagram**：pointwise / hybrid λ=1.0 / pairwise 三条。
