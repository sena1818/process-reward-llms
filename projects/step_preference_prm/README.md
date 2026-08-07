# Step Preference Modeling for Process Reward Models

Seminar project proposal for **Process Reward in Large Language Models**.

Primary paper: Lightman et al., 2023, *Let's Verify Step by Step*.

> **当前执行入口（2026-07-30）**：本文件保留项目论证和历史设计。
> 训练代码结构、严格主实验、UniCluster 命令和资源决策请以
> [`TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md`](TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md)
> 为准；`RUNBOOK_V0.md` 仅作为 RoBERTa sanity baseline 的旧说明。

## 1. Project in One Sentence

This project derives **exact-prefix `$+1>-1$` step comparisons** from PRM800K ratings and trains a scalar pairwise / hybrid Process Reward Model (PRM), then compares it against a standard pointwise PRM on process-level and outcome-level evaluations. The neutral label is a separate V1 question, not silently assumed to be an ordinal midpoint in V0.

Short version:

> Instead of using the clean `$+1/-1$` ratings only as separate binary labels, we also derive `$+1>-1$` comparisons between candidates sharing exactly the same reasoning context and train a PRM to rank them.

## 0. V0 Implementation Status

V0 data materialization, node-balanced encoder/Qwen-LoRA reward-model
training, the hybrid lambda sweep, validation-only threshold calibration,
problem-cluster bootstrap evaluation, and epoch-level resume are implemented.
The complete local materialization contains:

```text
pointwise +1/-1 examples: 935,143
exact-prefix +1>-1 pairs: 236,539
strict cleaned exact-prefix nodes: 58,428
known first-error trajectories: 85,316
```

Build and validate the data with:

```bash
python scripts/01_build_splits.py
python scripts/02_build_pairs.py
python scripts/03_build_pointwise.py
python scripts/materialize_training_nodes.py
python scripts/validate_v0_data.py
```

After installing `requirements.txt`, exercise the real tokenizer/model/backward
path with tiny smoke runs:

```bash
python scripts/04_train_pointwise.py --smoke
python scripts/05_train_pairwise.py --smoke
python scripts/06_train_hybrid.py --smoke
```

To exercise the Qwen causal-LoRA path on a laptop, add `--model` and
`--device`; off CUDA the smoke profile drops to fp32 automatically. See section
6.1 of the [training and UniCluster
guide](TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md) for the full local sequence.

See `TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md` for the current experiment
matrix and UniCluster execution flow.

## 2. Motivation

`Let's Verify Step by Step` showed that **process supervision** is more useful than only supervising final answers. Its PRM is mainly trained as a pointwise step classifier:

```text
 question
+ previous reasoning steps
+ current candidate step
-> label: process label (positive / neutral / negative in the original data)
```

This is powerful, but it does not explicitly use the local comparative signal available when alternatives share the exact same context:

```text
under the same context (V0):
  +1 next step > -1 next step
```

The project asks:

> Can we make better or more label-efficient PRMs by adding exact-prefix derived comparisons to step-level labels?

The pair term uses the Bradley--Terry form that also appears in preference-learning papers, but the target is different from DPO:

- DPO / Step-DPO optimizes a **generator / policy**.
- This project optimizes a **verifier / reward model**.

## 3. Scope and Non-Goals

### What This Project Does

- Uses PRM800K step labels from `Let's Verify Step by Step`.
- Audits how many exact-prefix positive/neutral/negative step alternatives exist.
- Constructs step-level preference pairs from human labels.
- Trains pointwise, pairwise, and hybrid PRM variants.
- Evaluates process-level behavior, not only final-answer accuracy.

### What This Project Does Not Do

- It does not reproduce full OpenAI-scale PRM training.
- It does not rely on Monte Carlo rollout labels.
- It does not train a solver with Step-DPO.
- It does not claim to beat the original paper globally.
- It does not use synthetic hard negatives as the main contribution.

The realistic claim is:

> In a small-scale seminar setting, hybrid pointwise + pairwise supervision can improve label efficiency and first-error localization compared with a pointwise PRM baseline.

## 4. Relationship to Other Seminar Papers

| Paper / Idea | What It Does | Difference from This Project |
|---|---|---|
| `Let's Verify Step by Step` | Human step labels, pointwise PRM | This project reuses its labels but changes the training objective to include pairwise / ordinal preferences. |
| Math-Shepherd | Uses Monte Carlo rollout to auto-label steps | This project does not estimate labels from final answers or rollouts. |
| Step-DPO | Optimizes the generator / policy using step preferences | This project trains the verifier / PRM, not the generator. |
| Q-value PRM | Learns rankings based on future value / expected success | This project learns local human step preference from PRM800K ratings, not future return. |
| Lessons of Developing PRMs | Warns that Best-of-N can hide poor process understanding | This project includes step-level and first-error evaluation to avoid that pitfall. |
| RetrievalPRM | Adds retrieved examples at inference time for OOD generalization | This project changes the training objective, not the inference context. |

## 5. Key Terminology

### ORM / Outcome Reward Model

Scores a full solution or final answer:

```text
reward(question, full_solution) -> scalar
```

It is useful but has poor credit assignment for long-chain reasoning.

### PRM / Process Reward Model

Scores a reasoning process step by step:

```text
reward(question, previous_steps, current_step) -> scalar
```

It can identify where reasoning first goes wrong.

### Pointwise PRM

Each step is trained independently:

```text
input:  question + prefix + candidate step
target: label in {-1, 0, +1} or binary correct/incorrect
```

### Pairwise / Preference PRM

Two candidate steps under the same context are compared:

```text
context: question + previous steps
candidate A: rating +1
candidate B: rating -1
target: A > B
```

### Hybrid PRM

Combines pointwise classification and pairwise ranking:

```text
loss = pointwise_loss + lambda * pairwise_loss
```

This is the recommended main method.

## 6. Data

### Primary Dataset

Use **PRM800K**, released with `Let's Verify Step by Step`.

Expected data structure:

```text
problem
solution / trajectory
step tree
candidate completions
rating per candidate completion: -1, 0, +1
```

Important label meanings:

| Rating | Meaning | How to Use |
|---|---|---|
| `+1` | Correct / valid progress | Positive step |
| `0` | Not wrong, but does not make useful progress | Neutral; reserved for a separately stated V1 assumption |
| `-1` | Incorrect / invalid step | Negative step |

V0/V1 use these labels as follows:

```text
V0: +1 > -1 only
V1 hypothesis: +1 > 0 > -1, after semantic validation
```

### Critical First Task: Data Audit

Before training anything, run an audit:

1. Count total problems.
2. Count total annotated steps.
3. Count label distribution: `+1`, `0`, `-1`.
4. Count exact-prefix nodes with multiple candidate completions.
5. Count exact-prefix preference pairs:
   - `+1 > -1`
   - `+1 > 0`
   - `0 > -1`
6. Count problems with at least one first-error point.

Why this matters:

> The whole method is strongest when there are same-prefix alternatives. If PRM800K has fewer exact-prefix pairs than expected, the fallback is to train a hybrid ordinal PRM using all pointwise labels and only use exact-prefix pairs where available.

### Optional Evaluation Data

Use a small MATH or MATH500-style held-out set for Best-of-N reranking if compute allows.

This is optional because the main project should be process-level evaluation.

## 7. Pair Construction

### Recommended Main Pair Type: Exact-Prefix Pairs

This is the cleanest and should be the main method:

```text
same question
same previous steps / prefix
different candidate next steps
different ratings
```

Example:

```text
Context:
  Problem: solve for x.
  Step 1: 3x + 2 = 11

Candidate A:
  3x = 9
  rating = +1

Candidate B:
  3x = 13
  rating = -1

Preference:
  A > B
```

V0/V1 pair policy:

| Pair | Status | Weight |
|---|---:|---:|
| `+1 > -1` | V0 main pair | `1.0` |
| `+1 > 0` | V1 only, after validating ordinal semantics | validation-only hyperparameter |
| `0 > -1` | V1 only, after validating ordinal semantics | validation-only hyperparameter |

### Secondary Pair Type: First-Error Pairs

For a single trajectory:

```text
prefix before first negative step > prefix including first negative step
```

This tests whether the model notices the transition from valid reasoning to invalid reasoning.

Use this as an additional experiment, not the first implementation target.

### Optional Pair Type: Synthetic Hard Negatives

Generate minimally corrupted steps from positive steps.

Examples:

- arithmetic corruption
- sign error
- invalid algebraic transformation
- swapping a variable
- using an unsupported assumption

This is optional and risky because synthetic errors may create distribution shift. It should not be the main contribution.

## 8. Model Architectures

The project should support two model tiers.

### Tier 1: Lightweight Baseline

Use this first. It is cheap and enough to prove the method.

```text
text input -> frozen encoder -> pooled embedding -> MLP scorer
```

Recommended encoders:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`
- `intfloat/e5-small-v2`

Scorer:

```text
MLP(
  Linear(hidden, hidden),
  GELU/ReLU,
  Dropout,
  Linear(hidden, 1 or 3)
)
```

Advantages:

- fast on CPU or a small GPU
- easy to debug
- suitable for data audit and first experiments

Limitations:

- weaker mathematical reasoning
- less faithful to large PRM setting

### Tier 2: Small LM + LoRA

This is now the strict main experiment after the encoder smoke path works.

Recommended backbone:

- `Qwen/Qwen2.5-Math-1.5B` base

Architecture:

```text
question + prefix + candidate step
-> causal LM / transformer
-> pooled final token hidden state
-> scalar reward head
```

Training:

- LoRA on attention / MLP modules
- scalar reward head trained from scratch
- 4-bit or 8-bit quantization if needed

Suggested LoRA settings:

```yaml
r: 8 or 16
alpha: 16 or 32
dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

Use the lightweight encoder only to validate the pipeline, then run the
controlled LoRA comparison with the fixed Qwen base model.

## 9. Input Formatting

Use one stable prompt format for all models.

```text
Problem:
{problem_text}

Previous steps:
{step_1}
{step_2}
...
{step_t_minus_1}

Candidate step:
{candidate_step}
```

For pairwise training:

```text
Context:
Problem:
...

Previous steps:
...

Candidate A:
...

Candidate B:
...

Preference:
A is better than B
```

Implementation recommendation:

- Encode candidates separately with the same scorer:

```text
r_pos = scorer(context, positive_step)
r_neg = scorer(context, negative_step)
```

- Do not make the model read both candidates jointly at first.

This keeps the reward score reusable for inference and first-error localization.

## 10. Loss Functions

The V0 loss design has one scalar scorer $r_\theta(c,s)$ and uses only the
clean `$+1/-1$` subset. The raw `0` label is not silently converted to “half
correct.”

For a node $n$, let $\mathcal I_n$ be its retained candidates, $b_n=|\mathcal
I_n|$, $P_n$ its positive indices, and $R_n$ its negative indices. Training
batches contain only nodes with $|P_n|,|R_n|>0$.

```text
pointwise:  mean_i BCEWithLogits(r_ni, 1[y_ni = +1])
pairwise:   mean_(i,j in P_n x R_n) softplus(-(r_ni - r_nj))
hybrid:     mean_n(pointwise_n) + lambda * mean_n(pairwise_n)
```

The pair term has Bradley--Terry form for one derived comparison. All pairs in
one node are correlated because they come from the same ratings, so the target
objective is a node-balanced weighted composite likelihood, not an independent
joint likelihood. Do not pre-expand pairs in the data loader: forward each
candidate once, then form score differences within the node.

Implementation status: the strict main pipeline now uses
`data/processed/nodes_v0`, one shared node DataLoader, one forward per
candidate, and the same-cohort node-balanced aggregation above. Flat point and
pair artifacts remain only for audit, compatibility, and optional secondary
evaluation. Qwen-LoRA results must still be produced by the UniCluster
smoke/pilot/main sequence before numerical claims are made.

Use

```text
lambda in {0, 0.1, 0.3, 0.5, 1.0}
```

and select it on validation only. `$\lambda=0$` is the pointwise baseline;
pure pairwise is a separate run. There is no pre-declared default such as
`0.3`.

The full derivation, including the Bernoulli point model, the discordant-pair
conditional derivation of Bradley--Terry, the weighted composite likelihood,
and the optional proper ordinal extension for `0`, is in
[`idea/PRM_loss_design.md`](../../idea/PRM_loss_design.md).

## 11. Experiments

### Experiment 0: Data Audit

Required before model training.

Outputs:

```text
data_audit.json
pair_stats.json
label_distribution.png
pair_distribution.png
```

Questions to answer:

- How many exact-prefix pairs are available?
- Are `0` labels common enough to use?
- Is there enough data for train/val/test splits by problem?

### Experiment 1: Pointwise Baseline

Train standard PRM:

```text
input: question + prefix + candidate step
target: rating
loss: BCE / CE / regression
```

Purpose:

> Reproduce the core training style of `Let's Verify Step by Step` in a small-scale setting.

### Experiment 2: Pairwise PRM

Train only with pairwise ranking:

```text
input: same-prefix preference pairs
loss: pairwise ranking
```

Purpose:

> Test whether preferences alone can learn useful step scoring.

Expected limitation:

> Pure pairwise training may rank steps well but have worse score calibration.

### Experiment 3: Hybrid PRM

Train with both pointwise and pairwise losses.

This is the main method.

Purpose:

> Combine calibration from pointwise labels with comparative signal from preference pairs.

### Experiment 4: Label Efficiency

Train pointwise and hybrid models with:

```text
10%, 30%, 50%, 100% of training problems
```

Report:

```text
x-axis: label budget
y-axis: step F1 / pairwise accuracy / first-error localization
```

Hypothesis:

> Hybrid PRM should outperform pointwise PRM more clearly in low-data regimes.

### Experiment 5: Best-of-N Reranking

Optional but useful.

Procedure:

1. Generate or use existing `N` candidate solutions per problem.
2. Score each step with the PRM.
3. Aggregate step scores into a solution score.
4. Pick the best solution.
5. Evaluate final answer accuracy.

Suggested `N`:

```text
N in {4, 8, 16}
```

Aggregation options:

```text
min step score
mean step score
sum step score
product of step probabilities
```

Recommended default:

```text
solution_score = min_t r_t
```

Reason:

> A single bad reasoning step should strongly penalize the whole solution.

## 12. Evaluation Metrics

### Step-Level Classification

Metrics:

- accuracy
- macro F1
- positive/negative F1
- AUROC if using scalar scores

This tests basic step correctness judgment.

### Pairwise Accuracy

For held-out preference pairs:

```text
accuracy = mean[ r(c, s+) > r(c, s-) ]
```

This directly evaluates the proposed training signal.

### First-Error Localization

For each trajectory with a known first negative step:

```text
predicted first error = first step where score < threshold
```

Metrics:

- exact match
- within +/- 1 step
- mean absolute step error
- first-error detection F1

This is the most important process-level metric.

### Best-of-N Accuracy

Use only as an external downstream metric.

Important warning:

> Best-of-N alone is not enough, because a model can select final-answer-correct solutions without truly understanding the reasoning process.

This point connects directly to `Lessons of Developing PRMs`.

## 13. Data Splitting

Split by problem, not by individual step.

Recommended split:

```text
train: 80% problems
val:   10% problems
test:  10% problems
```

Why:

> If steps from the same problem appear in both train and test, the model may memorize problem-specific information.

Keep pair construction inside each split only.

Do not create train pairs from a test problem.

## 14. Suggested Repository Structure

```text
step_preference_prm/
  README.md
  requirements.txt
  configs/
    audit.yaml
    train_pointwise.yaml
    train_pairwise.yaml
    train_hybrid.yaml
    eval.yaml
  data/
    raw/
      prm800k/
    processed/
      audit/
      pointwise/
      pairs/
      splits/
  src/
    prm_pref/
      __init__.py
      data/
        load_prm800k.py
        audit.py
        build_pointwise.py
        build_pairs.py
        split_by_problem.py
      models/
        embedding_scorer.py
        lm_reward_model.py
      training/
        losses.py
        train_pointwise.py
        train_pairwise.py
        train_hybrid.py
      eval/
        eval_step.py
        eval_pairwise.py
        eval_first_error.py
        eval_best_of_n.py
      utils/
        text_format.py
        metrics.py
        seed.py
  scripts/
    00_audit_data.py
    01_build_splits.py
    02_build_pairs.py
    03_train_pointwise.py
    04_train_pairwise.py
    05_train_hybrid.py
    06_eval_all.py
  outputs/
    audit/
    runs/
    figures/
    tables/
  report/
    proposal.md
    slides_outline.md
```

## 15. Implementation Order

### Phase 1: Data Audit

1. Download PRM800K.
2. Inspect raw JSON format.
3. Write loader.
4. Count labels and exact-prefix pair availability.
5. Decide whether exact-prefix pairs are enough.

Exit criterion:

```text
outputs/audit/data_audit.json exists
outputs/audit/pair_stats.json exists
```

### Phase 2: Pointwise Pipeline

1. Build pointwise examples.
2. Split by problem.
3. Train lightweight embedding + MLP scorer.
4. Evaluate step-level metrics.

Exit criterion:

```text
Pointwise baseline produces non-random step-level F1.
```

### Phase 3: Pairwise Pipeline

1. Build same-prefix preference pairs.
2. Train pairwise scorer.
3. Evaluate pairwise accuracy.

Exit criterion:

```text
Pairwise model ranks held-out pairs better than random.
```

### Phase 4: Hybrid Model

1. Train with pointwise + pairwise loss.
2. Tune `lambda_pair`.
3. Compare against pointwise baseline.

Exit criterion:

```text
Hybrid improves pairwise accuracy and/or first-error localization.
```

### Phase 5: Label Efficiency

1. Subsample train problems.
2. Train pointwise and hybrid models at each data budget.
3. Plot learning curves.

Exit criterion:

```text
label_efficiency_curve.png exists
```

### Phase 6: Optional Best-of-N

1. Prepare candidate solutions.
2. Score and rerank.
3. Compare final answer accuracy.

Exit criterion:

```text
Best-of-N table exists, but it is not the only evaluation result.
```

## 16. Main Tables for the Report

### Table 1: Data Audit

| Statistic | Value |
|---|---:|
| Number of problems | |
| Number of annotated steps | |
| `+1` labels | |
| `0` labels | |
| `-1` labels | |
| Exact-prefix nodes with alternatives | |
| `+1 > -1` pairs | |
| `+1 > 0` pairs | |
| `0 > -1` pairs | |

### Table 2: Main Process-Level Results

| Model | Step F1 | Pairwise Acc | First Error Exact | First Error +/-1 |
|---|---:|---:|---:|---:|
| Pointwise PRM | | | | |
| Pairwise PRM | | | | |
| Hybrid PRM | | | | |

### Table 3: Label Efficiency

| Data Budget | Pointwise F1 | Hybrid F1 | Pointwise First Error | Hybrid First Error |
|---:|---:|---:|---:|---:|
| 10% | | | | |
| 30% | | | | |
| 50% | | | | |
| 100% | | | | |

### Table 4: Optional Best-of-N

| Model | N=4 | N=8 | N=16 |
|---|---:|---:|---:|
| Pointwise PRM | | | |
| Hybrid PRM | | | |

## 17. Expected Findings

The ideal result:

```text
Hybrid PRM > Pointwise PRM on pairwise accuracy
Hybrid PRM > Pointwise PRM on first-error localization
Hybrid PRM especially helps in low-data settings
Best-of-N may improve modestly, but it is not the main claim
```

Acceptable negative result:

```text
Pairwise loss improves pairwise ranking but not final Best-of-N.
```

This is still meaningful because it supports the `Lessons` argument that process-level evaluation and outcome-level evaluation are different.

## 18. Fallback Plans

### If Exact-Prefix Pairs Are Too Few

Fallback:

- Use all pointwise labels for CE/regression.
- Use only available exact-prefix pairs as auxiliary loss.
- Emphasize label audit as a finding.

Revised claim:

> Exact-prefix preferences are sparse in PRM800K, but even limited pairwise supervision can be used as an auxiliary signal.

### If Pairwise-Only Performs Poorly

Fallback:

- Make hybrid the main method.
- Explain that pairwise ranking lacks calibration.
- Use this to motivate combining pointwise and pairwise losses.

### If Best-of-N Does Not Improve

Fallback:

- Do not hide it.
- Emphasize step-level and first-error results.
- Connect to `Lessons of Developing PRMs`: outcome-level metrics do not fully reflect process understanding.

### If LoRA Training Is Too Expensive

Fallback:

- Use frozen embedding + MLP.
- Present the LM + LoRA version as future work.

## 19. Presentation Narrative

Suggested 10-minute structure:

1. **Background**
   - ORM vs PRM
   - `Let's Verify Step by Step`
   - why process supervision matters

2. **Gap**
   - original PRM training is mostly pointwise
   - step labels contain ordinal / preference structure

3. **Method**
   - convert `+1, 0, -1` into preferences
   - pairwise ranking loss
   - hybrid pointwise + pairwise loss

4. **Experiments**
   - data audit
   - pointwise vs pairwise vs hybrid
   - label efficiency
   - first-error localization

5. **Relationship to Other Papers**
   - not Math-Shepherd: no rollout
   - not Step-DPO: trains verifier, not generator
   - not Q-value PRM: local human preference, not future value

6. **Conclusion**
   - pairwise process supervision is a lightweight extension of PRM training
   - process-level evaluation is necessary

## 20. Current Minimal Commands

```bash
# 0. Audit data
python scripts/00_audit_data.py --config configs/audit.yaml

# 1. Build splits and pairs
python scripts/01_build_splits.py
python scripts/02_build_pairs.py
python scripts/03_build_pointwise.py
python scripts/materialize_training_nodes.py

# 2. Train encoder sanity baselines
python scripts/04_train_pointwise.py --smoke
python scripts/05_train_pairwise.py --smoke
python scripts/06_train_hybrid.py --smoke

# 3. Evaluate
python scripts/07_eval_all.py --config configs/eval.yaml
```

## 21. Final Claim to Aim For

The final seminar claim should be modest and precise:

> We extend the process-supervised PRM setup from `Let's Verify Step by Step` by converting step-level human labels into ordinal preferences. In a small-scale reproduction, a hybrid pointwise + pairwise PRM improves process-level ranking and first-error localization, especially under limited label budgets.

Do not claim:

```text
We solve automatic PRM labeling.
We outperform OpenAI's full PRM.
We replace process supervision.
We train a better generator than Step-DPO.
```

The project is strongest when framed as:

```text
same data source,
cleaner supervision structure,
better process-level evaluation.
```
