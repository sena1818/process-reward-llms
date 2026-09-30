# Submission and Reproduction

This is the source for **Pointwise and Same-Prefix Pairwise Training for
Mathematical Process Reward Models**. The final V1 experiment uses
Qwen2.5-Math-1.5B with LoRA, six objectives, seeds 42/7/123, and one fixed
epoch. All 18 runs completed on bwUniCluster H100 GPUs. The report and
team contribution statement are submitted separately.

## Setup and data

Run these commands from `projects/step_preference_prm/` after cloning.
Python 3.11 is recommended. For GPU training, install a CUDA-enabled PyTorch
build appropriate for your GPU before installing the remaining requirements.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest tests -q

python scripts/download_prm800k.py --all
python scripts/01_build_splits.py
python scripts/02_build_pairs.py
python scripts/03_build_pointwise.py
python scripts/materialize_training_nodes.py
python scripts/validate_v0_data.py
```

The download script verifies SHA-256 checksums. Raw/processed data, model
caches, and checkpoints are generated locally and are not in Git. V1 reuses
the cleaned `nodes_v0` cohort: 47,852 train, 5,751 validation, and 4,825 test
nodes. This data directory name describes the cleaning version.

Recorded training environment: Python 3.11.11, PyTorch 2.10.0+cu128,
Transformers 4.57.6, PEFT 0.20.0, and one NVIDIA H100 per run. Each recorded
epoch took about 2.6 hours, including diagnostic validation. Full reproduction
requires GPU resources and downloading the base model; offline checks below
require neither.

## Final V1 training and evaluation

On a suitable CUDA GPU, run all objectives for each seed:

```bash
for seed in 42 7 123; do
    python scripts/04_train_pointwise.py --config experiments/qwen_lora_v1/train_pointwise.yaml --seed "$seed"
    python scripts/05_train_pairwise.py --config experiments/qwen_lora_v1/train_pairwise.yaml --seed "$seed"
    for weight in 0.1 0.3 0.5 1.0; do
        python scripts/06_train_hybrid.py --config experiments/qwen_lora_v1/train_hybrid.yaml --seed "$seed" --lambda-pair "$weight"
    done
done
python scripts/07_eval_all.py --config experiments/qwen_lora_v1/eval.yaml
python scripts/recalibrate_scores.py --runs-dir outputs/qwen_lora_v1_runs --output outputs/qwen_lora_v1_runs/platt_recalibration.json
```

V1 evaluates `last.pt` after one epoch for every run. Classification thresholds
use validation only; all hybrid weights are reported without selecting from
test results. Default `configs/train_*.yaml` files are historical encoder
sanity settings, not the final experiment.

For bwUniCluster, use the pilot, train, evaluation and summary jobs in
`experiments/unicluster/05c_*` through `08_*`; environment variables and job
dependencies are in [RUNBOOK_V1.md](RUNBOOK_V1.md). These Slurm scripts use
bwUniCluster modules/partitions. The Python commands above are the portable path.

## Small smoke check

After data preparation, a short CPU check exercises causal-LoRA training
with a smaller model:

```bash
python scripts/06_train_hybrid.py --config experiments/qwen_lora_v1/train_hybrid.yaml --smoke --model Qwen/Qwen2.5-0.5B --device cpu --lambda-pair 0.3
```

This downloads a smaller model, uses fp32 and shorter inputs, and performs
five local steps. It checks execution, not the report's performance.

## Archived results and offline reproduction

- [v1_summary.json](submission/artifacts/v1_summary.json): all 18 runs and
  the three-seed means/sample standard deviations used in the report.
- [metrics/](submission/artifacts/metrics/): original per-run evaluation
  JSON, validation-selected thresholds, and problem-cluster bootstrap intervals.
- [run_history.json](submission/artifacts/run_history.json): original
  epoch and diagnostic-validation records for all runs.
- [run_environments.json](submission/artifacts/run_environments.json):
  original environment records and final training job IDs; all use revision
  `0353318ccd3136a616979ba4ac2ff2bd62a39d38`.
- [v1_scores.zip](submission/artifacts/v1_scores.zip): the 36 original
  validation/test score dumps, compressed without changing their contents.
- [platt_recalibration.json](submission/artifacts/platt_recalibration.json):
  independently recomputed calibration parameters and aggregated results.

Reproduce Platt results from the bundled scores with standard-library Python,
without installing training dependencies:

```bash
python3 scripts/recalibrate_scores.py --output /tmp/prm_platt_recalibration.json
```

The script fits `sigmoid(a * score + b)` by candidate-level binary cross-entropy
on each run's validation scores, then evaluates test. All fitted slopes are
positive; mean pairwise-only ECE falls from 11.50% to 2.49% while AUROC is
preserved by the monotone map.

Hybrid lambda=1 improves mean node-macro pair accuracy from 83.72% to
84.27%, with positive differences for all three seeds. Pairwise-only training
has lower pooled AUROC and Macro-F1. Final-answer accuracy and label efficiency
were not evaluated. First-error localisation was excluded because annotation
usually stops at the error.

## Submission notes

The repository is private. Give the grader access when submitting a link,
or submit a ZIP of the final source. Include the separately finalised ACL
report, member names/matriculation numbers and contribution statement.
See [AI_USAGE.md](AI_USAGE.md) for AI assistance. Older proposals and runbooks
are retained as history; this guide and `experiments/qwen_lora_v1/` define
the final reproduction commands.
