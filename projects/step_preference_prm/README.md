# Step Preference Process Reward Models

The final course experiment compares pointwise, pairwise-only, and hybrid
losses on one shared, cleaned PRM800K cohort. It uses Qwen2.5-Math-1.5B
with LoRA, seeds 42/7/123, and a fixed one-epoch budget (18 completed runs).

Use [SUBMISSION_README.md](SUBMISSION_README.md) for installation, data
preparation, training, evaluation, and offline result reproduction.
See [AI_USAGE.md](AI_USAGE.md) for the specific AI-assisted components.

## Files

- `src/prm_pref/`: data processing, model, losses, training, and metrics.
- `scripts/`: data preparation, training, evaluation, and recalibration commands.
- `experiments/qwen_lora_v1/`: configurations used for the final results.
- `experiments/unicluster/`: Slurm jobs for bwUniCluster.
- `tests/`: correctness checks.
- `submission/artifacts/`: recorded metrics, histories, environments, and scores.

## Results

Hybrid lambda=1 increases mean same-prefix pair accuracy from 83.72% to
84.27%. Pairwise-only training has lower global AUROC, Macro-F1, and raw
probability calibration than pointwise training. Final-answer accuracy and
label efficiency were not evaluated; the first-error metric was excluded
because it is degenerate on these trajectories.

[RUNBOOK_V1.md](RUNBOOK_V1.md) records the final protocol and completed runs.
[PROJECT_DESIGN_HISTORY.md](PROJECT_DESIGN_HISTORY.md), `RUNBOOK_V0.md`,
and `TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md` preserve earlier planning and
setup notes. Those alternatives are not additional completed experiments.
