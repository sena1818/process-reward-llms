# Process Reward in Large Language Models

Seminar research repository for exact-prefix step preference modeling with
PRM800K.

## Submission and reproduction

The final experiment uses Qwen2.5-Math-1.5B with LoRA, one shared strict
58,428-node cohort, and one epoch for each of six objectives and three seeds
(42, 7, 123). All 18 runs are complete.

Start with the [submission and reproduction guide](projects/step_preference_prm/SUBMISSION_README.md)
for setup, final V1 commands, and archived results. See
[AI usage](projects/step_preference_prm/AI_USAGE.md) for assistance disclosure.
The report and team contribution statement are submitted separately.

Hybrid training modestly improves mean same-prefix pair accuracy (83.72%
to 84.27% at lambda=1). Pairwise-only training has lower global AUROC and
Macro-F1. Final-answer accuracy and label efficiency were not evaluated;
the first-error metric was excluded because it is degenerate on these trajectories.

## Repository structure

- [`idea/`](idea/): project specification and loss derivation.
- [`notes/`](notes/): research notes and seminar materials.
- [`lecture/`](lecture/): source lecture material.
- [`projects/step_preference_prm/`](projects/step_preference_prm/): data
  pipeline, reward-model training, evaluation, tests, and UniCluster jobs.

## Training project

The strict experiment compares pointwise, pairwise, and hybrid objectives on
the same cleaned exact-prefix node cohort. The main model is
`Qwen/Qwen2.5-Math-1.5B` with bf16 LoRA and a scalar reward head.

Start with:

- [Project README](projects/step_preference_prm/README.md)
- [Final V1 experiment record](projects/step_preference_prm/RUNBOOK_V1.md)
- [Historical training and UniCluster guide](projects/step_preference_prm/TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md)

Raw/processed data, environments, model caches, logs, and checkpoints are not
tracked. PRM800K can be downloaded reproducibly with:

```bash
cd projects/step_preference_prm
python scripts/download_prm800k.py --all
```
