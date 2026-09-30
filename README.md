# Process Reward in Large Language Models

Course project comparing pointwise, same-prefix pairwise, and hybrid process
reward models on PRM800K.

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

## Project files

The submitted implementation is in [`projects/step_preference_prm/`](projects/step_preference_prm/):
`src/` contains the model and pipeline, `scripts/` the command-line entry
points, `experiments/qwen_lora_v1/` the final settings, `tests/` the checks,
and `submission/artifacts/` the archived results.

Earlier proposals, research notes, and slides remain in `idea/`, `notes/`,
and `lecture/` as optional historical material. They are not required to run
or assess the final code.
