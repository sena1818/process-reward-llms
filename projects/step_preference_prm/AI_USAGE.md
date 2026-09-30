# AI Assistance Disclosure

AI tools were used for brainstorming, code review, experiment scripting,
and assistance diagnosing and fixing execution issues. This document identifies
the confirmed affected components. Human team members remain responsible
for checking the code, running the experiments, and verifying the results.

| Files or components | Assistance |
|---|---|
| `experiments/unicluster/` (including `common.sh` and the V1 pilot, train, evaluation and summary jobs) | AI-assisted drafting and revision of experiment execution and Slurm scripts, and troubleshooting job execution. |
| `experiments/qwen_lora_v1/*.yaml` | AI assistance preparing and revising the final experiment configurations. |
| `src/prm_pref/training/runner.py` | AI-assisted revisions and fixes to training execution. |
| `src/prm_pref/models/encoder_reward_model.py` | AI-assisted revisions and fixes to reward-model execution. |
| `src/prm_pref/data/input_packing.py` | AI-assisted revisions and fixes to input processing. |
| `scripts/recalibrate_scores.py`, `tests/test_recalibration.py` | Created with an AI coding agent during submission preparation to reproduce the validation-fitted Platt analysis. Checked against the saved validation/test score dumps and an analytical probability example. |
| Root/project README, `SUBMISSION_README.md`, and submission-status updates in `RUNBOOK_V1.md` | AI-assisted documentation updates based on the completed runs and saved results. |

This list records confirmed assistance; it does not assert that every other
file was written without AI. Additional source files that were created or
modified with AI assistance should be included if identified during the team's
final review. Report typesetting and language assistance are disclosed in the
separately submitted report.
