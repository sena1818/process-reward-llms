# Process Reward in Large Language Models

Seminar research repository for exact-prefix step preference modeling with
PRM800K.

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
- [Training and UniCluster guide](projects/step_preference_prm/TRAINING_AUDIT_AND_UNICLUSTER_GUIDE.md)

Raw/processed data, environments, model caches, logs, and checkpoints are not
tracked. PRM800K can be downloaded reproducibly with:

```bash
cd projects/step_preference_prm
python scripts/download_prm800k.py --all
```
