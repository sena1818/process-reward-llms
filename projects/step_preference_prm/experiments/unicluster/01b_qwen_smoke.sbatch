#!/bin/bash
# First real execution of the Qwen causal-LoRA path on cluster hardware.
# It runs at smoke scale (5 steps, length 128) but with the production
# backbone, dtype, and autocast, so a dtype, PEFT, or tokenizer problem
# surfaces in the development queue instead of inside the 4 h pilot.
#SBATCH --job-name=prm-qwen-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "${PRM_PROJECT_ROOT:?}/experiments/unicluster/common.sh"

overwrite_flag=()
if [[ "${PRM_OVERWRITE:-0}" == "1" ]]; then
    overwrite_flag=(--overwrite)
fi

python scripts/04_train_pointwise.py \
    --config experiments/qwen_lora_main/train_pointwise.yaml \
    --smoke "${overwrite_flag[@]}"

python scripts/05_train_pairwise.py \
    --config experiments/qwen_lora_main/train_pairwise.yaml \
    --smoke "${overwrite_flag[@]}"

python scripts/06_train_hybrid.py \
    --config experiments/qwen_lora_main/train_hybrid.yaml \
    --lambda-pair 0.3 \
    --smoke "${overwrite_flag[@]}"

python - <<'PY'
import json
import pathlib

# Fail loudly if the smoke produced no usable checkpoint metadata.
root = pathlib.Path("outputs/qwen_lora_runs_smoke")
runs = sorted(path.parent for path in root.glob("*/last.pt"))
if len(runs) != 3:
    raise SystemExit(f"Expected 3 smoke runs with last.pt, found {len(runs)}")
for run in runs:
    history = json.loads((run / "history.json").read_text())
    losses = history[-1]["validation"]
    if not all(value == value for value in losses.values()):
        raise SystemExit(f"Non-finite validation loss in {run.name}")
    print(run.name, losses)
print("Qwen smoke OK")
PY
