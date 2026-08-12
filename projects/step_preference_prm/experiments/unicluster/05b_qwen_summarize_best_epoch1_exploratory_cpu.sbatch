#!/bin/bash
# Summarize the isolated epoch-1 best-checkpoint exploratory metrics.
#SBATCH --job-name=prm-qwen-e1-summary
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PRM_PROJECT_ROOT:?Export PRM_PROJECT_ROOT before sbatch}"
: "${PRM_VENV:?Export PRM_VENV before sbatch}"

module purge
module load jupyter/ai
source "${PRM_VENV}/bin/activate"
cd "${PRM_PROJECT_ROOT}"

python scripts/07_eval_all.py \
    --config experiments/qwen_lora_main/eval_best_epoch1_exploratory.yaml \
    --summarize-only
