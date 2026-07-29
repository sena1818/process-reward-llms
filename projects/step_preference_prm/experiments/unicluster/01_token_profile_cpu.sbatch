#!/bin/bash
#SBATCH --job-name=prm-token-profile
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PRM_PROJECT_ROOT:?Export PRM_PROJECT_ROOT before sbatch}"
: "${PRM_VENV:?Export PRM_VENV before sbatch}"
: "${PRM_HF_HOME:?Export PRM_HF_HOME before sbatch}"

module purge
module load jupyter/ai
source "${PRM_VENV}/bin/activate"
cd "${PRM_PROJECT_ROOT}"
export HF_HOME="${PRM_HF_HOME}"
export TOKENIZERS_PARALLELISM=false

python scripts/profile_token_lengths.py \
    --config experiments/qwen_lora_main/train_hybrid.yaml
