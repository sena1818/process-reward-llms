#!/bin/bash
# V1 implementation gate: real H100 path before the 18-run confirmatory sweep.
#SBATCH --job-name=prm-qwen-v1-pilot
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "${PRM_PROJECT_ROOT:?}/experiments/unicluster/common.sh"

python scripts/06_train_hybrid.py \
    --config experiments/qwen_lora_v1/train_hybrid.yaml \
    --lambda-pair 0.3 \
    --pilot \
    ${PRM_OVERWRITE:+--overwrite}

# The confirmatory evaluation drops the first-error benchmark and persists
# scores, so the pilot must reach that path too; otherwise its first cluster
# execution would be the 18-run sweep.
python scripts/07_eval_all.py \
    --config experiments/qwen_lora_v1/eval_pilot.yaml \
    --run qwen_lora_hybrid_v1_lam0.3_pilot
