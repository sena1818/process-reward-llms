#!/bin/bash
#SBATCH --job-name=prm-qwen-pilot
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
    --config experiments/qwen_lora_main/train_hybrid.yaml \
    --lambda-pair 0.3 \
    --pilot
