#!/bin/bash
# Confirmatory V1: one fixed epoch, six objectives, one seed per submission.
#SBATCH --job-name=prm-qwen-v1
#SBATCH --partition=gpu_h100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source "${PRM_PROJECT_ROOT:?}/experiments/unicluster/common.sh"
: "${PRM_SEED:?Set PRM_SEED to one of the pre-registered seeds}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        python scripts/04_train_pointwise.py \
            --config experiments/qwen_lora_v1/train_pointwise.yaml \
            --seed "${PRM_SEED}"
        ;;
    1)
        python scripts/05_train_pairwise.py \
            --config experiments/qwen_lora_v1/train_pairwise.yaml \
            --seed "${PRM_SEED}"
        ;;
    2|3|4|5)
        lambda_values=(0.1 0.3 0.5 1.0)
        lambda_index=$((SLURM_ARRAY_TASK_ID - 2))
        python scripts/06_train_hybrid.py \
            --config experiments/qwen_lora_v1/train_hybrid.yaml \
            --lambda-pair "${lambda_values[$lambda_index]}" \
            --seed "${PRM_SEED}"
        ;;
    *)
        echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac
