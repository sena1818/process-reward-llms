#!/bin/bash
#SBATCH --job-name=prm-qwen-main
#SBATCH --partition=gpu_h100
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source "${PRM_PROJECT_ROOT:?}/experiments/unicluster/common.sh"

resume_flag=()
if [[ "${PRM_RESUME:-0}" == "1" ]]; then
    resume_flag=(--resume)
fi
seed_flag=()
if [[ -n "${PRM_SEED:-}" ]]; then
    seed_flag=(--seed "${PRM_SEED}")
fi
budget_flag=()
if [[ -n "${PRM_MAX_TRAIN_NODES:-}" ]]; then
    budget_flag=(--max-train-nodes "${PRM_MAX_TRAIN_NODES}")
fi

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        python scripts/04_train_pointwise.py \
            --config experiments/qwen_lora_main/train_pointwise.yaml \
            "${seed_flag[@]}" \
            "${budget_flag[@]}" \
            "${resume_flag[@]}"
        ;;
    1)
        python scripts/05_train_pairwise.py \
            --config experiments/qwen_lora_main/train_pairwise.yaml \
            "${seed_flag[@]}" \
            "${budget_flag[@]}" \
            "${resume_flag[@]}"
        ;;
    2|3|4|5)
        lambda_values=(0.1 0.3 0.5 1.0)
        lambda_index=$((SLURM_ARRAY_TASK_ID - 2))
        python scripts/06_train_hybrid.py \
            --config experiments/qwen_lora_main/train_hybrid.yaml \
            --lambda-pair "${lambda_values[$lambda_index]}" \
            "${seed_flag[@]}" \
            "${budget_flag[@]}" \
            "${resume_flag[@]}"
        ;;
    *)
        echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac
