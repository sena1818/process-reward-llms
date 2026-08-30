#!/bin/bash
# Evaluate one completed V1 training array, six objectives, one seed.
#SBATCH --job-name=prm-qwen-v1-eval
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
: "${PRM_SEED:?Set PRM_SEED to the matching completed training seed}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) run_name="qwen_lora_pointwise_v1_seed${PRM_SEED}" ;;
    1) run_name="qwen_lora_pairwise_v1_seed${PRM_SEED}" ;;
    2|3|4|5)
        lambda_values=(0.1 0.3 0.5 1.0)
        lambda_index=$((SLURM_ARRAY_TASK_ID - 2))
        run_name="qwen_lora_hybrid_v1_lam${lambda_values[$lambda_index]}_seed${PRM_SEED}"
        ;;
    *)
        echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

python scripts/07_eval_all.py \
    --config experiments/qwen_lora_v1/eval.yaml \
    --run "${run_name}" \
    --metrics-only
