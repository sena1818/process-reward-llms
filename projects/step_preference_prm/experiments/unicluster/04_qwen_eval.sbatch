#!/bin/bash
#SBATCH --job-name=prm-qwen-eval
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

source "${PRM_PROJECT_ROOT:?}/experiments/unicluster/common.sh"

if [[ -n "${PRM_DATA_ROOT:-}" ]]; then
    rsync -a "${PRM_PROJECT_ROOT}/data/processed/trajectories/" \
        "${PRM_DATA_ROOT}/trajectories/"
    # Only needed when evaluation.report_full_pointwise is true.  The
    # evaluator falls back to the workspace copy if this is absent.
    if [[ -d "${PRM_PROJECT_ROOT}/data/processed/pointwise" ]]; then
        rsync -a "${PRM_PROJECT_ROOT}/data/processed/pointwise/" \
            "${PRM_DATA_ROOT}/pointwise/"
    fi
fi

suffix=""
if [[ -n "${PRM_SEED:-}" ]]; then
    suffix+="_seed${PRM_SEED}"
fi
if [[ -n "${PRM_MAX_TRAIN_NODES:-}" ]]; then
    suffix+="_n${PRM_MAX_TRAIN_NODES}"
fi

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        run_name="qwen_lora_pointwise_v0${suffix}"
        ;;
    1)
        run_name="qwen_lora_pairwise_v0${suffix}"
        ;;
    2|3|4|5)
        lambda_values=(0.1 0.3 0.5 1.0)
        lambda_index=$((SLURM_ARRAY_TASK_ID - 2))
        run_name="qwen_lora_hybrid_v0_lam${lambda_values[$lambda_index]}${suffix}"
        ;;
    *)
        echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

python scripts/07_eval_all.py \
    --config experiments/qwen_lora_main/eval.yaml \
    --run "${run_name}" \
    --metrics-only
