#!/bin/bash
#SBATCH --job-name=prm-enc-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "${PRM_PROJECT_ROOT:?}/experiments/unicluster/common.sh"

overwrite_flag=()
if [[ "${PRM_OVERWRITE:-0}" == "1" ]]; then
    overwrite_flag=(--overwrite)
fi

python scripts/04_train_pointwise.py --smoke "${overwrite_flag[@]}"
python scripts/05_train_pairwise.py --smoke "${overwrite_flag[@]}"
python scripts/06_train_hybrid.py --smoke "${overwrite_flag[@]}"
