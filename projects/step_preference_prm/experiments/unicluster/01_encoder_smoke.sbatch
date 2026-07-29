#!/bin/bash
#SBATCH --job-name=prm-enc-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source "${PRM_PROJECT_ROOT:?}/experiments/unicluster/common.sh"

python scripts/04_train_pointwise.py --smoke
python scripts/05_train_pairwise.py --smoke
python scripts/06_train_hybrid.py --smoke
