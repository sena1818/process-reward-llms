#!/bin/bash
#SBATCH --job-name=prm-data-v0
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PRM_PROJECT_ROOT:?Export PRM_PROJECT_ROOT before sbatch}"
: "${PRM_VENV:?Export PRM_VENV before sbatch}"

module purge
module load jupyter/ai
source "${PRM_VENV}/bin/activate"
cd "${PRM_PROJECT_ROOT}"

python scripts/00_audit_data.py
python scripts/01_build_splits.py
python scripts/02_build_pairs.py
python scripts/03_build_pointwise.py
python scripts/materialize_training_nodes.py
python scripts/validate_v0_data.py
