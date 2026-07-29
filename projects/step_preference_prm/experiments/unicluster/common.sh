#!/bin/bash

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

python - <<'PY'
import torch
import transformers
import peft

print(
    {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpus": torch.cuda.device_count(),
    }
)
PY

if [[ -n "${TMPDIR:-}" ]]; then
    prm_staged_root="${TMPDIR}/prm-data"
    mkdir -p "${prm_staged_root}"
    rsync -a "${PRM_PROJECT_ROOT}/data/processed/nodes_v0/" \
        "${prm_staged_root}/nodes_v0/"
    export PRM_DATA_ROOT="${prm_staged_root}"
fi

nvidia-smi
