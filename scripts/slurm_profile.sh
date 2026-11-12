#!/usr/bin/env bash
#SBATCH --job-name=neusim-profile
#SBATCH --partition=dev
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
component=prefix
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 1 ]]; then component=dit; fi
"$PWD/.venv/bin/python" -m scripts.profile_pi05 --component "$component" \
    --output "results/profile-${SLURM_JOB_ID}-$component" --cycles 200000
