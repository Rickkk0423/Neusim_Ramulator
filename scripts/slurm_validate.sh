#!/usr/bin/env bash
#SBATCH --job-name=neusim-validate
#SBATCH --partition=dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
"$PYTHON" -m pytest -q
"$PYTHON" -m neusim_ramulator.stream_cli examples/pi05_tiny.json -o "results/pi05-tiny-${SLURM_JOB_ID}"
"$PYTHON" -m neusim_ramulator.stream_cli examples/pi05_pilot.json --describe-only
