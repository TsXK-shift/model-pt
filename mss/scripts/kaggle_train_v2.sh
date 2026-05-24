#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/mss}"
MUSDB_DIR="${MUSDB_DIR:-/tmp/musdb18hq}"
OUT_DIR="${OUT_DIR:-/kaggle/working/runs/tiny-hybrid-v2}"
LOG_DIR="${LOG_DIR:-/kaggle/working/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_v2.log}"
NPROC="${NPROC:-2}"

if [ ! -f "${PROJECT_DIR}/scripts/train_watch.py" ]; then
  echo "ERROR: train_watch.py not found in ${PROJECT_DIR}. Run kaggle_setup_v2.sh first."
  exit 1
fi

if [ ! -d "${MUSDB_DIR}/train" ] || [ ! -d "${MUSDB_DIR}/test" ]; then
  echo "ERROR: MUSDB18-HQ not found in ${MUSDB_DIR}. Run kaggle_setup_v2.sh first."
  exit 1
fi

mkdir -p "${OUT_DIR}/checkpoints" "${LOG_DIR}"
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Starting TinyHybridMSS v2 from zero"
echo "Project: ${PROJECT_DIR}"
echo "Data:    ${MUSDB_DIR}"
echo "Output:  ${OUT_DIR}"
echo "Log:     ${LOG_FILE}"

torchrun --nproc_per_node="${NPROC}" scripts/train_watch.py \
  --config configs/kaggle_t4x2.yaml \
  --data "${MUSDB_DIR}" \
  --out "${OUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
