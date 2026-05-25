#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/kaggle/working/runs/tiny-mask-v3}"
PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/mss}"
LOG_FILE="${LOG_FILE:-/kaggle/working/logs/train_v3.log}"
EXPORT_DIR="${EXPORT_DIR:-/kaggle/working/export_checkpoint-v3}"
ZIP_PATH="${ZIP_PATH:-/kaggle/working/tiny-mask-checkpoint-v3.zip}"

rm -rf "${EXPORT_DIR}"
rm -f "${ZIP_PATH}"
mkdir -p "${EXPORT_DIR}"

cp "${RUN_DIR}/checkpoints/last.pt" "${EXPORT_DIR}/last.pt"
cp "${RUN_DIR}/checkpoints/best.pt" "${EXPORT_DIR}/best.pt"
cp "${RUN_DIR}/resolved_config.json" "${EXPORT_DIR}/resolved_config.json"
cp "${PROJECT_DIR}/configs/kaggle_t4x2.yaml" "${EXPORT_DIR}/kaggle_t4x2.yaml"
cp "${LOG_FILE}" "${EXPORT_DIR}/train_v3.log" || true

cd /kaggle/working
zip -r "$(basename "${ZIP_PATH}")" "$(basename "${EXPORT_DIR}")"
ls -lh "${ZIP_PATH}"
