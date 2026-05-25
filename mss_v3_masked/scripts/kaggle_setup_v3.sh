#!/usr/bin/env bash
set -euo pipefail

MSS_ZIP_URL="${MSS_ZIP_URL:-https://raw.githubusercontent.com/TsXK-shift/model-pt/main/tiny-mask-mss-v3-clean.zip}"
PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/mss}"
ZIP_PATH="${ZIP_PATH:-/kaggle/working/tiny-mask-mss-v3-clean.zip}"
MUSDB_DIR="${MUSDB_DIR:-/tmp/musdb18hq}"
MUSDB_URL="${MUSDB_URL:-https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1}"

echo "Downloading MSS v3 project:"
echo "${MSS_ZIP_URL}"
rm -rf "${PROJECT_DIR}"
rm -f "${ZIP_PATH}"
wget -c "${MSS_ZIP_URL}" -O "${ZIP_PATH}"
unzip -q "${ZIP_PATH}" -d /kaggle/working

if [ ! -f "${PROJECT_DIR}/configs/kaggle_t4x2.yaml" ]; then
  echo "ERROR: ${PROJECT_DIR}/configs/kaggle_t4x2.yaml not found after unzip."
  find /kaggle/working -maxdepth 3 -type f | sort | head -80
  exit 1
fi

echo "Installing Kaggle requirements"
python -m pip install -q -r "${PROJECT_DIR}/requirements-kaggle.txt"

echo "Preparing MUSDB18-HQ in ${MUSDB_DIR}"
if [ ! -d "${MUSDB_DIR}/train" ] || [ ! -d "${MUSDB_DIR}/test" ]; then
  rm -rf "${MUSDB_DIR}"
  mkdir -p "${MUSDB_DIR}"
  cd "${MUSDB_DIR}"
  wget -c "${MUSDB_URL}" -O musdb18hq.zip
  unzip -q musdb18hq.zip
  rm -f musdb18hq.zip
fi

echo "Project:"
ls -lah "${PROJECT_DIR}"

echo "Dataset check:"
python - <<'PY'
from pathlib import Path
root = Path("/tmp/musdb18hq")
mixtures = list(root.glob("*/*/mixture.wav"))
print("train exists:", (root / "train").exists())
print("test exists:", (root / "test").exists())
print("mixtures:", len(mixtures))
print("first:", mixtures[0] if mixtures else "NONE")
PY

echo "Model check:"
python "${PROJECT_DIR}/scripts/inspect_model.py" --config "${PROJECT_DIR}/configs/kaggle_t4x2.yaml" --seconds 1

echo "Setup v3 done."
