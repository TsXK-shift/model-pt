#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/TsXK-shift/model-pt.git}"
REPO_REF="${REPO_REF:-main}"
SRC_DIR="${SRC_DIR:-/kaggle/working/model-pt-src}"
PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/mss}"
MUSDB_DIR="${MUSDB_DIR:-/tmp/musdb18hq}"
MUSDB_URL="${MUSDB_URL:-https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1}"

echo "Cloning project from: ${REPO_URL}"
rm -rf "${SRC_DIR}"
git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${SRC_DIR}"

if [ ! -d "${SRC_DIR}/mss" ]; then
  echo "ERROR: ${SRC_DIR}/mss not found. Your repo must contain the mss folder."
  find "${SRC_DIR}" -maxdepth 3 -type d | sort
  exit 1
fi

echo "Copying project to ${PROJECT_DIR}"
rm -rf "${PROJECT_DIR}"
cp -a "${SRC_DIR}/mss" "${PROJECT_DIR}"

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

echo "Dataset check:"
find "${MUSDB_DIR}" -maxdepth 3 -type f -name "mixture.wav" | head -10

echo "Project check:"
ls -lah "${PROJECT_DIR}"
echo "Config:"
sed -n '1,140p' "${PROJECT_DIR}/configs/kaggle_t4x2.yaml"

echo "Setup done."
