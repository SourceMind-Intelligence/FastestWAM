#!/usr/bin/env bash
# Sequential official asset download for tonight's LIBERO baseline.
# Secrets come from the repo .env if present; they are not echoed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
PY="${PYTHON:-/opt/conda/envs/fastwam/bin/python}"

mkdir -p outputs/logs
echo "[$(date -Is)] LIBERO download start" | tee outputs/logs/download_libero.log
"$PY" scripts/download_assets/download_benchmark_data.py --name LIBERO --yes \
    2>&1 | tee -a outputs/logs/download_libero.log
echo "[$(date -Is)] LIBERO download done" | tee -a outputs/logs/download_libero.log

echo "[$(date -Is)] foundation checkpoint download start" | tee outputs/logs/download_foundation.log
"$PY" scripts/download_assets/download_openwam_checkpoints.py \
    --family alpha --name OpenWAM-Alpha-Pretrain-Foundation-Model --yes \
    2>&1 | tee -a outputs/logs/download_foundation.log
echo "[$(date -Is)] foundation checkpoint download done" | tee -a outputs/logs/download_foundation.log
