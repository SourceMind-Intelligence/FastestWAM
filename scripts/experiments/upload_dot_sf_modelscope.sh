#!/usr/bin/env bash
# Upload DoT SF MIP-2 final checkpoint to ModelScope (private).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
set -a
source "${ROOT}/.env"
set +a

export PATH="/root/tengfei/envs/openwam/bin:${PATH}"
export MODELSCOPE_API_TOKEN="${MS_TOKEN:?MS_TOKEN missing in .env}"

REPO_ID="${REPO_ID:-TengfeiMa13206/FastestWAM-DoT-SF-MIP2-LIBERO}"
STAGE="${STAGE:-/tmp/ms_upload_dot_sf_85110}"
LOG="${ROOT}/outputs/logs/modelscope_upload_dot_sf.log"
mkdir -p "${ROOT}/outputs/logs"

if [[ ! -d "${STAGE}" ]]; then
  echo "[ERROR] staging dir missing: ${STAGE}" >&2
  exit 1
fi

echo "[$(date -Is)] upload ${STAGE} -> ${REPO_ID}" | tee "${LOG}"
modelscope upload "${REPO_ID}" "${STAGE}" . \
  --repo-type model \
  --token "${MS_TOKEN}" \
  --commit-message "Upload DoT SF MIP-2 LIBERO final checkpoint_step_5100" \
  --max-workers 4 \
  2>&1 | tee -a "${LOG}"
echo "[$(date -Is)] upload exit=$?" | tee -a "${LOG}"
