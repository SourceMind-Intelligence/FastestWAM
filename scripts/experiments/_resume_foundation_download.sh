#!/usr/bin/env bash
# Resume OpenWAM-Alpha foundation weights via hf-mirror + the 37890 proxy.
# Do not print secrets. Safe to re-run; aria2 -c resumes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
if [[ -f "${ROOT}/.env" ]]; then
    set -a
    source "${ROOT}/.env"
    set +a
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

OUT_DIR="${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"
mkdir -p "$OUT_DIR"
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN missing; cannot authenticate Hugging Face" >&2
    exit 1
fi

# Prefer the ModelScope/hf-mirror proxy. Fall back to whatever .env exported.
PROXY="${HTTPS_PROXY:-${https_proxy:-http://172.22.121.81:37890}}"
URL="https://hf-mirror.com/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model/resolve/main/checkpoint_step_154000.safetensors"

echo "[$(date -Is)] resuming foundation download via proxy ${PROXY} -> ${OUT_DIR}"
echo "[$(date -Is)] source=${URL}"
exec aria2c -c -x 16 -s 16 -j 1 \
    --file-allocation=none \
    --max-tries=0 --retry-wait=5 --timeout=120 \
    --http-proxy="${PROXY}" \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    --dir="${OUT_DIR}" \
    --out=checkpoint_step_154000.safetensors \
    "${URL}"
