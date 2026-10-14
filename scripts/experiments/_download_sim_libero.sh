#!/usr/bin/env bash
# Download OpenWAM-Alpha-Sim-LIBERO via hf-mirror + the 37890 HTTP proxy.
# huggingface_hub is skipped: this host's leftover ALL_PROXY is SOCKS.
# Secrets stay in .env.
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
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[ERROR] HF_TOKEN missing; cannot authenticate Hugging Face" >&2
    exit 1
fi

unset ALL_PROXY all_proxy
PROXY="${HTTPS_PROXY:-${https_proxy:-http://172.22.121.81:37890}}"
export https_proxy="${PROXY}"
export http_proxy="${PROXY}"
export HTTPS_PROXY="${PROXY}"
export HTTP_PROXY="${PROXY}"

OUT_DIR="${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
mkdir -p "${OUT_DIR}/tokenizer/google/umt5-xxl"
BASE="https://hf-mirror.com/OpenWAM/OpenWAM-Alpha-Sim-LIBERO/resolve/main"

curl_auth() {
    curl -fsSL --proxy "${PROXY}" -H "Authorization: Bearer ${HF_TOKEN}" "$@"
}

echo "[$(date -Is)] downloading Sim-LIBERO small files -> ${OUT_DIR}"
for rel in config.yaml README.md .gitattributes normalization_stats.npy \
    tokenizer/google/umt5-xxl/tokenizer_config.json \
    tokenizer/google/umt5-xxl/special_tokens_map.json \
    tokenizer/google/umt5-xxl/spiece.model \
    tokenizer/google/umt5-xxl/tokenizer.json; do
    dest="${OUT_DIR}/${rel}"
    mkdir -p "$(dirname "$dest")"
    if [[ -f "${dest}" && ! -e "${dest}.aria2" ]]; then
        echo "  skip ${rel}"
        continue
    fi
    echo "  get ${rel}"
    curl_auth -o "${dest}" "${BASE}/${rel}"
done

WEIGHT="${OUT_DIR}/checkpoint_step_10690.safetensors"
EXPECTED_BYTES="${EXPECTED_WEIGHT_BYTES:-24813767464}"
if [[ -f "${WEIGHT}" && ! -e "${WEIGHT}.aria2" ]]; then
    actual="$(stat -c '%s' "${WEIGHT}")"
    if [[ "${actual}" == "${EXPECTED_BYTES}" ]]; then
        echo "[$(date -Is)] weights already complete (${actual} bytes)"
        exit 0
    fi
fi

URL="${BASE}/checkpoint_step_10690.safetensors"
echo "[$(date -Is)] resuming weights via proxy ${PROXY}"
echo "[$(date -Is)] source=${URL}"
exec aria2c -c -x 16 -s 16 -j 1 \
    --file-allocation=none \
    --max-tries=0 --retry-wait=5 --timeout=120 \
    --http-proxy="${PROXY}" \
    --header="Authorization: Bearer ${HF_TOKEN}" \
    --dir="${OUT_DIR}" \
    --out=checkpoint_step_10690.safetensors \
    "${URL}"
