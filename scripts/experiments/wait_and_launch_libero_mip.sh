#!/usr/bin/env bash
# Wait for foundation weights, run 8-GPU MIP debug, then 8-GPU 10-epoch training.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"

export PYTHONUNBUFFERED=1
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
WEIGHT_FILE="${FOUNDATION_CKPT}/checkpoint_step_154000.safetensors"
EXPECTED_BYTES="${EXPECTED_WEIGHT_BYTES:-24813767464}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${ROOT}/outputs/logs"
PIPE_LOG="${LOG_DIR}/mip_pipeline_${STAMP}.log"
mkdir -p "$LOG_DIR"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE_LOG"; }

wait_for_foundation() {
    log "Waiting for complete foundation weights: ${WEIGHT_FILE}"
    local last_disk=0
    local stall=0
    while true; do
        local bytes=0
        local disk=0
        if [[ -f "${WEIGHT_FILE}" ]]; then
            bytes="$(stat -c '%s' "${WEIGHT_FILE}")"
            disk="$(du --block-size=1 -- "${WEIGHT_FILE}" | awk '{print $1}')"
        fi
        local aria="no"
        [[ -e "${WEIGHT_FILE}.aria2" ]] && aria="yes"
        local aria_alive="no"
        pgrep -f "aria2c .*checkpoint_step_154000.safetensors" >/dev/null 2>&1 && aria_alive="yes"
        log "progress stat=${bytes} allocated=${disk} aria2_sidecar=${aria} aria2_alive=${aria_alive}"
        if [[ "${bytes}" == "${EXPECTED_BYTES}" && "${disk}" -ge 20000000000 && "${aria}" == "no" ]]; then
            log "Foundation checkpoint is complete."
            return 0
        fi
        if [[ "${aria_alive}" == "no" && "${aria}" == "yes" ]]; then
            log "downloader died; restarting hf-mirror resume"
            tmux send-keys -t mip_mvp:foundation C-c 2>/dev/null || true
            sleep 1
            tmux send-keys -t mip_mvp:foundation "cd ${ROOT} && bash scripts/experiments/_resume_foundation_download.sh" Enter
        elif [[ "${aria}" == "yes" && "${disk}" -le "${last_disk}" ]]; then
            stall=$((stall + 1))
            if [[ "${stall}" -ge 5 ]]; then
                log "download stalled 5 min; restarting hf-mirror resume"
                tmux send-keys -t mip_mvp:foundation C-c 2>/dev/null || true
                sleep 1
                tmux send-keys -t mip_mvp:foundation "cd ${ROOT} && bash scripts/experiments/_resume_foundation_download.sh" Enter
                stall=0
            fi
        else
            stall=0
        fi
        last_disk="${disk}"
        sleep 60
    done
}

wait_for_foundation

DEBUG_LOG="${LOG_DIR}/libero_alpha_mip_debug_${STAMP}.log"
DEBUG_PARENT="${ROOT}/outputs/experiments/libero_alpha_mip_debug_${STAMP}"
log "Starting 8-GPU MIP debug (20 steps, batch_size=1)"
set +e
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 MASTER_PORT=29511 STAMP="${STAMP}" \
    OUTPUT_PATH="${DEBUG_PARENT}" LOG_PATH="${DEBUG_LOG}" \
    bash "${ROOT}/scripts/experiments/run_libero_alpha_mip.sh" debug
DEBUG_RC=$?
set -e
log "Debug launcher exit code: ${DEBUG_RC}"
if [[ "${DEBUG_RC}" -ne 0 ]]; then
    log "ERROR: debug launcher failed; not starting overnight."
    exit 11
fi
if ! /usr/bin/env python "${ROOT}/scripts/experiments/_check_debug_log.py" \
        "${DEBUG_LOG}" --min-steps 20 --expect-nproc 8 --ckpt-dir "${DEBUG_PARENT}" \
        | tee -a "$PIPE_LOG"; then
    log "ERROR: debug log did not meet success criteria; not starting overnight."
    exit 11
fi

log "Debug succeeded. Launching overnight 8-GPU MIP 10-epoch run."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 MASTER_PORT=29512 STAMP="${STAMP}" \
    bash "${ROOT}/scripts/experiments/run_libero_alpha_mip.sh" overnight
log "Overnight MIP launcher exited"
