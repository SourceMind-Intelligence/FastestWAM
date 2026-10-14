#!/usr/bin/env bash
# After MIP eval finishes and Sim-LIBERO weights are complete, run the official
# flow-matching LIBERO checkpoint as a control.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
WEIGHT="${CKPT_DIR}/checkpoint_step_10690.safetensors"
EXPECTED_BYTES="${EXPECTED_WEIGHT_BYTES:-24813767464}"
MIP_OUTPUT="${MIP_OUTPUT:-${ROOT}/outputs/libero/mip_final_21370}"
OUT="${OUTPUT_DIR:-${ROOT}/outputs/libero/openwam_alpha_sim_libero_10690}"
LOG_DIR="${ROOT}/outputs/logs"
mkdir -p "$LOG_DIR"
PIPE_LOG="${LOG_DIR}/sim_libero_eval_pipeline.log"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE_LOG"; }

wait_for_weights() {
    log "Waiting for Sim-LIBERO weights: ${WEIGHT}"
    while true; do
        local bytes=0 disk=0 aria="no"
        if [[ -f "${WEIGHT}" ]]; then
            bytes="$(stat -c '%s' "${WEIGHT}")"
            disk="$(du --block-size=1 -- "${WEIGHT}" | awk '{print $1}')"
        fi
        [[ -e "${WEIGHT}.aria2" ]] && aria="yes"
        log "progress stat=${bytes} allocated=${disk} aria2_sidecar=${aria}"
        if [[ "${bytes}" == "${EXPECTED_BYTES}" && "${disk}" -ge 20000000000 && "${aria}" == "no" ]]; then
            [[ -f "${CKPT_DIR}/config.yaml" && -f "${CKPT_DIR}/normalization_stats.npy" ]] || {
                log "ERROR: config.yaml or normalization_stats.npy missing under ${CKPT_DIR}"
                exit 2
            }
            log "Sim-LIBERO checkpoint is complete."
            return 0
        fi
        sleep 60
    done
}

wait_for_mip_eval() {
    log "Waiting for MIP eval to release GPUs (output=${MIP_OUTPUT})"
    while pgrep -f "run_all_suites.py .*${MIP_OUTPUT}" >/dev/null 2>&1 \
        || pgrep -f "run_eval.sh .*mip_final_21370" >/dev/null 2>&1 \
        || pgrep -f "deploy.py .*checkpoint_step_21370" >/dev/null 2>&1; do
        log "MIP eval still running"
        sleep 60
    done
    log "MIP eval is not running."
}

wait_for_weights
wait_for_mip_eval
sleep 5
log "Starting official OpenWAM-Alpha-Sim-LIBERO eval"
OUTPUT_DIR="${OUT}" bash "${ROOT}/scripts/experiments/run_libero_eval_stable.sh" \
    "${CKPT_DIR}" checkpoint_step_10690.safetensors "${OUT}"
log "Official Sim-LIBERO eval exited $?"
