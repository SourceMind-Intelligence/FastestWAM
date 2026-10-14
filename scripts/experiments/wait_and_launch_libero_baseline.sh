#!/usr/bin/env bash
# Wait for the OpenWAM-Alpha foundation weights, run 8-GPU debug, then overnight.
# Intended to run inside tmux so Cursor/SSH disconnects cannot kill the job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"

export PATH="${PATH}:/opt/conda/envs/fastwam/bin"
export PYTHONUNBUFFERED=1
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
WEIGHT_FILE="${FOUNDATION_CKPT}/checkpoint_step_154000.safetensors"
EXPECTED_BYTES="${EXPECTED_WEIGHT_BYTES:-24813767464}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${ROOT}/outputs/logs"
PIPE_LOG="${LOG_DIR}/pipeline_${STAMP}.log"
DEBUG_PARENT="${ROOT}/outputs/experiments/libero_alpha_flow_debug_${STAMP}"
OVERNIGHT_PARENT="${ROOT}/outputs/experiments/libero_alpha_flow_baseline_${STAMP}"
DEBUG_LOG="${LOG_DIR}/libero_alpha_flow_debug_${STAMP}.log"
OVERNIGHT_LOG="${LOG_DIR}/libero_alpha_flow_baseline_${STAMP}.log"
GPU_LOG="${LOG_DIR}/gpu_monitor_${STAMP}.log"
GPU_CSV="${LOG_DIR}/gpu_util_${STAMP}.csv"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
MASTER_PORT_DEBUG="${MASTER_PORT_DEBUG:-29511}"
MASTER_PORT_OVERNIGHT="${MASTER_PORT_OVERNIGHT:-29512}"

mkdir -p "$LOG_DIR" "$DEBUG_PARENT" "$OVERNIGHT_PARENT"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE_LOG"; }

wait_for_foundation() {
    log "Waiting for complete foundation weights: ${WEIGHT_FILE}"
    log "Expected size: ${EXPECTED_BYTES} bytes"
    local stale=0
    local stall=0
    local last_disk=0
    while true; do
        local bytes=0
        local disk=0
        if [[ -f "${WEIGHT_FILE}" ]]; then
            bytes="$(stat -c '%s' "${WEIGHT_FILE}")"
            disk="$(du --block-size=1 -- "${WEIGHT_FILE}" | awk '{print $1}')"
        fi
        local du_human
        du_human="$(du -sh "${FOUNDATION_CKPT}" 2>/dev/null | awk '{print $1}')"
        local aria="no"
        [[ -e "${WEIGHT_FILE}.aria2" ]] && aria="yes"
        local aria_alive="no"
        pgrep -f "aria2c .*checkpoint_step_154000.safetensors" >/dev/null 2>&1 && aria_alive="yes"
        log "progress stat=${bytes} allocated=${disk} du=${du_human} aria2_sidecar=${aria} aria2_alive=${aria_alive}"
        if [[ "${bytes}" == "${EXPECTED_BYTES}" && "${disk}" -ge 20000000000 && "${aria}" == "no" ]]; then
            /usr/bin/env python - "${WEIGHT_FILE}" <<'PY'
import json, struct, sys
from pathlib import Path
p = Path(sys.argv[1])
raw = p.open("rb").read(8)
header_len = struct.unpack("<Q", raw)[0]
if header_len > 50_000_000:
    raise SystemExit(f"implausible safetensors header_len={header_len}")
js = p.open("rb").read(8 + header_len)[8:]
meta = json.loads(js)
if not isinstance(meta, dict) or len(meta) < 2:
    raise SystemExit("safetensors header is not a dict of tensors")
print(f"safetensors_ok tensors={len(meta)- (1 if '__metadata__' in meta else 0)} header_len={header_len}")
PY
            log "Foundation checkpoint is complete and parseable."
            return 0
        fi
        local need_restart=0
        if [[ "${aria_alive}" == "no" && "${aria}" == "yes" ]]; then
            stale=$((stale + 1))
            if [[ "${stale}" -ge 2 ]]; then
                need_restart=1
                stale=0
            fi
        else
            stale=0
        fi
        if [[ "${aria}" == "yes" && "${disk}" -le "${last_disk}" ]]; then
            stall=$((stall + 1))
            if [[ "${stall}" -ge 3 ]]; then
                log "download stalled (allocated unchanged for ${stall} min) — restarting resume downloader"
                need_restart=1
                stall=0
            fi
        else
            stall=0
        fi
        last_disk="${disk}"
        if [[ "${need_restart}" -eq 1 ]]; then
            log "restarting foundation downloader in tmux window foundation"
            tmux send-keys -t tonight_baseline:foundation C-c 2>/dev/null || true
            sleep 1
            tmux send-keys -t tonight_baseline:foundation "cd ${ROOT} && bash scripts/experiments/_resume_foundation_download.sh" Enter
        fi
        sleep 60
    done
}

snapshot_gpu() {
    local tag="$1"
    {
        echo "===== ${tag} $(date -Is) ====="
        nvidia-smi
        echo "----- cpu -----"
        uptime
        ps -eo pid,pcpu,pmem,rss,cmd --sort=-pcpu | awk 'NR<=20'
    } | tee -a "$GPU_LOG" >/dev/null
}

monitor_gpu_loop() {
    local stop_file="$1"
    while [[ ! -f "${stop_file}" ]]; do
        snapshot_gpu "poll"
        sleep 20
    done
}

parse_debug() {
    /usr/bin/env python "${ROOT}/scripts/experiments/_check_debug_log.py" \
        "$DEBUG_LOG" --min-steps 20 --ckpt-dir "$DEBUG_PARENT"
}

write_run_info() {
    local run_parent="$1"
    local actual_run
    actual_run="$(find "${run_parent}" -mindepth 1 -maxdepth 1 -type d ! -name hydra | sort | tail -n1 || true)"
    local info="${run_parent}/RUN_INFO.txt"
    local train_pid="${2:-}"
    cat >"$info" <<EOF
run_parent: ${run_parent}
actual_run_dir: ${actual_run}
pid: ${train_pid}
tmux_session: tonight_baseline
tmux_window: pipeline
start_timestamp: $(date -Is)
git_sha: $(git -C "${ROOT}" rev-parse HEAD)
foundation_ckpt: ${FOUNDATION_CKPT}
weight_file: ${WEIGHT_FILE}
command: NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_PORT=${MASTER_PORT_OVERNIGHT} STAMP=${STAMP} bash scripts/experiments/run_libero_alpha_flow_baseline.sh overnight
debug_log: ${DEBUG_LOG}
overnight_log: ${OVERNIGHT_LOG}
pipeline_log: ${PIPE_LOG}
gpu_log: ${GPU_LOG}
gpu_csv: ${GPU_CSV}
scaling_metrics: <actual_run_dir>/scaling_metrics.jsonl
plot_later: python scripts/experiments/plot_scaling_metrics.py --metrics <actual_run_dir>/scaling_metrics.jsonl --gpu ${GPU_CSV} --out <run_parent>/scaling_curves.png
current_step_last_checked: see overnight log / tqdm
how_to_resume: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE} MASTER_PORT=${MASTER_PORT_OVERNIGHT} bash scripts/train.sh dataloader=libero training.finetune_ckpt_path=null training.resume_ckpt_path=<actual_run_dir> training.num_epochs=null training.max_steps=5000 training.save_full_states_for_resume=true
how_to_attach: tmux attach -t tonight_baseline
EOF
    if [[ -n "${actual_run}" ]]; then
        cp "$info" "${actual_run}/RUN_INFO.txt"
    fi
    log "Wrote ${info}"
}

wait_initial_overnight_steps() {
    log "Watching overnight log for the first optimizer steps..."
    local waited=0
    while [[ "${waited}" -lt 7200 ]]; do
        if grep -Eiq "CUDA out of memory|NCCL error|Traceback \(most recent call last\)" "$OVERNIGHT_LOG" 2>/dev/null; then
            log "ERROR: overnight log shows a fatal pattern during startup."
            snapshot_gpu "overnight_fail"
            exit 12
        fi
        # tqdm postfix and/or wandb/train logs
        if grep -Eoq "loss[=:][0-9]|train/loss|Training:" "$OVERNIGHT_LOG" 2>/dev/null; then
            local n
            n="$(grep -cE "loss[=:]" "$OVERNIGHT_LOG" 2>/dev/null || true)"
            log "overnight loss-bearing lines so far: ${n}"
            if [[ "${n}" -ge 20 ]]; then
                snapshot_gpu "overnight_steps_ok"
                write_run_info "$OVERNIGHT_PARENT" "$(pgrep -f 'scripts/train.py' | tr '\n' ' ')"
                log "Initial overnight steps look healthy. Leaving the job running."
                return 0
            fi
        fi
        sleep 30
        waited=$((waited + 30))
    done
    log "WARN: timed out waiting for 20 overnight loss lines; job may still be initializing. Not killing it."
    write_run_info "$OVERNIGHT_PARENT" "$(pgrep -f 'scripts/train.py' | tr '\n' ' ')"
}

wait_for_foundation
# Re-source so a late-finishing ~/tengfei/envs/openwam is picked up.
# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"
log "python=$(command -v python) ($(python -c 'import sys; print(sys.executable)'))"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
snapshot_gpu "pre_debug"

log "Starting 8-GPU debug run"
STOP_GPU="${LOG_DIR}/.stop_gpu_mon_${STAMP}"
rm -f "$STOP_GPU"
monitor_gpu_loop "$STOP_GPU" &
MON_PID=$!

set +e
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT_DEBUG}" \
STAMP="${STAMP}" \
OUTPUT_PATH="${DEBUG_PARENT}" \
LOG_PATH="${DEBUG_LOG}" \
FOUNDATION_CKPT="${FOUNDATION_CKPT}" \
bash "${ROOT}/scripts/experiments/run_libero_alpha_flow_baseline.sh" debug
DEBUG_RC=$?
set -e
touch "$STOP_GPU"
wait "$MON_PID" || true
snapshot_gpu "post_debug"
log "Debug launcher exit code: ${DEBUG_RC}"

if [[ "${DEBUG_RC}" -ne 0 ]]; then
    log "ERROR: debug launcher failed; not starting overnight."
    exit 11
fi
if ! parse_debug | tee -a "$PIPE_LOG"; then
    log "ERROR: debug log did not meet success criteria; not starting overnight."
    exit 11
fi

log "Debug succeeded. Launching overnight 8-GPU LIBERO fine-tune."
snapshot_gpu "pre_overnight"
STOP_GPU_OVERNIGHT="${LOG_DIR}/.stop_gpu_mon_overnight_${STAMP}"
rm -f "$STOP_GPU_OVERNIGHT"
monitor_gpu_loop "$STOP_GPU_OVERNIGHT" &
MON2_PID=$!
bash "${ROOT}/scripts/experiments/monitor_gpus.sh" "$GPU_CSV" 10 &
CSV_PID=$!
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT_OVERNIGHT}" \
STAMP="${STAMP}" \
OUTPUT_PATH="${OVERNIGHT_PARENT}" \
LOG_PATH="${OVERNIGHT_LOG}" \
FOUNDATION_CKPT="${FOUNDATION_CKPT}" \
bash "${ROOT}/scripts/experiments/run_libero_alpha_flow_baseline.sh" overnight \
    &
OVERNIGHT_PID=$!
echo "${OVERNIGHT_PID}" > "${OVERNIGHT_PARENT}/train_launcher.pid"
write_run_info "$OVERNIGHT_PARENT" "${OVERNIGHT_PID}"
wait_initial_overnight_steps
log "Overnight launcher PID ${OVERNIGHT_PID} is running. Pipeline watcher now waits on it."
wait "${OVERNIGHT_PID}"
log "Overnight launcher exited with $?"
touch "$STOP_GPU_OVERNIGHT"
kill "${CSV_PID}" 2>/dev/null || true
wait "$MON2_PID" || true
write_run_info "$OVERNIGHT_PARENT" "${OVERNIGHT_PID}"
