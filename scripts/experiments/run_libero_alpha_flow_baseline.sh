#!/usr/bin/env bash
# OpenWAM-Alpha foundation -> official LIBERO DualSystem / Wan2.2-TI2V-5B baseline.
# No architecture / loss / normalization changes. Secrets are read from the
# environment or repo .env; nothing private is hard-coded here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"

export PATH="${PATH}:/opt/conda/envs/fastwam/bin"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

MODE="${1:-overnight}"  # debug | overnight
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
NPROC_PER_NODE="${NPROC_PER_NODE:-7}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"

if [[ ! -d "${FOUNDATION_CKPT}" ]]; then
    echo "[ERROR] foundation checkpoint missing: ${FOUNDATION_CKPT}" >&2
    exit 1
fi
if [[ ! -f "${FOUNDATION_CKPT}/config.yaml" ]]; then
    echo "[ERROR] foundation config.yaml missing under ${FOUNDATION_CKPT}" >&2
    exit 1
fi
# Official HF size for checkpoint_step_154000.safetensors (this checkout's Alpha foundation).
EXPECTED_WEIGHT_BYTES="${EXPECTED_WEIGHT_BYTES:-24813767464}"
WEIGHT_FILE="$(ls -1 "${FOUNDATION_CKPT}"/checkpoint_step_*.safetensors 2>/dev/null | head -n1 || true)"
if [[ -z "${WEIGHT_FILE}" || ! -f "${WEIGHT_FILE}" ]]; then
    echo "[ERROR] no checkpoint_step_*.safetensors under ${FOUNDATION_CKPT}" >&2
    exit 1
fi
if [[ -e "${WEIGHT_FILE}.aria2" ]]; then
    echo "[ERROR] incomplete download sidecar present: ${WEIGHT_FILE}.aria2" >&2
    exit 1
fi
ACTUAL_BYTES="$(stat -c '%s' "${WEIGHT_FILE}")"
DISK_BYTES="$(du --block-size=1 -- "${WEIGHT_FILE}" | awk '{print $1}')"
if [[ "${DISK_BYTES}" -lt 20000000000 ]]; then
    echo "[ERROR] foundation weights not fully on disk (stat=${ACTUAL_BYTES} allocated=${DISK_BYTES}): ${WEIGHT_FILE}" >&2
    exit 1
fi
if [[ "${ACTUAL_BYTES}" != "${EXPECTED_WEIGHT_BYTES}" ]]; then
    echo "[warn] foundation weight size ${ACTUAL_BYTES} != expected ${EXPECTED_WEIGHT_BYTES}" >&2
fi

WANDB_ARGS=()
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[warn] WANDB_API_KEY unset — disabling wandb so login cannot block training"
    WANDB_ARGS+=(project.wandb.project=null)
    export WANDB_MODE="${WANDB_MODE:-disabled}"
else
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_SILENT="${WANDB_SILENT:-true}"
    WANDB_ARGS+=(
        "project.wandb.project=${WANDB_PROJECT:-fastestwam}"
        "project.wandb.run_name=${WANDB_RUN_NAME:-libero_alpha_flow_baseline_${STAMP}}"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_ARGS+=("project.wandb.entity=${WANDB_ENTITY}")
    fi
fi

COMMON=(
    dataloader=libero
    model=dual_system
    "training.finetune_ckpt_path=${FOUNDATION_CKPT}"
    training.mixed_precision=bf16
    training.zero_stage=2
    training.use_gradient_checkpointing=true
    project.seed=42
    hydra.job.chdir=false
)

if [[ "${MODE}" == "debug" ]]; then
    OUT_PARENT="${OUTPUT_PATH:-${ROOT}/outputs/experiments/libero_alpha_flow_debug_${STAMP}}"
    LOG="${LOG_PATH:-${ROOT}/outputs/logs/libero_alpha_flow_debug_${STAMP}.log}"
    EXTRA=(
        training.debug=true
        "training.output_path=${OUT_PARENT}"
        "hydra.run.dir=${OUT_PARENT}/hydra"
        "${WANDB_ARGS[@]}"
    )
else
    OUT_PARENT="${OUTPUT_PATH:-${ROOT}/outputs/experiments/libero_alpha_flow_baseline_${STAMP}}"
    LOG="${LOG_PATH:-${ROOT}/outputs/logs/libero_alpha_flow_baseline_${STAMP}.log}"
    EXTRA=(
        training.debug=false
        training.num_epochs=null
        training.max_steps=5000
        training.save_steps=500
        training.keep_last_k_ckpts=3
        training.save_full_states_for_resume=true
        "training.output_path=${OUT_PARENT}"
        "hydra.run.dir=${OUT_PARENT}/hydra"
        "${WANDB_ARGS[@]}"
    )
fi

mkdir -p "$(dirname "$LOG")" "$OUT_PARENT"
echo "[$(date -Is)] MODE=${MODE} NPROC=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} FOUNDATION=${FOUNDATION_CKPT}" | tee "$LOG"
echo "[$(date -Is)] output_parent=${OUT_PARENT}" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" bash scripts/train.sh "${COMMON[@]}" "${EXTRA[@]}" \
    2>&1 | tee -a "$LOG"
