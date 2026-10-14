#!/usr/bin/env bash
# OpenWAM-Alpha foundation -> LIBERO DualSystem 4-step MIP (train/infer shared grid).
# 4×H100. Per-GPU batch matches the previous 8-GPU run (global batch 8).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export PATH="${PATH}:/opt/conda/envs/fastwam/bin"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${1:-overnight}"  # debug | overnight
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"

if [[ ! -d "${FOUNDATION_CKPT}" ]]; then
    echo "[ERROR] foundation checkpoint missing: ${FOUNDATION_CKPT}" >&2
    exit 1
fi
WEIGHT_FILE="$(ls -1 "${FOUNDATION_CKPT}"/checkpoint_step_*.safetensors 2>/dev/null | head -n1 || true)"
if [[ -z "${WEIGHT_FILE}" || ! -f "${WEIGHT_FILE}" ]]; then
    echo "[ERROR] no checkpoint_step_*.safetensors under ${FOUNDATION_CKPT}" >&2
    exit 1
fi

WANDB_ARGS=()
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[warn] WANDB_API_KEY unset — disabling wandb"
    WANDB_ARGS+=(project.wandb.project=null)
    export WANDB_MODE="${WANDB_MODE:-disabled}"
else
    export WANDB_MODE="${WANDB_MODE:-online}"
    WANDB_ARGS+=(
        "project.wandb.project=${WANDB_PROJECT:-fastestwam}"
        "project.wandb.run_name=${WANDB_RUN_NAME:-libero_alpha_mip4_${STAMP}}"
    )
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        WANDB_ARGS+=("project.wandb.entity=${WANDB_ENTITY}")
    fi
fi

COMMON=(
    dataloader=libero
    model=dual_system
    model.architecture.action_objective=mip
    model.architecture.mip_t_star=0.9
    model.architecture.mip_num_steps=4
    "training.finetune_ckpt_path=${FOUNDATION_CKPT}"
    training.mixed_precision=bf16
    training.zero_stage=2
    training.use_gradient_checkpointing=true
    project.seed=42
    hydra.job.chdir=false
)

if [[ "${MODE}" == "debug" ]]; then
    OUT_PARENT="${OUTPUT_PATH:-${ROOT}/outputs/experiments/libero_alpha_mip4_debug_${STAMP}}"
    LOG="${LOG_PATH:-${ROOT}/outputs/logs/libero_alpha_mip4_debug_${STAMP}.log}"
    EXTRA=(
        training.debug=true
        training.batch_size=1
        "training.output_path=${OUT_PARENT}"
        "hydra.run.dir=${OUT_PARENT}/hydra"
        "${WANDB_ARGS[@]}"
    )
else
    # Previous 8-GPU run used global batch 16 (2/GPU). Keep 2/GPU → global 8 on 4 GPUs.
    OUT_PARENT="${OUTPUT_PATH:-${ROOT}/outputs/experiments/libero_alpha_mip4_${STAMP}}"
    LOG="${LOG_PATH:-${ROOT}/outputs/logs/libero_alpha_mip4_${STAMP}.log}"
    EXTRA=(
        training.debug=false
        training.num_epochs=10
        training.max_steps=null
        training.batch_size=8
        training.save_steps=1000
        training.keep_last_k_ckpts=100
        training.save_full_states_for_resume=true
        "training.output_path=${OUT_PARENT}"
        "hydra.run.dir=${OUT_PARENT}/hydra"
        "${WANDB_ARGS[@]}"
    )
fi

mkdir -p "$(dirname "$LOG")" "$OUT_PARENT"
echo "[$(date -Is)] MODE=${MODE} NPROC=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} FOUNDATION=${FOUNDATION_CKPT}" | tee "$LOG"
echo "[$(date -Is)] MIP grid=linspace(0,0.9,4)={0,0.3,0.6,0.9} batch_size overrides=${EXTRA[*]}" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
  bash scripts/train.sh "${COMMON[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
