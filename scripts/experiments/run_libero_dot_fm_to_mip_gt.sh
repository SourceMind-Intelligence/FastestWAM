#!/usr/bin/env bash
# Soft FM→MIP conversion on DoT (next after fair-pair failure of Mixed SF).
#
# Fair-pair finding: FM continue 94.2% vs FM→Mixed-SF 3.2% with matched freeze.
# Cause: FM predicts velocity; MIP predicts clean A. Mixed amplifies bad pred0.
#
# This run:
#   - start from DoT-FM ckpt
#   - action_objective=mip, mip_refine_mode=gt  (no self-forcing)
#   - do NOT freeze video DiT; detach_bridge=false
#   - lower action_lr / video_lr to avoid wiping FM features
#
# Usage:
#   WAIT_FOR_SUITE=0 bash scripts/experiments/run_libero_dot_fm_to_mip_gt.sh train
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"

export PATH="/root/tengfei/envs/openwam/bin:/opt/conda/envs/fastwam/bin:${PATH}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE=0 NCCL_RAS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-disabled}"

MODE="${1:-train}"
FM_CKPT="${FM_CKPT:-${ROOT}/outputs/experiments/libero_dot_sf_2026-10-13_17-44-16/FM_flow/2026-10-14_14-39-10}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT}/outputs/experiments/libero_dot_fm_to_mip_gt_${STAMP}}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_RUN="${NPROC_PER_RUN:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
ACTION_LR="${ACTION_LR:-5e-5}"
VIDEO_LR="${VIDEO_LR:-1e-5}"
MASTER_PORT="${MASTER_PORT:-29761}"

[[ -d "${FM_CKPT}" || -f "${FM_CKPT}" ]] || { echo "missing FM ckpt ${FM_CKPT}" >&2; exit 1; }

mkdir -p "${SUITE_ROOT}" "${ROOT}/outputs/logs"
{
  echo "hypothesis=FM_to_MIP_gt_soft_convert_no_freeze"
  echo "fm_ckpt=${FM_CKPT}"
  echo "mip_refine_mode=gt detach_bridge=false video_dit_trainable"
  echo "action_lr=${ACTION_LR} video_lr=${VIDEO_LR} epochs=${NUM_EPOCHS}"
  echo "global_batch=$((NPROC_PER_RUN * BATCH_SIZE * GRAD_ACCUM))"
} | tee "${SUITE_ROOT}/RUN_INFO.txt"

# Default yaml freeze only (encoders/vae) — NOT video_backbone.dit
COMMON=(
    dataloader=libero
    model=dual_system_dot
    "training.finetune_ckpt_path=${FM_CKPT}"
    training.finetune_strict_load=true
    training.mixed_precision=bf16
    training.zero_stage=2
    training.use_gradient_checkpointing=true
    training.offload_optimizer_device=none
    "training.batch_size=${BATCH_SIZE}"
    "training.gradient_accumulation_steps=${GRAD_ACCUM}"
    "training.num_epochs=${NUM_EPOCHS}"
    training.max_steps=null
    training.save_steps=2000
    training.keep_last_k_ckpts=1
    training.max_full_ckpts=0
    training.save_full_states_for_resume=false
    'training.milestone_save_steps=[2000]'
    training.max_milestone_ckpts=1
    training.disk_guard_warn_gb=600
    training.disk_guard_critical_gb=400
    training.measure_flops=false
    project.seed=42
    hydra.job.chdir=false
    project.wandb.project=null
    model.architecture.dot_num_layers=4
    model.architecture.variant=joint_cross_attn
    model.architecture.action_objective=mip
    model.architecture.mip_t_star=0.9
    model.architecture.mip_num_steps=2
    model.architecture.mip_refine_mode=gt
    model.architecture.detach_bridge=false
    "training.action_lr=${ACTION_LR}"
    "training.video_lr=${VIDEO_LR}"
    training.learning_rate=1e-4
)

RUN_NAME="MIP_gt_from_FM"
OUT="${SUITE_ROOT}/${RUN_NAME}"
LOG="${ROOT}/outputs/logs/libero_dot_${RUN_NAME}_${STAMP}.log"
mkdir -p "${OUT}"

echo "[$(date -Is)] launch ${RUN_NAME} gpus=${GPUS} epochs=${NUM_EPOCHS}" | tee -a "${SUITE_ROOT}/launch.log"
nohup env CUDA_VISIBLE_DEVICES="${GPUS}" NPROC_PER_NODE="${NPROC_PER_RUN}" MASTER_PORT="${MASTER_PORT}" \
  bash scripts/train.sh "${COMMON[@]}" \
    "training.output_path=${OUT}" \
    "hydra.run.dir=${OUT}/hydra" \
    training.debug=false \
  >"${LOG}" 2>&1 &
echo $! > "${OUT}/train.pid"
disown "$(cat "${OUT}/train.pid")" 2>/dev/null || true
echo "  pid=$(cat "${OUT}/train.pid") log=${LOG} suite=${SUITE_ROOT}"

[[ "${WAIT_FOR_SUITE:-0}" == "1" ]] && wait "$(cat "${OUT}/train.pid")" || true
