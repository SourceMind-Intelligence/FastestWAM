#!/usr/bin/env bash
# Beat DoT-FM on shallow ActionDiT: warm-start Mixed SF MIP-2 from the
# strong DoT-FM checkpoint, freeze Video DiT, detach bridges, raise action LR.
#
# Diagnosis (LIBERO 2-step NFE-matched):
#   DoT-FM  (Action trained) : 94.15%
#   DoT-SF  (Action ~random) : ~4%   → SF cannot bootstrap from noise on 4-layer Action
#
# Hypothesis: Mixed SF needs a decent pred_0. Starting from DoT-FM Action and
# concentrating capacity on ActionDiT (freeze video / detach bridges) lets the
# 2-step MIP refine beat 2-step FM.
#
# Usage:
#   # 4 free GPUs while SF eval occupies 0,1 (+render 4,5):
#   GPUS=2,3,6,7 NPROC_PER_RUN=4 WAIT_FOR_SUITE=0 \
#     bash scripts/experiments/run_libero_dot_sf_beat_fm.sh train
#   bash scripts/experiments/run_libero_dot_sf_beat_fm.sh smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"

export PATH="/root/tengfei/envs/openwam/bin:/opt/conda/envs/fastwam/bin:${PATH}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODE="${1:-train}"   # smoke | train
FM_CKPT="${FM_CKPT:-${ROOT}/outputs/experiments/libero_dot_sf_2026-10-13_17-44-16/FM_flow/2026-10-14_14-39-10}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT}/outputs/experiments/libero_dot_sf_beat_fm_${STAMP}}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
GPUS="${GPUS:-2,3,6,7}"
NPROC_PER_RUN="${NPROC_PER_RUN:-4}"
MASTER_PORT="${MASTER_PORT:-29741}"
# 4-GPU: bs=8 × accum=4 → global 128 (matches prior 8×8×2)
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
ACTION_LR="${ACTION_LR:-2e-4}"

if [[ ! -d "${FM_CKPT}" ]] && [[ ! -f "${FM_CKPT}" ]]; then
    echo "[ERROR] DoT-FM checkpoint missing: ${FM_CKPT}" >&2
    exit 1
fi

FREE_GB="$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')"
if [[ "${FREE_GB}" -lt 400 ]]; then
    echo "[ERROR] need >=400GiB free (have ${FREE_GB}GiB)" >&2
    exit 1
fi

mkdir -p "${SUITE_ROOT}" "${ROOT}/outputs/logs"
{
  echo "hypothesis=FM_warmstart_MixedSF_freeze_video_beat_FM2step"
  echo "fm_ckpt=${FM_CKPT}"
  echo "arch=dual_system_dot joint_cross_attn dot_num_layers=4"
  echo "freeze+=video_backbone.dit detach_bridge=true"
  echo "action_lr=${ACTION_LR} video frozen"
  echo "gpus=${GPUS} nproc=${NPROC_PER_RUN} global_batch=$((NPROC_PER_RUN * BATCH_SIZE * GRAD_ACCUM))"
  echo "num_epochs=${NUM_EPOCHS}"
  echo "finetune_strict_load=true  # same DoT arch as FM ckpt"
} | tee "${SUITE_ROOT}/RUN_INFO.txt"

# Full freeze list (yaml defaults + Video DiT). Hydra list override replaces.
FREEZE_LIST='[video_backbone.text_encoder,video_backbone.reason1,video_backbone.vae,video_backbone.image_encoder,video_backbone.video_encoder,video_backbone.dit]'

COMMON_TRAIN=(
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
    model.architecture.dot_num_layers=4
    model.architecture.variant=joint_cross_attn
    model.architecture.action_objective=mip
    model.architecture.mip_t_star=0.9
    model.architecture.mip_num_steps=2
    model.architecture.mip_refine_mode=mixed
    model.architecture.detach_bridge=true
    "model.freeze=${FREEZE_LIST}"
    "training.action_lr=${ACTION_LR}"
    training.learning_rate=1e-4
)

if [[ "${MODE}" == "smoke" ]]; then
    echo "[smoke] FM-warmstart Mixed SF (action-only) debug"
    CUDA_VISIBLE_DEVICES="${GPUS}" NPROC_PER_NODE="${NPROC_PER_RUN}" MASTER_PORT="${MASTER_PORT}" \
        bash scripts/train.sh \
        "${COMMON_TRAIN[@]}" \
        training.debug=true \
        training.batch_size=1 \
        training.gradient_accumulation_steps=1 \
        "training.output_path=${SUITE_ROOT}/smoke" \
        "hydra.run.dir=${SUITE_ROOT}/smoke/hydra" \
        project.wandb.project=null
    echo "[smoke] OK — ${SUITE_ROOT}"
    exit 0
fi

if [[ "${MODE}" != "train" ]]; then
    echo "Usage: $0 {smoke|train}" >&2
    exit 2
fi

RUN_NAME="SF_mip_mixed_from_FM_freeze_video"
OUT_PARENT="${SUITE_ROOT}/${RUN_NAME}"
LOG="${ROOT}/outputs/logs/libero_dot_${RUN_NAME}_${STAMP}.log"
mkdir -p "${OUT_PARENT}"

wandb_args=(project.wandb.project=null)
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "[$(date -Is)] launch ${RUN_NAME} gpus=${GPUS} port=${MASTER_PORT} epochs=${NUM_EPOCHS} action_lr=${ACTION_LR}" \
  | tee -a "${SUITE_ROOT}/launch.log"

nohup env \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    NPROC_PER_NODE="${NPROC_PER_RUN}" \
    MASTER_PORT="${MASTER_PORT}" \
    bash scripts/train.sh \
        "${COMMON_TRAIN[@]}" \
        "training.output_path=${OUT_PARENT}" \
        "hydra.run.dir=${OUT_PARENT}/hydra" \
        training.debug=false \
        "${wandb_args[@]}" \
    >"${LOG}" 2>&1 &
echo $! > "${OUT_PARENT}/train.pid"
disown "$(cat "${OUT_PARENT}/train.pid")" 2>/dev/null || true
echo "  pid=$(cat "${OUT_PARENT}/train.pid") log=${LOG}"
echo "[$(date -Is)] suite root ${SUITE_ROOT}"

if [[ "${WAIT_FOR_SUITE:-0}" == "1" ]]; then
    wait "$(cat "${OUT_PARENT}/train.pid")" || true
fi
