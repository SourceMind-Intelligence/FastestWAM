#!/usr/bin/env bash
# Fair compare from the SAME DoT-FM checkpoint, SAME +N epochs budget:
#   arm SF  — Mixed MIP-2 Self-Forcing
#   arm FM  — continue Flow Matching
#
# Hyperparams matched (only action_objective differs):
#   freeze video DiT, detach_bridge, action_lr=2e-4, global batch 128
#
# Usage:
#   # SF already trained under beat_fm suite — only need FM continue:
#   GPUS=0,1,2,3,4,5,6,7 NPROC_PER_RUN=8 GRAD_ACCUM=2 \
#     SUITE_ROOT=outputs/experiments/libero_dot_fair_fm_vs_sf_<stamp> \
#     bash scripts/experiments/run_libero_dot_fair_fm_vs_sf.sh train_fm
#
#   bash scripts/experiments/run_libero_dot_fair_fm_vs_sf.sh train_sf
#   bash scripts/experiments/run_libero_dot_fair_fm_vs_sf.sh both
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
export WANDB_MODE="${WANDB_MODE:-disabled}"

MODE="${1:-train_fm}"   # train_sf | train_fm | both | smoke
FM_CKPT="${FM_CKPT:-${ROOT}/outputs/experiments/libero_dot_sf_2026-10-13_17-44-16/FM_flow/2026-10-14_14-39-10}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT}/outputs/experiments/libero_dot_fair_fm_vs_sf_${STAMP}}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_RUN="${NPROC_PER_RUN:-8}"
# 8-GPU: bs=8 × accum=2 → global 128 (same as prior 4×8×4 SF arm)
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
ACTION_LR="${ACTION_LR:-2e-4}"
MASTER_PORT_SF="${MASTER_PORT_SF:-29751}"
MASTER_PORT_FM="${MASTER_PORT_FM:-29752}"

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
  echo "fair_pair=from_DoT_FM_ckpt_+${NUM_EPOCHS}ep_matched_hparams"
  echo "fm_ckpt=${FM_CKPT}"
  echo "arch=dual_system_dot joint_cross_attn dot_num_layers=4"
  echo "freeze+=video_backbone.dit detach_bridge=true action_lr=${ACTION_LR}"
  echo "global_batch=$((NPROC_PER_RUN * BATCH_SIZE * GRAD_ACCUM)) gpus=${GPUS}"
  echo "arms=FM_continue vs SF_mip_mixed  (only action_objective differs)"
  echo "finetune_strict_load=true"
} | tee "${SUITE_ROOT}/RUN_INFO.txt"

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
    model.architecture.detach_bridge=true
    "model.freeze=${FREEZE_LIST}"
    "training.action_lr=${ACTION_LR}"
    training.learning_rate=1e-4
    project.wandb.project=null
)

launch_arm() {
    local run_name="$1"
    local master_port="$2"
    shift 2
    local obj_args=("$@")

    local out_parent="${SUITE_ROOT}/${run_name}"
    local log="${ROOT}/outputs/logs/libero_dot_${run_name}_${STAMP}.log"
    mkdir -p "${out_parent}"

    echo "[$(date -Is)] launch ${run_name} gpus=${GPUS} port=${master_port} epochs=${NUM_EPOCHS}" \
      | tee -a "${SUITE_ROOT}/launch.log"

    nohup env \
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        NPROC_PER_NODE="${NPROC_PER_RUN}" \
        MASTER_PORT="${master_port}" \
        bash scripts/train.sh \
            "${COMMON_TRAIN[@]}" \
            "${obj_args[@]}" \
            "training.output_path=${out_parent}" \
            "hydra.run.dir=${out_parent}/hydra" \
            training.debug=false \
        >"${log}" 2>&1 &
    echo $! > "${out_parent}/train.pid"
    disown "$(cat "${out_parent}/train.pid")" 2>/dev/null || true
    echo "  pid=$(cat "${out_parent}/train.pid") log=${log}"
}

wait_pid_file() {
    local f="$1"
    [[ -f "$f" ]] || return 0
    local pid
    pid="$(cat "$f")"
    echo "[$(date -Is)] waiting for PID ${pid}"
    wait "${pid}" || true
}

SF_ARGS=(
    model.architecture.action_objective=mip
    model.architecture.mip_t_star=0.9
    model.architecture.mip_num_steps=2
    model.architecture.mip_refine_mode=mixed
)
FM_ARGS=(model.architecture.action_objective=flow)

case "${MODE}" in
  smoke)
    CUDA_VISIBLE_DEVICES="${GPUS}" NPROC_PER_NODE="${NPROC_PER_RUN}" MASTER_PORT="${MASTER_PORT_FM}" \
      bash scripts/train.sh \
      "${COMMON_TRAIN[@]}" "${FM_ARGS[@]}" \
      training.debug=true training.batch_size=1 training.gradient_accumulation_steps=1 \
      "training.output_path=${SUITE_ROOT}/smoke_fm" \
      "hydra.run.dir=${SUITE_ROOT}/smoke_fm/hydra"
    echo "[smoke] OK ${SUITE_ROOT}"
    ;;
  train_sf)
    launch_arm "SF_mip_mixed_from_FM" "${MASTER_PORT_SF}" "${SF_ARGS[@]}"
    [[ "${WAIT_FOR_SUITE:-0}" == "1" ]] && wait_pid_file "${SUITE_ROOT}/SF_mip_mixed_from_FM/train.pid"
    ;;
  train_fm)
    launch_arm "FM_continue_from_FM" "${MASTER_PORT_FM}" "${FM_ARGS[@]}"
    [[ "${WAIT_FOR_SUITE:-0}" == "1" ]] && wait_pid_file "${SUITE_ROOT}/FM_continue_from_FM/train.pid"
    ;;
  both)
    # sequential on same GPUs
    launch_arm "SF_mip_mixed_from_FM" "${MASTER_PORT_SF}" "${SF_ARGS[@]}"
    wait_pid_file "${SUITE_ROOT}/SF_mip_mixed_from_FM/train.pid"
    launch_arm "FM_continue_from_FM" "${MASTER_PORT_FM}" "${FM_ARGS[@]}"
    wait_pid_file "${SUITE_ROOT}/FM_continue_from_FM/train.pid"
    ;;
  *)
    echo "Usage: $0 {smoke|train_sf|train_fm|both}" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] suite root ${SUITE_ROOT}"

# Symlink already-finished SF arm into this suite if requested
if [[ -n "${EXISTING_SF_CKPT_DIR:-}" ]]; then
    mkdir -p "${SUITE_ROOT}"
    ln -sfn "${EXISTING_SF_CKPT_DIR}" "${SUITE_ROOT}/SF_mip_mixed_from_FM_EXISTING"
    echo "linked EXISTING_SF_CKPT_DIR -> ${SUITE_ROOT}/SF_mip_mixed_from_FM_EXISTING"
fi
