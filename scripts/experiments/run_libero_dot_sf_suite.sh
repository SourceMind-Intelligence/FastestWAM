#!/usr/bin/env bash
# DoT + SF MIP-2 vs DoT + FM comparison on LIBERO.
#
# DoT = dual_system joint_cross_attn + 4-layer ActionDiT (dot_num_layers=4).
# Warm-start from OpenWAM-Alpha MoT foundation with partial weight load
# (video transfers; action DoT mostly random-init).
#
# Budget (8×H100, global batch 128, ~2-day wall for both runs + eval):
#   num_epochs=4 each  (~prior 5-ep MoT FM ≈16h on 4GPU; 8GPU+DoT should be faster;
#   MIP-2 ~1.5× FM → SF≈10–14h, FM≈6–9h, eval leftover)
#
# Modes:
#   smoke          — 8-GPU Mixed SF, few debug steps
#   train_sf       — DoT + Mixed SF MIP-2 on all 8 GPUs  [start here]
#   train_fm       — DoT + FM on all 8 GPUs (after SF)
#   overnight      — train_sf then train_fm sequentially
#
# Usage:
#   bash scripts/experiments/run_libero_dot_sf_suite.sh smoke
#   WAIT_FOR_SUITE=0 bash scripts/experiments/run_libero_dot_sf_suite.sh train_sf
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

MODE="${1:-overnight}"   # smoke | train_sf | train_fm | overnight
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT}/outputs/experiments/libero_dot_sf_${STAMP}}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
NPROC_PER_RUN=8
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

if [[ ! -d "${FOUNDATION_CKPT}" ]]; then
    echo "[ERROR] foundation checkpoint missing: ${FOUNDATION_CKPT}" >&2
    exit 1
fi

FREE_GB="$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')"
if [[ "${FREE_GB}" -lt 400 ]]; then
    echo "[ERROR] need >=400GiB free (have ${FREE_GB}GiB)" >&2
    exit 1
fi

mkdir -p "${SUITE_ROOT}" "${ROOT}/outputs/logs"
echo "${STAMP}" > "${SUITE_ROOT}/STAMP.txt"
{
  echo "foundation=${FOUNDATION_CKPT}"
  echo "arch=dual_system_dot joint_cross_attn dot_num_layers=4"
  echo "layout=8gpu_sequential SF_then_FM"
  echo "global_batch=128 (8*8*grad_accum2)"
  echo "num_epochs=${NUM_EPOCHS}"
  echo "finetune_strict_load=false"
} > "${SUITE_ROOT}/RUN_INFO.txt"

COMMON_TRAIN=(
    dataloader=libero
    model=dual_system_dot
    "training.finetune_ckpt_path=${FOUNDATION_CKPT}"
    training.finetune_strict_load=false
    training.mixed_precision=bf16
    training.zero_stage=2
    training.use_gradient_checkpointing=true
    training.offload_optimizer_device=none
    # 8-GPU: bs=8 × accum=2 → global 128 (matches official). bs=16 risks MIP OOM.
    training.batch_size=8
    training.gradient_accumulation_steps=2
    "training.num_epochs=${NUM_EPOCHS}"
    training.max_steps=null
    # Disk budget (~1–2TiB free): NO full DeepSpeed resume (~90GiB each).
    # Keep 1 rolling model + 1 mid milestone only (~22GiB × 2 peak).
    training.save_steps=4000
    training.keep_last_k_ckpts=1
    training.max_full_ckpts=0
    training.save_full_states_for_resume=false
    'training.milestone_save_steps=[4000]'
    training.max_milestone_ckpts=1
    training.disk_guard_warn_gb=600
    training.disk_guard_critical_gb=400
    training.measure_flops=false
    project.seed=42
    hydra.job.chdir=false
    model.architecture.dot_num_layers=4
    model.architecture.variant=joint_cross_attn
)

launch_run() {
    local run_id="$1"
    local master_port="$2"
    local objective="$3"
    local refine_mode="$4"
    local run_name="$5"

    local out_parent="${SUITE_ROOT}/${run_name}"
    local log="${ROOT}/outputs/logs/libero_dot_${run_name}_${STAMP}.log"
    mkdir -p "${out_parent}"

    local wandb_args=()
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        wandb_args+=(project.wandb.project=null)
        export WANDB_MODE="${WANDB_MODE:-disabled}"
    else
        export WANDB_MODE="${WANDB_MODE:-online}"
        wandb_args=(
            "project.wandb.project=${WANDB_PROJECT:-fastestwam}"
            "project.wandb.run_name=dot_${run_name}_${STAMP}"
        )
        if [[ -n "${WANDB_ENTITY:-}" ]]; then
            wandb_args+=("project.wandb.entity=${WANDB_ENTITY}")
        fi
    fi

    local extra=(
        "training.output_path=${out_parent}"
        "hydra.run.dir=${out_parent}/hydra"
        training.debug=false
        "${wandb_args[@]}"
    )
    if [[ "${objective}" == "mip" ]]; then
        extra+=(
            model.architecture.action_objective=mip
            model.architecture.mip_t_star=0.9
            model.architecture.mip_num_steps=2
            "model.architecture.mip_refine_mode=${refine_mode}"
        )
    else
        extra+=(model.architecture.action_objective=flow)
    fi

    echo "[$(date -Is)] launch ${run_id} name=${run_name} gpus=${GPUS} port=${master_port} obj=${objective} refine=${refine_mode} epochs=${NUM_EPOCHS}" | tee -a "${SUITE_ROOT}/launch.log"
    # nohup so the job survives when this launcher exits under WAIT_FOR_SUITE=0
    nohup env \
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        NPROC_PER_NODE="${NPROC_PER_RUN}" \
        MASTER_PORT="${master_port}" \
        bash scripts/train.sh "${COMMON_TRAIN[@]}" "${extra[@]}" \
        >"${log}" 2>&1 &
    echo $! > "${out_parent}/train.pid"
    disown "$(cat "${out_parent}/train.pid")" 2>/dev/null || true
    echo "  pid=$(cat "${out_parent}/train.pid") log=${log}"
}

wait_pid_file() {
    local f="$1"
    if [[ ! -f "$f" ]]; then return 0; fi
    local pid
    pid="$(cat "$f")"
    echo "[$(date -Is)] waiting for PID ${pid}"
    wait "${pid}" || true
}

if [[ "${MODE}" == "smoke" ]]; then
    echo "[smoke] DoT Mixed SF MIP-2 on 8 GPUs (debug)"
    CUDA_VISIBLE_DEVICES="${GPUS}" NPROC_PER_NODE=8 MASTER_PORT=29701 \
        bash scripts/train.sh \
        "${COMMON_TRAIN[@]}" \
        model.architecture.action_objective=mip \
        model.architecture.mip_num_steps=2 \
        model.architecture.mip_refine_mode=mixed \
        training.debug=true \
        training.batch_size=1 \
        training.gradient_accumulation_steps=1 \
        training.save_full_states_for_resume=false \
        "training.output_path=${SUITE_ROOT}/smoke_dot_mip_mixed" \
        "hydra.run.dir=${SUITE_ROOT}/smoke_dot_mip_mixed/hydra" \
        project.wandb.project=null
    echo "[smoke] OK — suite root ${SUITE_ROOT}"
    exit 0
fi

case "${MODE}" in
  train_sf)
    launch_run SF 29711 mip mixed "SF_mip_mixed"
    if [[ "${WAIT_FOR_SUITE:-1}" == "1" ]]; then
        wait_pid_file "${SUITE_ROOT}/SF_mip_mixed/train.pid"
    fi
    ;;
  train_fm)
    launch_run FM 29712 flow - "FM_flow"
    if [[ "${WAIT_FOR_SUITE:-1}" == "1" ]]; then
        wait_pid_file "${SUITE_ROOT}/FM_flow/train.pid"
    fi
    ;;
  overnight)
    launch_run SF 29711 mip mixed "SF_mip_mixed"
    if [[ "${WAIT_FOR_SUITE:-1}" == "1" ]]; then
        wait_pid_file "${SUITE_ROOT}/SF_mip_mixed/train.pid"
        launch_run FM 29712 flow - "FM_flow"
        wait_pid_file "${SUITE_ROOT}/FM_flow/train.pid"
    fi
    ;;
  *)
    echo "Usage: $0 {smoke|train_sf|train_fm|overnight}" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] suite root ${SUITE_ROOT}"
