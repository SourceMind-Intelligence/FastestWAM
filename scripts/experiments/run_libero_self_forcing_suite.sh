#!/usr/bin/env bash
# LIBERO self-forcing suite from OpenWAM-Alpha foundation.
#
# Memory reality: DualSystem + ZeRO-2 + MIP-2 does NOT fit on 2×H100 at
# useful batch sizes. This launcher uses 4 GPUs per run (2 concurrent):
#
#   Wave 1 (priority):  C Mixed SF  GPU 0-3   |   A FM baseline  GPU 4-7
#   Wave 2 (after W1):  B Orig MIP  GPU 0-3   |   D Pred SF      GPU 4-7
#
# Effective batch = official 8×16=128 via 4×16×grad_accum=2.
# Checkpoints: model-only milestones @ 4k/8k/10k opt-steps; 1 full resume.
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

MODE="${1:-overnight}"   # smoke | overnight | wave1 | wave2 | launch_one
RUN_FILTER="${2:-all}"   # all | A | B | C | D
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE_ROOT="${SUITE_ROOT:-${ROOT}/outputs/experiments/libero_sf_suite_${STAMP}}"
NPROC_PER_RUN=4

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
  echo "layout=4gpu_x2_waves"
  echo "global_batch=128 (4*8*grad_accum4)"
  echo "note=2gpu layout OOMs DualSystem+MIP; using 4gpu/run × 2 waves"
} > "${SUITE_ROOT}/RUN_INFO.txt"

COMMON_TRAIN=(
    dataloader=libero
    model=dual_system
    "training.finetune_ckpt_path=${FOUNDATION_CKPT}"
    training.mixed_precision=bf16
    training.zero_stage=2
    training.use_gradient_checkpointing=true
    training.offload_optimizer_device=none
    # 4-GPU: bs=8 × accum=4 → global 128 (matches official 8×16).
    # bs=16 on 4 GPUs risks ZeRO-2 optimizer OOM vs 8-GPU partition.
    training.batch_size=8
    training.gradient_accumulation_steps=4
    training.num_epochs=5
    training.max_steps=null
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
)

launch_run() {
    local run_id="$1"
    local gpus="$2"
    local master_port="$3"
    local objective="$4"
    local refine_mode="$5"
    local run_name="$6"

    local out_parent="${SUITE_ROOT}/${run_name}"
    local log="${ROOT}/outputs/logs/libero_sf_${run_name}_${STAMP}.log"
    mkdir -p "${out_parent}"

    local wandb_args=()
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        wandb_args+=(project.wandb.project=null)
        export WANDB_MODE="${WANDB_MODE:-disabled}"
    else
        export WANDB_MODE="${WANDB_MODE:-online}"
        wandb_args+=(
            "project.wandb.project=${WANDB_PROJECT:-fastestwam}"
            "project.wandb.run_name=sf_${run_name}_${STAMP}"
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

    echo "[$(date -Is)] launch ${run_id} name=${run_name} gpus=${gpus} port=${master_port} obj=${objective} refine=${refine_mode}" | tee -a "${SUITE_ROOT}/launch.log"
    (
        export CUDA_VISIBLE_DEVICES="${gpus}"
        export NPROC_PER_NODE="${NPROC_PER_RUN}"
        export MASTER_PORT="${master_port}"
        bash scripts/train.sh "${COMMON_TRAIN[@]}" "${extra[@]}"
    ) >"${log}" 2>&1 &
    echo $! > "${out_parent}/train.pid"
    echo "  pid=$(cat "${out_parent}/train.pid") log=${log}"
}

should_run() {
    local id="$1"
    [[ "${RUN_FILTER}" == "all" || "${RUN_FILTER}" == "${id}" ]]
}

if [[ "${MODE}" == "launch_one" ]]; then
    RUN_FILTER="${2:?run id A|B|C|D required}"
fi

# Smoke: Mixed SF (C) 20 steps on 4 GPUs, then FM (A).
if [[ "${MODE}" == "smoke" ]]; then
    echo "[smoke] Mixed Self-Forcing MIP-2 on 4 GPUs"
    CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29601 \
        bash scripts/train.sh \
        "${COMMON_TRAIN[@]}" \
        model.architecture.action_objective=mip \
        model.architecture.mip_num_steps=2 \
        model.architecture.mip_refine_mode=mixed \
        training.debug=true \
        training.batch_size=1 \
        training.gradient_accumulation_steps=1 \
        training.save_full_states_for_resume=false \
        "training.output_path=${SUITE_ROOT}/smoke_C_mip_mixed" \
        "hydra.run.dir=${SUITE_ROOT}/smoke_C_mip_mixed/hydra" \
        project.wandb.project=null
    echo "[smoke] OK — suite root ${SUITE_ROOT}"
    exit 0
fi

launch_wave1() {
    if should_run C; then launch_run C "0,1,2,3" 29603 mip mixed "C_mip_mixed"; fi
    if should_run A; then launch_run A "4,5,6,7" 29601 flow - "A_fm"; fi
}

launch_wave2() {
    if should_run B; then launch_run B "0,1,2,3" 29602 mip gt "B_mip_gt"; fi
    if should_run D; then launch_run D "4,5,6,7" 29604 mip pred "D_mip_pred"; fi
}

wait_wave_pids() {
    local pids=()
    for d in "$@"; do
        local f="${SUITE_ROOT}/${d}/train.pid"
        if [[ -f "$f" ]]; then pids+=("$(cat "$f")"); fi
    done
    if [[ ${#pids[@]} -eq 0 ]]; then return 0; fi
    echo "[$(date -Is)] waiting for PIDs: ${pids[*]}"
    wait "${pids[@]}" || true
}

case "${MODE}" in
  wave1) launch_wave1 ;;
  wave2) launch_wave2 ;;
  overnight)
    launch_wave1
    if [[ "${WAIT_FOR_SUITE:-1}" == "1" ]]; then
        wait_wave_pids C_mip_mixed A_fm
        launch_wave2
        if [[ "${WAIT_FOR_SUITE:-1}" == "1" ]]; then
            wait_wave_pids B_mip_gt D_mip_pred
        fi
    fi
    ;;
  *)
    echo "Usage: $0 {smoke|overnight|wave1|wave2|launch_one} [A|B|C|D]" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] launched under ${SUITE_ROOT}"
echo "Monitor: tail -f ${ROOT}/outputs/logs/libero_sf_*_${STAMP}.log"
if [[ "${WAIT_FOR_SUITE:-0}" == "0" && "${MODE}" != "overnight" ]]; then
    echo "Detached. PIDs under ${SUITE_ROOT}/*/train.pid"
fi
