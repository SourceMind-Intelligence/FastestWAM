#!/usr/bin/env bash
# Short FLOPs calibration: measure step TFLOPs for FM vs MIP-2 with FlopCounterMode.
# Uses batch=1 on 4 GPUs to avoid VRAM blow-up. Writes scaling_metrics.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"
export PATH="/root/tengfei/envs/openwam/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OBJ="${1:-mip}"          # flow | mip
REFINE="${2:-mixed}"     # gt | mixed | pred (mip only)
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="${ROOT}/outputs/experiments/libero_sf_flops_${OBJ}_${STAMP}"
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
mkdir -p "$OUT" "${ROOT}/outputs/logs"

EXTRA=(model.architecture.action_objective=flow)
if [[ "$OBJ" == "mip" ]]; then
  EXTRA=(
    model.architecture.action_objective=mip
    model.architecture.mip_num_steps=2
    "model.architecture.mip_refine_mode=${REFINE}"
  )
fi

CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE=4 MASTER_PORT="${MASTER_PORT:-29801}" \
  bash scripts/train.sh \
  dataloader=libero model=dual_system \
  "training.finetune_ckpt_path=${FOUNDATION_CKPT}" \
  training.mixed_precision=bf16 training.zero_stage=2 \
  training.use_gradient_checkpointing=true \
  training.batch_size=1 training.gradient_accumulation_steps=1 \
  training.num_epochs=null training.max_steps=6 \
  training.save_steps=0 training.save_full_states_for_resume=false \
  training.measure_flops=true \
  project.seed=42 hydra.job.chdir=false \
  "training.output_path=${OUT}" "hydra.run.dir=${OUT}/hydra" \
  project.wandb.project=null \
  "${EXTRA[@]}" \
  2>&1 | tee "${ROOT}/outputs/logs/libero_sf_flops_${OBJ}_${STAMP}.log"

echo "Look for [compute] calibrated step_tflops=... and ${OUT}/*/scaling_metrics.jsonl"
