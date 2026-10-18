#!/usr/bin/env bash
# Six-microbatch probes, no weight/optimizer checkpoint writes.
# FLOPs are measured per rank at batch=1, not retrospective full-run budgets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH="/root/tengfei/envs/openwam/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${PROBE_GPUS:-4,5,6,7}" NPROC_PER_NODE=4
export NCCL_NVLS_ENABLE=0 NCCL_RAS_ENABLE=0 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FOUNDATION="$ROOT/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"
OUT="$ROOT/outputs/flops_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > outputs/flops_audit_latest.txt
for arm in dot4_flow dot4_mip mot_flow mot_mip; do
  model=dual_system; strict=true
  [[ "$arm" == dot4_* ]] && model=dual_system_dot && strict=false
  objective="${arm##*_}"
  MASTER_PORT=29871 timeout --signal=TERM --kill-after=60 1200 bash scripts/train.sh \
    dataloader=libero "model=$model" "model.architecture.action_objective=$objective" \
    "training.finetune_ckpt_path=$FOUNDATION" "training.finetune_strict_load=$strict" \
    training.mixed_precision=bf16 training.zero_stage=2 training.use_gradient_checkpointing=true \
    training.batch_size=1 training.gradient_accumulation_steps=1 \
    training.num_epochs=null training.max_steps=6 training.debug=false \
    training.save_steps=0 training.save_full_states_for_resume=false training.milestone_save_steps=null \
    training.measure_flops=true project.seed=42 hydra.job.chdir=false \
    "training.output_path=$OUT/$arm" "hydra.run.dir=$OUT/$arm/hydra" \
    project.wandb.project=null > "$OUT/$arm.log" 2>&1 || {
      printf '%s failed\n' "$arm" >> "$OUT/failures.txt"
    }
done
