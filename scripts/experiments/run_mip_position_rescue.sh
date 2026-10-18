#!/usr/bin/env bash
# Matched action-only continuation: historical MIP-GT checkpoint, scale 0 vs .1.
# Runs on 4-7 after FLOPs probes finish; separate from policy GPUs 0-3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PATH="/root/tengfei/envs/openwam/bin:$PATH"
export NCCL_NVLS_ENABLE=0 NCCL_RAS_ENABLE=0 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SERVER_PYTHON=/root/tengfei/envs/openwam/bin/python
export LIBERO_PYTHON=/root/fasteval-libero/envs/libero/bin/python
export LIBERO_PATH=/root/fasteval-libero/LIBERO
OUT="$ROOT/outputs/mip_position_rescue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > outputs/mip_position_rescue_latest.txt
while tmux has-session -t last4h_flops 2>/dev/null; do sleep 10; done
CKPT="$ROOT/outputs/experiments/libero_weekend_autoresearch_2026-10-16_22-07-41/DOT4_MIP_gt_from_foundation/2026-10-17_18-21-37"
FREEZE='[video_backbone.text_encoder,video_backbone.reason1,video_backbone.vae,video_backbone.image_encoder,video_backbone.video_encoder,video_backbone.dit]'
run_arm() {
  local name="$1" scale="$2" gpus="$3" port="$4" policy_gpu="$5" render_gpu="$6" eval_port="$7"
  if CUDA_VISIBLE_DEVICES="$gpus" NPROC_PER_NODE=2 MASTER_PORT="$port" \
    timeout --signal=TERM --kill-after=60 4800 bash scripts/train.sh \
      dataloader=libero model=dual_system_dot \
      "training.finetune_ckpt_path=$CKPT" training.finetune_strict_load=true \
      model.architecture.action_objective=mip model.architecture.mip_refine_mode=gt \
      model.architecture.detach_bridge=true "model.architecture.action_position_scale=$scale" \
      "model.freeze=$FREEZE" training.action_lr=1e-4 training.learning_rate=1e-4 \
      training.batch_size=4 training.gradient_accumulation_steps=2 training.dataset_num_workers=4 \
      training.num_epochs=null training.max_steps=1500 training.debug=false \
      training.mixed_precision=bf16 training.zero_stage=2 training.use_gradient_checkpointing=true \
      training.save_steps=750 training.keep_last_k_ckpts=1 training.save_full_states_for_resume=false \
      training.milestone_save_steps=null training.measure_flops=false \
      project.seed=42 project.wandb.project=null hydra.job.chdir=false \
      "training.output_path=$OUT/$name" "hydra.run.dir=$OUT/$name/hydra" \
      > "$OUT/$name.train.log" 2>&1; then
    run=$(find "$OUT/$name" -name checkpoint_step_1500.safetensors -print -quit)
    if [[ -n "$run" ]]; then
      GPUS="$policy_gpu" RENDER_GPUS="$render_gpu" BASE_PORT="$eval_port" \
        timeout --signal=INT --kill-after=60 5400 \
        bash scripts/experiments/run_libero_eval_stable.sh "$(dirname "$run")" "$(basename "$run")" \
        "$OUT/${name}_eval" --denoise-steps 2 --denoise-mode sync \
        --diagnostic-protocol --num-trials 5 --seed 42 > "$OUT/$name.eval.log" 2>&1
    else
      echo "missing final checkpoint $name" >> "$OUT/failures.txt"; return 1
    fi
  else
    echo "training failed $name" >> "$OUT/failures.txt"; return 1
  fi
}
run_arm no_position 0.0 4,5 29881 4 5 18860 & p0=$!
run_arm sinusoidal_position 0.1 6,7 29882 6 7 18870 & p1=$!
rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
printf '%s\n' "$rc" > "$OUT/exit_code.txt"
exit "$rc"
