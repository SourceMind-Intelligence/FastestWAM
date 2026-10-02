#!/usr/bin/env bash
# RoboDojo arms continued from OpenWAM-alpha (shallow-head plan, 2026-10-02):
#   r1  pipeline check, flow matching, 2k steps
#   r3  control, flow matching, 12k steps (keeps the 10k and 12k checkpoints)
#   r4  history: 4 past frames, the episode's first frame plus frames 6, 4 and 2 s back (stride 50 at 25 Hz), 10k steps
#   r5  video Forward XM, K=2, mix 0.25, 10k steps
# 8 GPUs x batch 16 = global batch 128. Every arm uses the same LR schedule
# (500-step warmup, then constant LR) so the arms compare at equal steps.
# ALPHA_CKPT is alpha's checkpoint dir in this tree; LR overrides the default 2e-5.
# EXTRA_ARGS adds Hydra overrides (space-separated).
set -euo pipefail
cd /root/evan/Fastest-WAM-evan
usage="usage: ALPHA_CKPT=<dir> $0 r1|r3|r4|r5 smoke|train-start|train-resume [run-dir]"
arm="${1:?$usage}"
mode="${2:?$usage}"
run_dir="${3:-}"
alpha="${ALPHA_CKPT:?$usage}"
lr="${LR:-2e-5}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export NCCL_NVLS_ENABLE=0
export PATH="/root/openwam-eval/env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
save_steps=2000
case "$arm" in
  r1) steps=2000; save_steps=1000; arm_args=() ;;
  r3) steps=12000; arm_args=() ;;
  r4) steps=10000; arm_args=(dataloader.history_num_frames=4 dataloader.history_stride=50 dataloader.history_include_first_frame=true) ;;
  r5) steps=10000; arm_args=(training.video_xm_k=2 training.video_xm_mix=0.25) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
name="robodojo_alpha_${arm}_20261002"
read -r -a extra_args <<< "${EXTRA_ARGS:-}"
warmup_ratio=$(awk -v s="$steps" 'BEGIN { print 500 / s }')
base=(dataloader=robodojo model.architecture.action_objective=flow ${arm_args[@]+"${arm_args[@]}"} training.batch_size=16 training.gradient_accumulation_steps=1 training.zero_stage=2 training.keep_last_k_ckpts=2 training.num_epochs=null project.seed=42 ${extra_args[@]+"${extra_args[@]}"})
schedule=(training.max_steps="$steps" training.learning_rate="$lr" training.lr_scheduler=cosine training.lr_min_ratio=1.0 training.warmup_ratio="$warmup_ratio" training.save_steps="$save_steps" training.save_full_states_for_resume=true)
case "$mode" in
  smoke)
    exec bash scripts/train.sh "${base[@]}" training.finetune_ckpt_path="$alpha" training.max_steps=2 training.lr_scheduler=null training.save_steps=0 training.save_full_states_for_resume=false training.output_path="outputs/openwam_checkpoints/${name}_smoke" project.wandb.run_name="${name//_/-}-smoke"
    ;;
  train-start)
    exec bash scripts/train.sh "${base[@]}" "${schedule[@]}" training.finetune_ckpt_path="$alpha" training.output_path="outputs/openwam_checkpoints/$name" project.wandb.run_name="${name//_/-}"
    ;;
  train-resume)
    test -n "$run_dir" || { echo 'train-resume needs run-dir' >&2; exit 2; }
    exec bash scripts/train.sh "${base[@]}" "${schedule[@]}" training.resume_ckpt_path="$run_dir" project.wandb.run_name="${name//_/-}"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
