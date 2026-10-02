#!/usr/bin/env bash
set -euo pipefail
cd /root/evan/Fastest-WAM-evan
mode="${1:?usage: $0 smoke|train-start|train-resume [run-dir]}"
run_dir="${2:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export NCCL_NVLS_ENABLE=0
export PATH="/root/openwam-eval/env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
base=(dataloader=robodojo model.architecture.action_objective=flow training.video_xm_k=2 training.video_xm_mix=0.25 training.batch_size=16 training.gradient_accumulation_steps=1 training.zero_stage=2 training.keep_last_k_ckpts=2 project.seed=42)
case "$mode" in
  smoke)
    exec bash scripts/train.sh "${base[@]}" training.finetune_ckpt_path=outputs/openwam_checkpoints/2026-09-17_15-32-18 training.num_epochs=5 training.max_steps=2 training.lr_scheduler=null training.save_steps=0 training.save_full_states_for_resume=false training.output_path=outputs/openwam_checkpoints/robodojo_video_xm8_smoke_20260928 project.wandb.run_name=robodojo-video-xm8-smoke
    ;;
  train-start)
    exec bash scripts/train.sh "${base[@]}" training.finetune_ckpt_path=outputs/openwam_checkpoints/2026-09-17_15-32-18 training.num_epochs=5 training.max_steps=null training.lr_scheduler=cosine training.save_steps=1000 training.initial_save_steps=[10] training.save_full_states_for_resume=true training.output_path=outputs/openwam_checkpoints/robodojo_video_xm8_20260928 project.wandb.run_name=robodojo-video-xm8-20260928
    ;;
  train-resume)
    test -n "$run_dir" || { echo 'train-resume needs run-dir' >&2; exit 2; }
    exec bash scripts/train.sh "${base[@]}" training.resume_ckpt_path="$run_dir" training.num_epochs=5 training.max_steps=null training.lr_scheduler=cosine training.save_steps=1000 training.initial_save_steps=[10] training.save_full_states_for_resume=true project.wandb.run_name=robodojo-video-xm8-20260928
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
