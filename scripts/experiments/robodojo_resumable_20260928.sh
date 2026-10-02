#!/usr/bin/env bash
set -euo pipefail
cd /root/evan/Fastest-WAM-evan
mode="${1:?usage: $0 smoke-start|smoke-resume|train-start|train-resume [run-dir]}"
run_dir="${2:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
export NPROC_PER_NODE=7
export NCCL_NVLS_ENABLE=0
export PATH="/root/openwam-eval/env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
base=(dataloader=robodojo model.architecture.action_objective=flow training.batch_size=16 training.gradient_accumulation_steps=1 training.zero_stage=2 training.save_full_states_for_resume=true training.keep_last_k_ckpts=2 project.seed=42)
case "$mode" in
  smoke-start)
    exec bash scripts/train.sh "${base[@]}" training.finetune_ckpt_path=outputs/openwam_checkpoints/2026-09-17_15-32-18 training.num_epochs=5 training.max_steps=2 training.lr_scheduler=null training.save_steps=2 training.output_path=outputs/openwam_checkpoints/robodojo_resume_smoke_20260928 project.wandb.run_name=robodojo-resume-smoke
    ;;
  smoke-resume)
    test -n "$run_dir" || { echo 'smoke-resume needs run-dir' >&2; exit 2; }
    exec bash scripts/train.sh "${base[@]}" training.resume_ckpt_path="$run_dir" training.num_epochs=5 training.max_steps=3 training.lr_scheduler=null training.save_steps=2 project.wandb.run_name=robodojo-resume-smoke
    ;;
  train-start)
    exec bash scripts/train.sh "${base[@]}" training.finetune_ckpt_path=outputs/openwam_checkpoints/2026-09-17_15-32-18 training.num_epochs=5 training.max_steps=null training.lr_scheduler=cosine training.save_steps=1000 training.initial_save_steps=[10] training.output_path=outputs/openwam_checkpoints/robodojo_5epoch_resumable_20260928 project.wandb.run_name=robodojo-5epoch-resumable-20260928
    ;;
  train-resume)
    test -n "$run_dir" || { echo 'train-resume needs run-dir' >&2; exit 2; }
    exec bash scripts/train.sh "${base[@]}" training.resume_ckpt_path="$run_dir" training.num_epochs=5 training.max_steps=null training.lr_scheduler=cosine training.save_steps=1000 training.initial_save_steps=[10] project.wandb.run_name=robodojo-5epoch-resumable-20260928
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
