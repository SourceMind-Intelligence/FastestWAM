#!/usr/bin/env bash
# LIBERO DoT pair (shallow-head plan, 2026-10-02): the A/C recipe on a 1-layer DoT head.
#   l1  DoT + flow matching; compare with A (deep head + flow matching, 1938/2000)
#   l2  DoT + MIP with detached self-forcing (mip_refine_mode=mixed); compare with C (1957/2000)
# Recipe as A and C: foundation init, 5 epochs, global batch 128 on 4 GPUs, seed 42.
# Match BATCH_PER_GPU x GRAD_ACCUM (default 16 x 2) and the LR to A's config.yaml.
# l1 takes GPUs 0-3 and l2 takes 4-7 by default (GPUS overrides), so the pair runs side by side.
set -euo pipefail
cd /root/evan/Fastest-WAM-evan
usage="usage: FOUNDATION_CKPT=<dir> LIBERO_DATA=<dir> $0 l1|l2 smoke|train-start|train-resume [run-dir]"
arm="${1:?$usage}"
mode="${2:?$usage}"
run_dir="${3:-}"
foundation="${FOUNDATION_CKPT:?$usage}"
data="${LIBERO_DATA:?$usage}"
case "$arm" in
  l1) gpus="${GPUS:-0,1,2,3}"; port=29501; arm_args=(model.architecture.action_objective=flow) ;;
  l2) gpus="${GPUS:-4,5,6,7}"; port=29502; arm_args=(model.architecture.action_objective=mip model.architecture.mip_refine_mode=mixed) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$gpus"
export NPROC_PER_NODE=4
export MASTER_PORT="$port"
export NCCL_NVLS_ENABLE=0
export PATH="/root/openwam-eval/env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
name="libero_dot_${arm}_20261002"
base=(dataloader=libero dataloader.dataset_dir="$data" model.architecture.variant=dot model.architecture.dot_num_action_layers=1 ${arm_args[@]+"${arm_args[@]}"} training.batch_size="${BATCH_PER_GPU:-16}" training.gradient_accumulation_steps="${GRAD_ACCUM:-2}" training.zero_stage=2 training.keep_last_k_ckpts=2 training.num_epochs=5 project.seed=42)
schedule=(training.max_steps=null training.lr_scheduler=cosine training.save_steps=5000 training.save_full_states_for_resume=true)
case "$mode" in
  smoke)
    exec bash scripts/train.sh "${base[@]}" training.finetune_ckpt_path="$foundation" training.max_steps=2 training.lr_scheduler=null training.save_steps=0 training.save_full_states_for_resume=false training.output_path="outputs/openwam_checkpoints/${name}_smoke" project.wandb.run_name="${name//_/-}-smoke"
    ;;
  train-start)
    exec bash scripts/train.sh "${base[@]}" "${schedule[@]}" training.finetune_ckpt_path="$foundation" training.output_path="outputs/openwam_checkpoints/$name" project.wandb.run_name="${name//_/-}"
    ;;
  train-resume)
    test -n "$run_dir" || { echo 'train-resume needs run-dir' >&2; exit 2; }
    exec bash scripts/train.sh "${base[@]}" "${schedule[@]}" training.resume_ckpt_path="$run_dir" project.wandb.run_name="${name//_/-}"
    ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
