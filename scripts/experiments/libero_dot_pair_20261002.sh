#!/usr/bin/env bash
# LIBERO DoT pair (shallow-head plan, 2026-10-02): the A/C recipe on a 1-layer DoT head.
#   l1     DoT + flow matching; compare with A (deep head + flow matching, 1938/2000)
#   l2     DoT + MIP with detached self-forcing (mip_refine_mode=mixed); compare with C (1957/2000)
#   l1e10  l1 for 10 epochs, the plan's follow-up when l1 lands more than 20/2000 below A
# Recipe as A and C (their RUN_INFO.txt): foundation init, 5 epochs, seed 42, global batch 128 as
# 4 GPUs x 8 per GPU x 4 accumulation steps, and the repo's LR schedule (1e-4, cosine down to 1e-6).
# BATCH_PER_GPU, GRAD_ACCUM and LR override those; EXTRA_ARGS adds Hydra overrides (space-separated).
# l1 and l1e10 take GPUs 0-3 and l2 takes 4-7 by default. GPUS overrides that; on 1, 2 or 8 cards the
# accumulation steps change so the global batch stays 128.
# smoke takes two optimizer steps and keeps those weights, so the LIBERO eval can be smoke-tested on them.
set -euo pipefail
cd /root/evan/Fastest-WAM-evan
usage="usage: FOUNDATION_CKPT=<dir> LIBERO_DATA=<dir> $0 l1|l2|l1e10 smoke|train-start|train-resume [run-dir]"
arm="${1:?$usage}"
mode="${2:?$usage}"
run_dir="${3:-}"
foundation="${FOUNDATION_CKPT:?$usage}"
data="${LIBERO_DATA:?$usage}"
epochs=5
case "$arm" in
  l1) gpus="${GPUS:-0,1,2,3}"; port=29501; arm_args=(model.architecture.action_objective=flow) ;;
  l2) gpus="${GPUS:-4,5,6,7}"; port=29502; arm_args=(model.architecture.action_objective=mip model.architecture.mip_refine_mode=mixed) ;;
  l1e10) gpus="${GPUS:-0,1,2,3}"; port=29501; epochs=10; arm_args=(model.architecture.action_objective=flow) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
nproc="$(awk -F, '{ print NF }' <<< "$gpus")"
if [[ -z "${GRAD_ACCUM:-}" ]] && ((16 % nproc != 0)); then
  echo "GPUS=$gpus: use 1, 2, 4 or 8 cards, or set GRAD_ACCUM, so the global batch stays 128" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpus"
export NPROC_PER_NODE="$nproc"
export MASTER_PORT="$port"
export NCCL_NVLS_ENABLE=0
export PATH="/root/openwam-eval/env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
name="libero_dot_${arm}_20261002"
accum="${GRAD_ACCUM:-$((16 / nproc))}"
lr_args=()
if [[ -n "${LR:-}" ]]; then lr_args=(training.learning_rate="$LR"); fi
read -r -a extra_args <<< "${EXTRA_ARGS:-}"
base=(dataloader=libero dataloader.dataset_dir="$data" model.architecture.variant=dot model.architecture.dot_num_action_layers=1 ${arm_args[@]+"${arm_args[@]}"} training.batch_size="${BATCH_PER_GPU:-8}" training.gradient_accumulation_steps="$accum" training.zero_stage=2 training.keep_last_k_ckpts=2 training.num_epochs="$epochs" project.seed=42 ${lr_args[@]+"${lr_args[@]}"} ${extra_args[@]+"${extra_args[@]}"})
schedule=(training.max_steps=null training.lr_scheduler=cosine training.save_steps=5000 training.save_full_states_for_resume=true)
case "$mode" in
  smoke)
    # max_steps counts micro-batches. save_steps above it skips the periodic save but keeps the final weights.
    exec bash scripts/train.sh "${base[@]}" training.finetune_ckpt_path="$foundation" training.max_steps=$((2 * accum)) training.lr_scheduler=null training.save_steps=1000 training.save_full_states_for_resume=false training.output_path="outputs/openwam_checkpoints/${name}_smoke" project.wandb.run_name="${name//_/-}-smoke"
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
