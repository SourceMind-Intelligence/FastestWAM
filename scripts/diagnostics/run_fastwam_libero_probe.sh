#!/usr/bin/env bash
set -euo pipefail
mode="${1:?readout mode: all, none, single:N, drop:N}"
gpu="${2:?physical GPU index}"
task_file="${3:?task list path}"
output_dir="${4:?new output directory}"
trials="${5:-1}"

repo=/root/evan/Fastest-WAM-evan
official="$repo/third_party/FastWAM-official"
assets="$repo/assets/fastwam_official"
case "$mode" in
  all|none|single:[0-9]*|drop:[0-9]*) ;;
  *) echo "Invalid readout mode: $mode" >&2; exit 2 ;;
esac
[[ "$gpu" =~ ^[0-7]$ && "$trials" =~ ^[0-9]+$ ]] || exit 2
test -s "$task_file"
test -s "$assets/libero_uncond_2cam224.pt"
test -s "$assets/libero_uncond_2cam224_dataset_stats.json"
test ! -e "$output_dir" || { echo "Output exists: $output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
printf 'source_commit=7faa71108368fbb3b6885649f112af607427a2d4\ncheckpoint_revision=8eaceeb24c3cc92ff2a9c9a9d266a4941b836705\nreadout=%s\ngpu=%s\ntrials=%s\n' "$mode" "$gpu" "$trials" > "$output_dir/probe_metadata.txt"
export PYTHONPATH="$official/runtime_site:$official/src:/root/fasteval-libero/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_ROOT="$assets/libero_config"
export LIBERO_CONFIG_PATH="$assets/libero_config"
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/FastWAM/checkpoints
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$gpu"
export MUJOCO_EGL_DEVICE_ID="$gpu"
export FASTWAM_VIDEO_READOUT="$mode"
export PYTHONUNBUFFERED=1
cd "$official"
exec /root/tengfei/envs/openwam/bin/python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  "ckpt=$assets/libero_uncond_2cam224.pt" \
  "EVALUATION.dataset_stats_path=$assets/libero_uncond_2cam224_dataset_stats.json" \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.compile_action_infer=false \
  "EVALUATION.num_trials=$trials" \
  "EVALUATION.output_dir=$output_dir" \
  "MULTIRUN.task_file=$task_file" \
  MULTIRUN.num_gpus=1
