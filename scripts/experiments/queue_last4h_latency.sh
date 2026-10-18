#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
# Avoid contention and do not send requests to an active evaluation server.
while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ') -gt 2000 ]]; do sleep 10; done
OUT="outputs/latency_audit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
printf '%s\n' "$PWD/$OUT" > outputs/latency_audit_latest.txt
for pair in \
 'dot4:outputs/experiments/libero_dot_sf_2026-10-13_17-44-16/FM_flow/2026-10-14_14-39-10/checkpoint_step_17092.safetensors' \
 'dot8:outputs/experiments/libero_weekend_autoresearch_2026-10-16_22-07-41/DOT8_FM_flow/2026-10-17_08-33-36/checkpoint_step_17092.safetensors' \
 'mot_fm:outputs/experiments/libero_sf_suite_2026-09-25_00-52-16/A_fm/2026-09-25_00-54-12/checkpoint_step_42730.safetensors'; do
 name="${pair%%:*}"; checkpoint="${pair#*:}"
 /root/tengfei/envs/openwam/bin/python scripts/experiments/benchmark_checkpoint_latency.py \
   --checkpoint "$checkpoint" --output "$OUT/$name.json" > "$OUT/$name.log" 2>&1
done
