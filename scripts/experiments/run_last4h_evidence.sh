#!/usr/bin/env bash
# Four independent evaluation processes; each has one policy and one EGL GPU.
# No training changes. Matched first-ten-trial comparisons can be extracted
# from the archived 50-trial baselines. Set BUDGET_SECONDS before launch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SERVER_PYTHON="${SERVER_PYTHON:-/root/tengfei/envs/openwam/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/root/fasteval-libero/envs/libero/bin/python}"
export LIBERO_PATH="${LIBERO_PATH:-/root/fasteval-libero/LIBERO}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1
export NCCL_NVLS_ENABLE=0 NCCL_RAS_ENABLE=0
unset CUDA_VISIBLE_DEVICES
BUDGET_SECONDS="${BUDGET_SECONDS:-13200}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${ROOT}/outputs/last4h_${STAMP}"
mkdir -p "$OUT"
printf '%s\n' "$OUT" > outputs/last4h_latest.txt
run_arm() {
  local name="$1" gpu="$2" render="$3" port="$4" dir="$5" ckpt="$6" steps="$7" seed="$8"
  GPUS="$gpu" RENDER_GPUS="$render" BASE_PORT="$port" \
    timeout --signal=INT --kill-after=60 "$BUDGET_SECONDS" \
    bash scripts/experiments/run_libero_eval_stable.sh "$dir" "$ckpt" "$OUT/$name" \
    --denoise-steps "$steps" --denoise-mode sync --num-trials 10 --seed "$seed" --diagnostic-protocol \
    > "$OUT/$name.log" 2>&1 &
  pids+=("$!")
  printf '%s\t%s\n' "$name" "$!" >> "$OUT/pids.tsv"
}
pids=()
DOT4="$ROOT/outputs/experiments/libero_dot_sf_2026-10-13_17-44-16/FM_flow/2026-10-14_14-39-10"
DOT8="$ROOT/outputs/experiments/libero_weekend_autoresearch_2026-10-16_22-07-41/DOT8_FM_flow/2026-10-17_08-33-36"
MOT="$ROOT/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16/A_fm/2026-09-25_00-54-12"
run_arm dot4_fm_nfe1_seed42 0 4 18820 "$DOT4" checkpoint_step_17092.safetensors 1 42
run_arm dot8_fm_nfe1_seed42 1 5 18830 "$DOT8" checkpoint_step_17092.safetensors 1 42
run_arm mot_fm_nfe1_seed42 2 6 18840 "$MOT" checkpoint_step_42730.safetensors 1 42
run_arm dot4_fm_nfe2_seed43 3 7 18850 "$DOT4" checkpoint_step_17092.safetensors 2 43
rc=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then code=0; else code=$?; rc=1; fi
  printf '%s\t%s\n' "$pid" "$code" >> "$OUT/exit_codes.tsv"
done
exit "$rc"
