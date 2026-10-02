#!/usr/bin/env bash
# LIBERO eval of a phase-1 run (shallow-head plan, 2026-10-02) on its latest checkpoint, with the
# canonical protocol A and C were scored on: 4 suites x 10 tasks x 50 trials, seed 42.
#   smoke  one trial of the first task, to check the policy server and client on a new architecture
#   full   the whole benchmark; rerunning the same tag resumes from the results already written
# STEPS sets the flow-matching denoise steps (default 10). A MIP checkpoint always takes its 2 passes.
# Results land in outputs/libero/phase1_20261002/<tag>/summary.json. EVAL_ARGS adds scheduler flags.
set -euo pipefail
cd /root/evan/Fastest-WAM-evan
usage="usage: LIBERO_PYTHON=<python> LIBERO_PATH=<dir> $0 smoke|full <run-dir> <tag>"
mode="${1:?$usage}"
run_dir="${2:?$usage}"
tag="${3:?$usage}"
export LIBERO_PYTHON="${LIBERO_PYTHON:?$usage}" LIBERO_PATH="${LIBERO_PATH:?$usage}"
run_dir="$(cd "$run_dir" && pwd)"
ckpt="$(ls "$run_dir"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -1)"
test -n "$ckpt" || { echo "no checkpoint in $run_dir" >&2; exit 2; }
export PATH="/root/openwam-eval/env/bin:$PATH"
# The LIBERO client may run from another user's install; never write bytecode caches into it.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export SERVER_PYTHON=/root/openwam-eval/env/bin/python
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}" REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-2}" BASE_PORT="${BASE_PORT:-8920}"
export OUTPUT_DIR="$PWD/outputs/libero/phase1_20261002/$tag"
read -r -a eval_args <<< "${EVAL_ARGS:-}"
args=(--denoise-steps "${STEPS:-10}" ${eval_args[@]+"${eval_args[@]}"})
case "$mode" in
  smoke) args+=(--smoke) ;;
  full) ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac
exec bash benchmarks/libero/run_eval.sh "$run_dir" "$(basename "$ckpt")" "${args[@]}"
