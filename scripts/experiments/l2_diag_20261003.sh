#!/usr/bin/env bash
# L2 diagnostics (2026-10-03): where does the 1-layer DoT head with MIP (L2) lose to flow matching (L1)?
# Four LIBERO tasks hold 152 of L2's 293-trial gap to L1: goal 9 and Long 3, 4 and 8. Each gets 50
# trials (the pinned protocol) in three runs, all twelve task runs at once, one task per policy server:
#   l2_mip    L2, both MIP passes (unused dims as trained), recorded, both passes traced
#   l2_pass1  L2, the first pass alone (OPENWAM_DOT_MIP_PASSES=1), recorded, traced
#   l1_fm2    L1, 2 flow steps, recorded
# Recordings (frames, actions, end-effector states per trial; benchmarks/libero/episode_recorder.py),
# traces (openwam/model/architectures/dual_system/dot.py) and results land in
# outputs/libero/phase1_20261002/diag_20261003/<run>/<suite>_task<NN>/. The slowest task sets the
# wall time, about 1 to 1.5 h. Summarize with scripts/diagnostics/l2_diag_summary_20261003.py.
# Run it from the phase-1 tree once R4 has finished, with logs/phase1-20261002/PAUSE holding the queue,
# and remove PAUSE afterwards. While it runs, the queue also waits, since it sees run_all_suites.py.
# LIBERO_PYTHON and LIBERO_PATH come from the environment, else from logs/phase1-20261002/queue.env.
set -euo pipefail
cd "${OPENWAM_ROOT:-/root/evan/Fastest-WAM-evan}"
log_dir=logs/phase1-20261002
out=outputs/libero/phase1_20261002/diag_20261003

queue_env() { grep "^$1=" "$log_dir/queue.env" 2> /dev/null | tail -n 1 | cut -d= -f2- | tr -d "\"'"; }
LIBERO_PYTHON="${LIBERO_PYTHON:-$(queue_env LIBERO_PYTHON)}"
LIBERO_PATH="${LIBERO_PATH:-$(queue_env LIBERO_PATH)}"
[[ -n "$LIBERO_PYTHON" && -n "$LIBERO_PATH" ]] || {
  echo "set LIBERO_PYTHON and LIBERO_PATH (not found in $log_dir/queue.env either)" >&2
  exit 2
}
if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 { busy = 1 } END { exit busy ? 0 : 1 }'; then
  echo "a GPU is in use; run this once R4 has finished and the queue is held by $log_dir/PAUSE" >&2
  exit 1
fi

latest_run_dir() { ls -d "outputs/openwam_checkpoints/$1"/*/ 2> /dev/null | sort | tail -n 1 | sed 's:/$::'; }
latest_ckpt() { ls "$1"/checkpoint_step_*.safetensors 2> /dev/null | sort -V | tail -n 1; }
l2_dir="$(latest_run_dir libero_dot_l2_20261002)"
l1_dir="$(latest_run_dir libero_dot_l1_20261002)"
for dir in "$l2_dir" "$l1_dir"; do
  [[ -n "$dir" && -n "$(latest_ckpt "$dir")" ]] || { echo "no checkpoint for L1 or L2 under outputs/openwam_checkpoints" >&2; exit 2; }
done

export LIBERO_PYTHON LIBERO_PATH
export PATH="/root/openwam-eval/env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export SERVER_PYTHON=/root/openwam-eval/env/bin/python
export OPENWAM_LIBERO_RECORD_EVERY=2

# task <run> <run dir> <suite> <task id> <gpu> <port> [VAR=value ...]: one task, one policy server.
task() {
  local run="$1" dir="$2" suite="$3" id="$4" gpu="$5" port="$6"
  shift 6
  local dest="$PWD/$out/$run/${suite}_task$(printf %02d "$id")"
  mkdir -p "$dest"
  env "$@" GPUS="$gpu" REPLICAS_PER_GPU=1 BASE_PORT="$port" OUTPUT_DIR="$dest" \
    OPENWAM_LIBERO_RECORD_DIR="$dest/record" \
    bash benchmarks/libero/run_eval.sh "$PWD/$dir" "$(basename "$(latest_ckpt "$dir")")" \
    --denoise-steps 2 --suites "$suite" --task-ids "$id" > "$dest/run.log" 2>&1
}

tasks=("goal 9" "long 3" "long 4" "long 8")
pids=()
names=()
for i in "${!tasks[@]}"; do
  read -r suite id <<< "${tasks[$i]}"
  task l2_mip "$l2_dir" "$suite" "$id" "$i" $((8920 + 10 * i)) \
    OPENWAM_DOT_MIP_PASSES=2 OPENWAM_DOT_MIP_UNUSED_DIMS=anchor \
    OPENWAM_DOT_MIP_TRACE_DIR="$PWD/$out/l2_mip/${suite}_task$(printf %02d "$id")/trace" &
  pids+=($!)
  names+=("l2_mip $suite $id")
  task l2_pass1 "$l2_dir" "$suite" "$id" $((4 + i)) $((8960 + 10 * i)) \
    OPENWAM_DOT_MIP_PASSES=1 \
    OPENWAM_DOT_MIP_TRACE_DIR="$PWD/$out/l2_pass1/${suite}_task$(printf %02d "$id")/trace" &
  pids+=($!)
  names+=("l2_pass1 $suite $id")
  task l1_fm2 "$l1_dir" "$suite" "$id" "$i" $((9000 + 10 * i)) &
  pids+=($!)
  names+=("l1_fm2 $suite $id")
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) started ${#pids[@]} task runs: L2 $l2_dir, L1 $l1_dir" | tee -a "$log_dir/l2_diag.log"

failed=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then
    status=ok
  else
    status="FAIL exit $?"
    failed=1
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${names[$k]}: $status" | tee -a "$log_dir/l2_diag.log"
done
exit "$failed"
