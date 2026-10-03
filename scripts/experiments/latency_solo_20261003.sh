#!/usr/bin/env bash
# Re-measure the latency bench's rows one checkpoint at a time, after R5 (2026-10-03).
# In latency_bench_20261003.sh the eight jobs overlapped, and their torch.compile warm-ups slowed the jobs
# being timed (L1 at 2 steps: p50 77.0 ms in parallel, 41.7 ms alone). This runs the same bench with the same
# settings on one GPU, one job at a time:
#   A, C, L1, L2, alpha, L1-nocompile    as in latency_bench_20261003.sh
#   alpha-eval, R4-eval                  RoboDojo eval settings, as in xpl_history_check_20261003.sh
# Start it while R5 trains, from a checkout of this branch: the bench and openwam come from that checkout,
# the checkpoints, the queue's PAUSE and the outputs from the main tree.
#   nohup setsid bash scripts/experiments/latency_solo_20261003.sh > /dev/null 2>&1 < /dev/null &
# It sets the queue's PAUSE (unless already there) so the queue holds before its next stage, waits until every
# GPU reads free twice a minute apart, measures, then removes the PAUSE it set, whatever happens.
# Reports: <main tree>/outputs/latency_20261003_solo/<label>.json and summary.md; start and end go to status.txt.
set -uo pipefail
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${MAIN_TREE:-/root/evan/Fastest-WAM-evan}"
log_dir="${QUEUE_LOG_DIR:-logs/phase1-20261002}"
out="${SOLO_OUT:-outputs/latency_20261003_solo}"
mkdir -p "$out"
PY="${LATENCY_PYTHON:-/root/openwam-eval/env/bin/python}"
bench="$src/scripts/diagnostics/latency_bench_20261003.py"
note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$out/run.log"; }
status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$log_dir/status.txt"; }

created_pause=0
if [[ ! -e "$log_dir/PAUSE" ]]; then
  touch "$log_dir/PAUSE"
  created_pause=1
  status "PAUSE set by the solo latency re-run; it removes it when done"
  note "PAUSE set"
fi
release() {
  if [[ "$created_pause" == 1 ]]; then
    rm -f "$log_dir/PAUSE"
    status "PAUSE removed by the solo latency re-run"
    note "PAUSE removed"
  fi
}
trap release EXIT

# The queue's own test (phase1_queue_20261002.sh).
gpus_busy() {
  pgrep -f 'scripts/train.py|benchmarks/libero/run_all_suites.py' > /dev/null && return 0
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 { busy = 1 } END { exit busy ? 0 : 1 }'
}
deadline=$(($(date +%s) + ${WAIT_MAX_S:-86400}))
clear=0
note "waiting for every GPU to be free (tree $src at $(git -C "$src" rev-parse --short HEAD 2> /dev/null))"
while ((clear < 2)); do
  if gpus_busy; then clear=0; else clear=$((clear + 1)); fi
  if ((clear < 2)); then
    (($(date +%s) < deadline)) || { note "GPUs still busy after ${WAIT_MAX_S:-86400} s; not measuring"; exit 1; }
    sleep "${POLL_S:-60}"
  fi
done

export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

latest_run_dir() { ls -d "outputs/openwam_checkpoints/$1"/*/ 2> /dev/null | sort | tail -n 1 | sed 's:/$::'; }
tengfei=/root/tengfei/FastestWAM/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16
a_dir="${LATENCY_A_DIR:-$tengfei/A_fm/2026-09-25_00-54-12}"
c_dir="${LATENCY_C_DIR:-$tengfei/C_mip_mixed/2026-09-25_00-54-11}"
l1_dir="$(latest_run_dir libero_dot_l1_20261002)"
l2_dir="$(latest_run_dir libero_dot_l2_20261002)"
r4_dir="$(latest_run_dir robodojo_alpha_r4_20261002)"
alpha_dir=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboDojo

failed=0
solo() { # solo <label> <benchmark> <ckpt dir> [bench args...]
  local label="$1" benchmark="$2" dir="$3"
  shift 3
  if [[ ! -f "$dir/config.yaml" ]]; then
    note "$label: no config.yaml in '$dir', skipped"
    return 0
  fi
  note "$label start: $dir"
  if CUDA_VISIBLE_DEVICES="${SOLO_GPU:-0}" timeout "${JOB_TIMEOUT_S:-1200}" "$PY" "$bench" --ckpt-dir "$dir" \
    --label "$label" --benchmark "$benchmark" --device cuda:0 --out "$out/$label.json" "$@" > "$out/$label.log" 2>&1; then
    note "$label: ok"
  else
    note "$label: FAIL exit $? (see $out/$label.log)"
    failed=$((failed + 1))
  fi
}

status "LATENCY-SOLO start: one job at a time on GPU ${SOLO_GPU:-0}"
eval_settings=(--modes fm10-nocache --compile false --probe 0 --warmup 5 --iters 50 --breakdown-iters 10)
solo A libero "$a_dir"
solo C libero "$c_dir"
solo L1 libero "$l1_dir" --probe 0
solo L2 libero "$l2_dir" --probe 0
solo alpha robodojo "$alpha_dir"
solo L1-nocompile libero "$l1_dir" --compile false --probe 0
solo alpha-eval robodojo "$alpha_dir" "${eval_settings[@]}"
solo R4-eval robodojo "$r4_dir" "${eval_settings[@]}"

"$PY" "$bench" --summarize "$out" > "$out/summary.md" || note "summary failed"
status "LATENCY-SOLO done: $failed job(s) failed; $out/summary.md"
note "done: $failed job(s) failed"
exit $((failed > 0))
