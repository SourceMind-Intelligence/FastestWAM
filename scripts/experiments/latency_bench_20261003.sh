#!/usr/bin/env bash
# Inference latency and GPU memory of the deep and shallow action heads (2026-10-03, Evan's ask):
# scripts/diagnostics/latency_bench_20261003.py on each checkpoint, one H100 per checkpoint, all at once.
#   A        30-layer joint head, flow matching (LIBERO)      fm1 fm2 fm10 fm10-nocache, joint vs video probe
#   C        30-layer joint head, MIP (LIBERO)                mip2, joint vs video probe
#   L1       1-layer DoT head, flow matching (LIBERO)         fm1 fm2 fm10
#   L2       1-layer DoT head, MIP (LIBERO)                   mip2 mip1
#   alpha    released OpenWAM-α, 30-layer joint, flow (RoboDojo inputs)   fm1 fm2 fm10 fm10-nocache, probe
#   alpha-mip  α's weights through the MIP path, the deep-MIP compute on RoboDojo inputs   mip2
#   L1-nocompile, A-nocompile  the same with torch.compile off, to show what compile does on each path
# Then L1 at 2 steps once more on its own, to check the parallel runs did not slow each other.
# Reports: outputs/latency_20261003/<label>.json and summary.md.
#
# Start it any time while R4 trains: it sets logs/phase1-20261002/PAUSE (unless already there) so the
# queue holds before its next stage, waits until every GPU is free, measures, then removes the PAUSE
# it set, whatever happens. A and C are read from Tengfei's tree and only read.
set -euo pipefail
cd "${OPENWAM_ROOT:-/root/evan/Fastest-WAM-evan}"
log_dir=logs/phase1-20261002
out=outputs/latency_20261003
mkdir -p "$out"
PY="${LATENCY_PYTHON:-/root/openwam-eval/env/bin/python}"
bench=scripts/diagnostics/latency_bench_20261003.py

note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$out/run.log"; }
status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$log_dir/status.txt"; }

created_pause=0
if [[ ! -e "$log_dir/PAUSE" ]]; then
  touch "$log_dir/PAUSE"
  created_pause=1
  status "PAUSE set by the latency bench; it removes it when done"
  note "PAUSE set"
fi
release() {
  if [[ "$created_pause" == 1 ]]; then
    rm -f "$log_dir/PAUSE"
    status "PAUSE removed by the latency bench"
    note "PAUSE removed"
  fi
}
trap release EXIT

gpus_busy() {
  pgrep -f 'scripts/train.py|benchmarks/libero/run_all_suites.py' > /dev/null && return 0
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 { busy = 1 } END { exit busy ? 0 : 1 }'
}
deadline=$(($(date +%s) + ${WAIT_MAX_S:-43200}))
clear=0
note "waiting for every GPU to be free"
while ((clear < 2)); do
  if gpus_busy; then clear=0; else clear=$((clear + 1)); fi
  if ((clear < 2)); then
    (($(date +%s) < deadline)) || { note "GPUs still busy after ${WAIT_MAX_S:-43200} s; not measuring"; exit 1; }
    sleep 60
  fi
done

export PATH="/root/openwam-eval/env/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

latest_run_dir() { ls -d "outputs/openwam_checkpoints/$1"/*/ 2> /dev/null | sort | tail -n 1 | sed 's:/$::'; }
tengfei=/root/tengfei/FastestWAM/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16
a_dir="${LATENCY_A_DIR:-$tengfei/A_fm/2026-09-25_00-54-12}"
c_dir="${LATENCY_C_DIR:-$tengfei/C_mip_mixed/2026-09-25_00-54-11}"
l1_dir="$(latest_run_dir libero_dot_l1_20261002)"
l2_dir="$(latest_run_dir libero_dot_l2_20261002)"
alpha_dir=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboDojo

# Leave free host memory for each load before starting the next one.
ram_free_gb() { awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo; }
pids=()
names=()
job() { # job <gpu> <label> <benchmark> <ckpt dir> [bench args...]
  local gpu="$1" label="$2" benchmark="$3" dir="$4"
  shift 4
  if [[ ! -f "$dir/config.yaml" ]]; then
    note "$label: no config.yaml in '$dir', skipped"
    return 0
  fi
  while (($(ram_free_gb) < ${MIN_FREE_RAM_GB:-120})); do sleep 15; done
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$bench" --ckpt-dir "$dir" --label "$label" --benchmark "$benchmark" \
    --device cuda:0 --out "$out/$label.json" "$@" > "$out/$label.log" 2>&1 &
  pids+=($!)
  names+=("$label")
  note "$label started on GPU $gpu: $dir"
  sleep "${STAGGER_S:-30}"
}

status "LATENCY start"
job 0 A libero "$a_dir"
job 1 C libero "$c_dir"
job 2 L1 libero "$l1_dir"
job 3 L2 libero "$l2_dir"
job 4 alpha robodojo "$alpha_dir"
job 5 alpha-mip robodojo "$alpha_dir" --force-objective mip --modes mip2 --probe 0
job 6 L1-nocompile libero "$l1_dir" --compile false --probe 0
job 7 A-nocompile libero "$a_dir" --compile false --probe 0

failed=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then
    note "${names[$k]}: ok"
  else
    note "${names[$k]}: FAIL exit $? (see $out/${names[$k]}.log)"
    failed=$((failed + 1))
  fi
done

pids=()
names=()
STAGGER_S=0 job 0 L1-solo libero "$l1_dir" --modes fm2 --iters 50 --breakdown-iters 5 --probe 0
for k in "${!pids[@]}"; do
  wait "${pids[$k]}" && note "${names[$k]}: ok" || { note "${names[$k]}: FAIL"; failed=$((failed + 1)); }
done

"$PY" "$bench" --summarize "$out" > "$out/summary.md" || note "summary failed"
status "LATENCY done: $failed job(s) failed; $out/summary.md"
note "done: $failed job(s) failed"
exit $((failed > 0))
