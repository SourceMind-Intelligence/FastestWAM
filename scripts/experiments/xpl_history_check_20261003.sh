#!/usr/bin/env bash
# Offline check of the RoboDojo history adapter (benchmarks/robodojo/xpolicylab) on h100-box before R4's RoboDojo
# panel (2026-10-03). No simulator: adapter_actions.py drives the adapters through synthetic episodes on real weights.
#   G0  latency bench on R4 (history), 10 flow steps, no cache, compile off: the per-call cost the eval pays
#   G1  the same on alpha
#   G2  stock XPolicyLab adapter (fa431ec, its vendored openwam), alpha, 10 envs in one batch   } same weights and
#   G3  history adapter (this tree), alpha, 10 envs one at a time                               } inputs
#   G4  history adapter, R4, 10 envs, 224 steps (from step 150 all four history slots differ): the frames reach the
#       model, and the per-chunk time with history
#   G5  stock adapter, alpha, 1 env      } batch of one on both sides: the code difference alone
#   G6  history adapter, alpha, 1 env    }
#   G7  a 2 GB placeholder, so the phase-1 queue's GPU gate keeps R5 waiting until these finish (2 h at most)
# Waits for the latency bench to exit (it removes its PAUSE then; the queue next needs two free GPU reads 60 s apart,
# and the placeholder is up within seconds). Stage the adapter first (stage_xpolicylab.sh with the same --copy and
# --run-root), then start this from that clone, detached:
#   BENCH_PID=<pid> nohup bash scripts/experiments/xpl_history_check_20261003.sh > /dev/null 2>&1 &
# Outputs: <main tree>/outputs/xpl_check_20261003/ (report.md sums them up); start and end go to the queue's status.txt.
set -uo pipefail
src="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
main="${MAIN_TREE:-/root/evan/Fastest-WAM-evan}"
robo="${ROBODOJO:-/root/XPolicyLab/RoboDojo}"
copy="${XPL_COPY:-/root/xpl-history}"
run_root="${XPL_RUN_ROOT:-/root/xpl-history-run}"
PY="${CHECK_PYTHON:-/root/openwam-eval/env/bin/python}"
out="${CHECK_OUT:-$main/outputs/xpl_check_20261003}"
log_dir="${QUEUE_LOG_DIR:-$main/logs/phase1-20261002}"
mkdir -p "$out/bench"
note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$out/run.log"; }
status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$log_dir/status.txt"; }

commit="$(git -C "$src" rev-parse HEAD)"
staged="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["fastestwam_commit"])' "$copy/stage.json" 2> /dev/null)"
[[ "$staged" == "$commit" && -L "$run_root/XPolicyLab" ]] || { note "stage $src ($commit) with stage_xpolicylab.sh first"; exit 1; }

if [[ -n "${BENCH_PID:-}" ]]; then
  grep -q latency_bench "/proc/$BENCH_PID/cmdline" 2> /dev/null || { note "pid $BENCH_PID is not the latency bench"; exit 1; }
  note "waiting for the latency bench (pid $BENCH_PID) to exit"
  while kill -0 "$BENCH_PID" 2> /dev/null; do sleep 5; done
fi
# The queue's own test (phase1_queue_20261002.sh): never start next to a training run or an eval.
gpus_busy() {
  pgrep -f 'scripts/train.py|benchmarks/libero/run_all_suites.py' > /dev/null && return 0
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 { busy = 1 } END { exit busy ? 0 : 1 }'
}
if gpus_busy; then
  note "GPUs busy (R5 or another job already started); nothing run"
  status "XPLCHECK skipped: GPUs were busy after the latency bench"
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
hold="$out/HOLD"
touch "$hold"
CUDA_VISIBLE_DEVICES="${HOLD_GPU:-7}" "$PY" - "$hold" "${HOLD_MAX_S:-7200}" > "$out/hold.log" 2>&1 <<'PY' &
import os, sys, time
import torch
x = torch.empty(2 << 30, dtype=torch.uint8, device="cuda")
flag, deadline = sys.argv[1], time.time() + float(sys.argv[2])
print(f"holding {x.numel() >> 20} MiB on {torch.cuda.get_device_name(0)}", flush=True)
while os.path.exists(flag) and time.time() < deadline:
    time.sleep(5)
PY
holder=$!
released=""
release() {
  [[ -n "$released" ]] && return 0
  released=1
  rm -f "$hold"
  kill "$holder" 2> /dev/null
  status "XPLCHECK GPU ${HOLD_GPU:-7} released"
  note "placeholder released"
}
trap release EXIT
for _ in $(seq 60); do
  grep -q '^holding' "$out/hold.log" 2> /dev/null && break
  kill -0 "$holder" 2> /dev/null || break
  sleep 1
done
if ! grep -q '^holding' "$out/hold.log" 2> /dev/null; then
  note "the placeholder did not start (see $out/hold.log); nothing run, so R5 is not held up"
  status "XPLCHECK skipped: GPU placeholder failed"
  exit 1
fi
status "XPLCHECK start: GPU ${HOLD_GPU:-7} held so R5 waits for the history adapter check"
note "placeholder pid $holder on GPU ${HOLD_GPU:-7}"

alpha="$main/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboDojo"
r4="$(ls -d "$main"/outputs/openwam_checkpoints/robodojo_alpha_r4_20261002/*/ 2> /dev/null | sort | tail -n 1)"
r4="${r4%/}"
vendored="$robo/XPolicyLab/policy/OpenWAM/OpenWAM"
drv="$src/benchmarks/robodojo/xpolicylab/adapter_actions.py"
bench="$src/scripts/diagnostics/latency_bench_20261003.py"
note "R4 checkpoint dir: $r4 ($(ls "$r4" 2> /dev/null | grep -c safetensors) safetensors file(s))"

ram_free_gb() { awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo; }
pids=()
names=()
job() { # job <gpu> <name> <PYTHONPATH> <python args...>
  local gpu="$1" name="$2" pythonpath="$3"
  shift 3
  while (($(ram_free_gb) < ${MIN_FREE_RAM_GB:-120})); do sleep 15; done
  (cd "$main" && CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$pythonpath" timeout "${JOB_TIMEOUT_S:-2400}" "$PY" "$@") \
    > "$out/$name.log" 2>&1 &
  pids+=($!)
  names+=("$name")
  note "$name started on GPU $gpu"
  sleep "${STAGGER_S:-30}"
}
fast=(--modes fm10-nocache --compile false --probe 0 --warmup 5 --iters 50 --breakdown-iters 10)
stock=(--module XPolicyLab.policy.OpenWAM.model --openwam-root "$vendored")
ours=(--module XPolicyLab.policy.OpenWAM.model --openwam-root "$src")
job 0 bench-R4 "" "$bench" --ckpt-dir "$r4" --label R4-eval --benchmark robodojo "${fast[@]}" --out "$out/bench/R4-eval.json"
job 1 bench-alpha "" "$bench" --ckpt-dir "$alpha" --label alpha-eval --benchmark robodojo "${fast[@]}" --out "$out/bench/alpha-eval.json"
job 2 stock10 "$robo" "$drv" "${stock[@]}" --ckpt-dir "$alpha" --envs 10 --steps 128 --out "$out/stock10.npz"
job 3 ours10 "$run_root" "$drv" "${ours[@]}" --ckpt-dir "$alpha" --envs 10 --steps 128 --out "$out/ours10.npz"
job 4 r4hist "$run_root" "$drv" "${ours[@]}" --ckpt-dir "$r4" --envs 10 --steps 224 --out "$out/r4hist.npz"
job 5 stock1 "$robo" "$drv" "${stock[@]}" --ckpt-dir "$alpha" --envs 1 --steps 64 --out "$out/stock1.npz"
STAGGER_S=0 job 6 ours1 "$run_root" "$drv" "${ours[@]}" --ckpt-dir "$alpha" --envs 1 --steps 64 --out "$out/ours1.npz"

failed=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then
    note "${names[$k]}: ok"
  else
    note "${names[$k]}: FAIL exit $? (see $out/${names[$k]}.log)"
    failed=$((failed + 1))
  fi
done
release

{
  echo "# History adapter check, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "Tree $src at $commit; stage $copy/stage.json; R4 $r4; GPU $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
  echo "$failed job(s) failed"
  for name in stock10 ours10 r4hist stock1 ours1; do
    echo
    echo "## $name"
    grep -h -E '^\[Open|^step ' "$out/$name.log" 2> /dev/null | head -n 12
    sed -n '/^{/,/^}/p' "$out/$name.log" 2> /dev/null
  done
  for pair in "stock10 ours10" "stock1 ours1" "stock10 stock1" "ours10 ours1"; do
    set -- $pair
    echo
    echo "## compare $1 vs $2"
    "$PY" "$drv" --compare "$out/$1.npz" "$out/$2.npz" 2>&1 | tail -n 20
  done
  echo
  echo "## latency bench"
  "$PY" "$bench" --summarize "$out/bench" 2>&1
} > "$out/report.md"
status "XPLCHECK done: $failed job(s) failed; $out/report.md"
note "done: $failed job(s) failed"
exit $((failed > 0))
