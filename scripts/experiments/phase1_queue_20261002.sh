#!/usr/bin/env bash
# Phase-1 queue on h100-box (shallow-head plan, 2026-10-02): runs the phase-1 stages in order, unattended.
# Start it detached from the tree, with the paths it needs:
#   FOUNDATION_CKPT=<dir> LIBERO_DATA=<dir> LIBERO_PYTHON=<python> LIBERO_PATH=<dir> \
#     setsid nohup scripts/experiments/phase1_queue_20261002.sh all >/dev/null 2>&1 </dev/null &
# "smokes" stops after the smoke stages. Stages, each logging under logs/phase1-20261002/:
#   smoke_l      l1 and l2 smokes side by side (two optimizer steps each)
#   smoke_r1     the r1 smoke, once alpha is in the tree
#   evalsmoke_l  one LIBERO episode on each smoke checkpoint
#   train_l1, train_l2                      L1 and L2 side by side, 5 epochs each
#   eval_l1_fm10, eval_l2_mip, eval_l1_fm2  full LIBERO on the final L1 and L2 weights, all 8 GPUs
#   train_r1, train_r3, train_r4, train_r5  one after another, once alpha is in the tree
# Alpha is ALPHA_CKPT, else the path written to logs/phase1-20261002/alpha_ckpt.txt; the R stages wait for it.
# A failed smoke or training stage stops the queue; a failed eval is logged and the queue moves on.
# A training run that dies is resumed once from its latest full state, or restarted when it has none.
# Finished stages go to done.txt, so rerunning the queue picks up where it stopped. status.txt gets
# one line per event. While a file named PAUSE sits in the log dir, no new stage starts.
set -uo pipefail

log_dir=logs/phase1-20261002
libero_sh=scripts/experiments/libero_dot_pair_20261002.sh
robodojo_sh=scripts/experiments/robodojo_alpha_arms_20261002.sh
eval_sh=scripts/experiments/libero_eval_phase1_20261002.sh

status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$log_dir/status.txt"; }
is_done() { grep -qx "$1" "$log_dir/done.txt" 2>/dev/null; }
mark_done() {
  echo "$1" >> "$log_dir/done.txt"
  status "DONE $1"
}
# Run dirs are timestamped, so the last one in sort order is the newest.
latest_run_dir() { ls -d "outputs/openwam_checkpoints/$1"/*/ 2>/dev/null | sort | tail -n 1 | sed 's:/$::'; }
latest_ckpt() { ls "$1"/checkpoint_step_*.safetensors 2>/dev/null | sort -V | tail -n 1; }
has_state() { ls "$1"/accel_state_step_*/trainer_state.json > /dev/null 2>&1; }
have_eval() { [[ -n "${LIBERO_PYTHON:-}" && -n "${LIBERO_PATH:-}" ]]; }
alpha_dir() { echo "${ALPHA_CKPT:-$(cat "$log_dir/alpha_ckpt.txt" 2> /dev/null)}"; }
alpha_ready() {
  local dir
  dir="$(alpha_dir)"
  [[ -n "$dir" && -f "$dir/config.yaml" ]] && ls "$dir"/*.safetensors > /dev/null 2>&1
}

gpus_busy() {
  pgrep -f 'scripts/train.py|benchmarks/libero/run_all_suites.py' > /dev/null && return 0
  command -v nvidia-smi > /dev/null || return 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 { busy = 1 } END { exit busy ? 0 : 1 }'
}

# Hold the next stage while PAUSE exists, then until every GPU reads free twice, a minute apart.
gate() {
  local clear=0 said=""
  while [[ -e "$log_dir/PAUSE" ]]; do
    [[ -n "$said" ]] || status "PAUSED before $1"
    said=1
    sleep 60
  done
  said=""
  while ((clear < 2)); do
    if gpus_busy; then
      clear=0
      [[ -n "$said" ]] || status "WAIT $1: GPUs busy"
      said=1
    else
      clear=$((clear + 1))
    fi
    ((clear < 2)) && sleep 60
  done
  return 0
}

# train_arm <launcher> <arm> <run name>: one training run, resumed or restarted once if it dies.
train_arm() {
  local launcher="$1" arm="$2" name="$3" attempt dir
  for attempt in 1 2; do
    dir="$(latest_run_dir "$name")"
    if [[ -n "$dir" ]] && has_state "$dir"; then
      status "RUN $name resume from $dir (attempt $attempt)"
      ALPHA_CKPT="$(alpha_dir)" "$launcher" "$arm" train-resume "$dir" >> "$log_dir/$name.log" 2>&1 && return 0
    else
      status "RUN $name start (attempt $attempt)"
      ALPHA_CKPT="$(alpha_dir)" "$launcher" "$arm" train-start >> "$log_dir/$name.log" 2>&1 && return 0
    fi
    status "FAIL $name attempt $attempt exit $?"
    sleep 60
  done
  return 1
}

stop() {
  status "STOP $1"
  exit 1
}

smoke_pair() {
  local rc1=0 rc2=0 p1 p2 arm
  "$libero_sh" l1 smoke > "$log_dir/smoke_l1.log" 2>&1 &
  p1=$!
  "$libero_sh" l2 smoke > "$log_dir/smoke_l2.log" 2>&1 &
  p2=$!
  wait "$p1" || rc1=$?
  wait "$p2" || rc2=$?
  status "smoke l1 exit $rc1, l2 exit $rc2"
  ((rc1 == 0 && rc2 == 0)) || return 1
  for arm in l1 l2; do
    [[ -n "$(latest_ckpt "$(latest_run_dir "libero_dot_${arm}_20261002_smoke")")" ]] || {
      status "smoke $arm wrote no checkpoint"
      return 1
    }
  done
}

smoke_r1() {
  is_done smoke_r1 && return 0
  gate smoke_r1
  status "RUN smoke_r1 from $(alpha_dir)"
  ALPHA_CKPT="$(alpha_dir)" "$robodojo_sh" r1 smoke > "$log_dir/smoke_r1.log" 2>&1 || stop "smoke_r1 failed, see smoke_r1.log"
  mark_done smoke_r1
}

evalsmoke_pair() {
  local rc1=0 rc2=0 p1 p2
  GPUS=0 REPLICAS_PER_GPU=1 BASE_PORT=8920 "$eval_sh" smoke "$(latest_run_dir libero_dot_l1_20261002_smoke)" smoke_l1 > "$log_dir/evalsmoke_l1.log" 2>&1 &
  p1=$!
  GPUS=4 REPLICAS_PER_GPU=1 BASE_PORT=8960 STEPS=2 "$eval_sh" smoke "$(latest_run_dir libero_dot_l2_20261002_smoke)" smoke_l2 > "$log_dir/evalsmoke_l2.log" 2>&1 &
  p2=$!
  wait "$p1" || rc1=$?
  wait "$p2" || rc2=$?
  status "eval smoke l1 exit $rc1, l2 exit $rc2"
  ((rc1 == 0 && rc2 == 0))
}

# pair_arm <arm>: one arm of the LIBERO pair, as its own stage.
pair_arm() {
  is_done "train_$1" && return 0
  train_arm "$libero_sh" "$1" "libero_dot_$1_20261002" && mark_done "train_$1"
}

# eval_stage <stage> <run name> <tag> <denoise steps>
eval_stage() {
  local dir
  is_done "$1" && return 0
  gate "$1"
  dir="$(latest_run_dir "$2")"
  status "RUN $1 on $dir/$(basename "$(latest_ckpt "$dir")")"
  if STEPS="$4" "$eval_sh" full "$dir" "$3" > "$log_dir/$1.log" 2>&1; then
    mark_done "$1"
  else
    status "FAIL $1 exit $? (skipped; rerun the queue to retry)"
  fi
}

main() {
  local mode="${1:-all}" p1 p2 arm said=""
  cd "${OPENWAM_ROOT:-/root/evan/Fastest-WAM-evan}" || exit 1
  mkdir -p "$log_dir"
  exec 9> "$log_dir/queue.lock"
  flock -n 9 || {
    echo 'another phase-1 queue is running' >&2
    exit 1
  }
  : "${FOUNDATION_CKPT:?set FOUNDATION_CKPT}" "${LIBERO_DATA:?set LIBERO_DATA}"
  export FOUNDATION_CKPT LIBERO_DATA
  echo "$$" > "$log_dir/queue.pid"
  status "START queue ($mode) pid $$ at $(git rev-parse --short HEAD 2> /dev/null)"

  if ! is_done smoke_l; then
    gate smoke_l
    status "RUN smoke_l"
    smoke_pair || stop "smoke_l failed, see smoke_l1.log and smoke_l2.log"
    mark_done smoke_l
  fi
  if alpha_ready; then smoke_r1; fi
  if have_eval && ! is_done evalsmoke_l; then
    gate evalsmoke_l
    status "RUN evalsmoke_l"
    if evalsmoke_pair; then mark_done evalsmoke_l; else status "FAIL evalsmoke_l (training goes ahead; see evalsmoke_l*.log)"; fi
  fi
  if [[ "$mode" == smokes ]]; then
    status "SMOKES DONE"
    return 0
  fi

  if ! is_done train_l1 || ! is_done train_l2; then
    gate train_l
    pair_arm l1 &
    p1=$!
    sleep 30
    pair_arm l2 &
    p2=$!
    wait "$p1"
    wait "$p2"
    is_done train_l1 && is_done train_l2 || stop "train_l failed, see libero_dot_l*_20261002.log"
  fi

  if have_eval; then
    eval_stage eval_l1_fm10 libero_dot_l1_20261002 l1_fm10 10
    eval_stage eval_l2_mip libero_dot_l2_20261002 l2_mip 2
    eval_stage eval_l1_fm2 libero_dot_l1_20261002 l1_fm2 2
  else
    status "SKIP LIBERO evals: LIBERO_PYTHON or LIBERO_PATH not set"
  fi

  while ! alpha_ready; do
    [[ -n "$said" ]] || status "WAIT alpha: set ALPHA_CKPT or write its path to $log_dir/alpha_ckpt.txt"
    said=1
    sleep 600
  done
  smoke_r1
  for arm in r1 r3 r4 r5; do
    is_done "train_$arm" && continue
    gate "train_$arm"
    train_arm "$robodojo_sh" "$arm" "robodojo_alpha_${arm}_20261002" || stop "train_$arm failed, see robodojo_alpha_${arm}_20261002.log"
    mark_done "train_$arm"
  done
  status "QUEUE DONE"
}

main "$@"; exit $?
