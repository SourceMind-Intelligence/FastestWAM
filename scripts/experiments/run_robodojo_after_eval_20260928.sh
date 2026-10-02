#!/usr/bin/env bash
set -euo pipefail
cd /root/evan/Fastest-WAM-evan
log_dir=logs/robodojo-resumable-20260928
mkdir -p "$log_dir"
exec 9>"$log_dir/queue.lock"
flock -n 9 || { echo 'Another RoboDojo queue is active'; exit 1; }
status="$log_dir/status.txt"
stamp() { date -Is; }
set_status() { printf '%s %s\n' "$(stamp)" "$*" | tee "$status"; }
trap 'rc=$?; if [ "$rc" -ne 0 ]; then set_status "FAILED exit=$rc"; fi' EXIT
busy() {
  pgrep -f 'benchmarks/libero-plus/run_all_suites.py|benchmarks/libero-plus/single_eval.py|launch_plus_seed43_pair_gpu3.sh|scripts/train.py' >/dev/null && return 0
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'NR != 7 && $1 > 1000 { busy=1 } END { exit busy ? 0 : 1 }'
}
wait_free() {
  local clear=0
  while (( clear < 2 )); do
    if busy; then
      clear=0
      set_status 'QUEUED waiting for GPU evaluations and training to finish'
    else
      clear=$((clear+1))
      set_status "QUEUED GPUs clear check $clear/2"
    fi
    sleep 60
  done
}
wait_free
set_status 'SMOKE_START'
scripts/experiments/robodojo_resumable_20260928.sh smoke-start > "$log_dir/smoke-start.log" 2>&1
smoke_dir=''
for d in outputs/openwam_checkpoints/robodojo_resume_smoke_20260928/*; do
  if [[ -f "$d/accel_state_step_2/trainer_state.json" ]]; then smoke_dir="$d"; fi
done
test -n "$smoke_dir"
test -s "$smoke_dir/checkpoint_step_2.safetensors"
printf '%s\n' "$smoke_dir" > "$log_dir/smoke-run-dir.txt"
set_status "SMOKE_RESUME from $smoke_dir"
scripts/experiments/robodojo_resumable_20260928.sh smoke-resume "$smoke_dir" > "$log_dir/smoke-resume.log" 2>&1
test -f "$smoke_dir/accel_state_step_3/trainer_state.json"
test -s "$smoke_dir/checkpoint_step_3.safetensors"
grep -q '\[resume\] resumed at global_step=2' "$log_dir/smoke-resume.log"
set_status 'SMOKE_PASSED checkpoint step 2 restored and advanced to step 3'
wait_free
set_status 'TRAIN_START five epochs from prior RoboDojo step-24000 weights'
scripts/experiments/robodojo_resumable_20260928.sh train-start > "$log_dir/train.log" 2>&1 &
train_pid=$!
printf '%s\n' "$train_pid" > "$log_dir/train.pid"
for _ in $(seq 1 120); do
  for d in outputs/openwam_checkpoints/robodojo_5epoch_resumable_20260928/*; do
    if [[ -f "$d/config.yaml" ]]; then
      printf '%s\n' "$d" > "$log_dir/train-run-dir.txt"
    fi
  done
  [[ -f "$log_dir/train-run-dir.txt" ]] && break
  kill -0 "$train_pid" 2>/dev/null || break
  sleep 10
done
if [[ -f "$log_dir/train-run-dir.txt" ]]; then
  set_status "TRAIN_RUNNING pid=$train_pid run_dir=$(cat "$log_dir/train-run-dir.txt")"
fi
wait "$train_pid"
run_dir=$(cat "$log_dir/train-run-dir.txt")
shopt -s nullglob
states=("$run_dir"/accel_state_step_*/trainer_state.json)
weights=("$run_dir"/checkpoint_step_*.safetensors)
test "${#states[@]}" -gt 0
test "${#weights[@]}" -gt 0
set_status "TRAIN_COMPLETE run_dir=$run_dir"
