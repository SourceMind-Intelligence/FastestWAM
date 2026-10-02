#!/usr/bin/env bash
# Mac side. Run a RoboDojo eval from a config: plan it, draw L40S pods on an allowed driver, start one queue runner on
# each. Pods take (task, seed) cells from a shared queue, longest first, and stop themselves when it is empty.
#   launch.sh <config.env> [KEY=VALUE ...]          print the plan (cells, pods, cost); starts nothing
#   launch.sh <config.env> [KEY=VALUE ...] --go     start it (this spends money: about $1.09 per pod-hour)
# Examples
#   launch.sh configs/openwam-alpha_single.env TASKS=stack_bowls --go        one task, seed 0, 1 pod
#   launch.sh configs/openwam-alpha_full.env --go                            42 tasks x 3 seeds, 24 pods
#   launch.sh configs/openwam-alpha_full.env TASKS=memory PODS=6 --go        one dimension
#   launch.sh configs/openwam-alpha_full.env PODS=8 --go                     later: add 8 pods to the same queue
# Finished cells under OUT are never run again; a cell left half-done by a dead pod is released when no pod of this
# eval is running, and RoboDojo resumes it from its own manifest (same run id).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
go=0; args=()
for a in "$@"; do [ "$a" = --go ] && go=1 || args+=("$a"); done
[ ${#args[@]} -ge 1 ] || { sed -n 2,13p "$0"; exit 2; }
stage=$(mktemp -d -t fw-launch.XXXXXX)
python3 "$HERE/plan.py" "${args[@]}" --write "$stage" || exit 2
[ $go = 1 ] || { echo "plan only: add --go to start it"; exit 0; }
source "$HERE/_lib.sh"   # the RunPod helpers are needed only from here on
load_config "${args[@]}"
launch=$(date -u +%Y%m%d-%H%M%S); cells=$(wc -l < "$stage/queue.txt" | tr -d ' ')
[ "$PODS" -gt "$cells" ] && PODS=$cells
runpodctl user 2>/dev/null | grep -E "clientBalance|currentSpendPerHr" | tr -d '\n'; echo

# First pod of the launch: scripts, queue and eval.env onto the volume, stale claims released, spend guard set.
prepare() {
  local id=$1 others=$2 f
  pod_run "$id" "mkdir -p $OUT/bin $OUT/queues $OUT/claims $OUT/failed $OUT/pods $OUT/cells" || return 1
  for f in cell.sh queue.sh; do pod_pipe "$id" "$HERE/pod/$f" "cat > $OUT/bin/$f" || return 1; done
  pod_pipe "$id" "$stage/queue.txt" "cat > $OUT/queues/$launch.txt" || return 1
  pod_pipe "$id" "$stage/eval.env" "cat > $OUT/eval.env.$launch" || return 1
  pod_run "$id" "set -e; cd $OUT
    if [ -f eval.env ] && ! diff <(grep -E '^(CKPT_DIR|CKPT_LABEL|POLICY_DIR|RUN_PREFIX)=' eval.env) <(grep -E '^(CKPT_DIR|CKPT_LABEL|POLICY_DIR|RUN_PREFIX)=' eval.env.$launch); then
      echo 'REFUSED: $OUT already holds results of another checkpoint or policy; use a new OUT'; rm eval.env.$launch; exit 9; fi
    mv eval.env.$launch eval.env
    . ./eval.env; [ -f \"\$CKPT_DIR/config.yaml\" ] || { echo \"REFUSED: no config.yaml in \$CKPT_DIR\"; exit 9; }
    if [ '$others' = 0 ]; then for c in claims/*/; do n=\$(basename \$c); [ -f cells/\$n/exit_code ] && [ ! -e failed/\$n ] || { [ -d \$c ] && rm -rf \$c failed/\$n && echo released \$n; }; done; fi
    limit=$CLAIM_LIMIT; [ \$limit = all ] && limit=$cells
    echo \$(( \$(ls claims | wc -l) + limit )) > claim_limit
    echo \"prepared: queue $launch ($cells cells), \$(ls claims | wc -l) cell(s) already claimed or done, claim limit \$(cat claim_limit)\""
}
start_runner() {
  pod_run "$1" "which tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1); tmux new -d -s queue 'bash $OUT/bin/queue.sh --out $OUT --queue $OUT/queues/$launch.txt 2>&1 | tee -a /root/queue.log'; sleep 5; tmux has-session -t queue"
}
others=$(eval_pods | wc -l | tr -d ' ')
good=0; draw=0; prepared=0; maxdraws=${MAXDRAWS:-$((PODS * 3 + 3))}
while [ $good -lt "$PODS" ] && [ $draw -lt "$maxdraws" ]; do
  draw=$((draw + 1)); label=fw-$NAME-$(date -u +%H%M%S)
  out=$(OEH_CREATE_TRIES=${OEH_CREATE_TRIES:-3} bash "$SKILL_DIR/scripts/create_pod.sh" "$label" 2>&1 | grep "^pod ")
  id=$(echo "$out" | awk '{print $2}')
  if [ -z "$id" ]; then
    # A pod that was created but never booted is not reported as a pod: delete it, it bills while it exists.
    for stray in $(pod list 2>/dev/null | awk -v l="-$label-" 'index($0, l) { print $1 }'); do
      runpodctl pod delete "$stray" > /dev/null 2>&1 && log "draw $draw: deleted $stray (created, never booted)"
    done
    log "draw $draw: no pod (out of stock or an API error)"; continue
  fi
  d=$(echo "$out" | sed -n 's/.*L40S, \([0-9.]*\).*/\1/p')
  case " $DRIVERS " in
    *" $d "*) ;;
    *) log "draw $draw: $id driver ${d:-unreadable} is not in DRIVERS ($DRIVERS): deleting"; runpodctl pod delete "$id" > /dev/null 2>&1; continue;;
  esac
  if [ $prepared = 0 ]; then
    msg=$(prepare "$id" "$others" 2>&1); rc=$?; echo "$msg" | tail -5
    if [ $rc != 0 ]; then log "prepare failed on $id: stopping it, nothing started"; pod stop "$id" > /dev/null; exit 1; fi
    prepared=1
  fi
  if start_runner "$id"; then good=$((good + 1)); log "draw $draw: $id driver $d: queue runner started ($good/$PODS)"
  else log "draw $draw: $id runner NOT started: stopping it"; pod stop "$id" > /dev/null; fi
done
log "done: $good of $PODS pod(s) started after $draw draw(s). Watch: bash status.sh ${args[0]}   Results: bash collect.sh ${args[0]}"
[ $good -gt 0 ]
