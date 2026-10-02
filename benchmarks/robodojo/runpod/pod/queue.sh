#!/usr/bin/env bash
# Pod side, inside tmux: take (task, seed) cells from the shared queue until none is left, then stop this pod.
#   queue.sh [--out DIR] [--queue FILE] [--grace SECONDS]
# The queue is --queue (default <out>/queue.txt), one "task seed" per line, longest first. A cell is claimed by creating the directory
# <out>/claims/<task>__s<seed> (mkdir is atomic on the volume), so pods can join at any time and a smaller fleet only
# runs longer. A cell whose runner fails keeps its claim and gets <out>/failed/<cell>; rerun it by removing both.
# Spend guard: <out>/claim_limit holds the number of cells that may be claimed in total. A pod that finds the limit
# reached stops itself instead of taking another cell, so an unattended fleet cannot spend past what was approved;
# raise the number from the Mac to release more cells.
set -uo pipefail
R=${R:-/workspace/oeh-campaign}
OUT=$R/fastestwam-baseline; grace=60; queue=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2;;
    --queue) queue=$2; shift 2;;
    --grace) grace=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
here=$(cd "$(dirname "$0")" && pwd)
pod=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
mkdir -p "$OUT/claims" "$OUT/failed" "$OUT/pods"
queue=${queue:-$OUT/queue.txt}
[ -f "$OUT/eval.env" ] && . "$OUT/eval.env"
plog=$OUT/pods/${pod:-$(hostname)}.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$plog"; }
log "pod ${pod:-?} host $(hostname) gpu $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) joins the queue"
n=0
while read -r task seed; do
  [ -n "${task:-}" ] || continue
  cell=${task}__s${seed}
  [ -e "$OUT/claims/$cell" ] && continue
  limit=$(cat "$OUT/claim_limit" 2>/dev/null || echo 0)
  [ "$(ls "$OUT/claims" | wc -l)" -lt "$limit" ] || { log "claim limit $limit reached"; break; }
  mkdir "$OUT/claims/$cell" 2>/dev/null || continue
  if [ "$(ls "$OUT/claims" | wc -l)" -gt "$limit" ]; then rmdir "$OUT/claims/$cell"; log "claim limit $limit reached"; break; fi
  echo "${pod:-$(hostname)} $(date -u +%FT%TZ)" > "$OUT/claims/$cell/owner"
  log "cell $cell start"
  bash "$here/cell.sh" "$task" "$seed" --out "$OUT" --max-hours "${MAX_CELL_HOURS:-5}" --no-stop < /dev/null; rc=$?
  [ "$rc" = 0 ] || touch "$OUT/failed/$cell"
  log "cell $cell exit $rc"
  n=$((n + 1))
  [ "$rc" -ge 90 ] && [ "$rc" -le 93 ] && { log "this pod cannot run cells (exit $rc): releasing $cell"; rm -rf "$OUT/claims/$cell" "$OUT/failed/$cell"; break; }
done < "$queue"
log "queue empty for this pod after $n cell(s); pod stops in ${grace}s"
sleep "$grace"
if [ -e "$OUT/keep-pod" ]; then log "kept: $OUT/keep-pod exists"; exit 0; fi
key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p')
if [ -n "$pod" ] && [ -n "$key" ]; then
  log "stopping this pod ($pod)"
  curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $key" https://api.runpod.io/graphql \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$pod\\\"}) { id desiredStatus } }\"}"; echo
else
  log "no pod id or pod key in this container: stop the pod from the Mac"
fi
