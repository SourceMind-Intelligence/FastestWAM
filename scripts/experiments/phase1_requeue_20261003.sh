#!/usr/bin/env bash
# Rerun the phase-1 queue once the running one exits, so stages added after it started (the 10-epoch
# L1, 2026-10-03) still run unattended. The rerun skips every stage in done.txt. It only happens when
# the running queue got through train_r5; a queue that stopped on a failure is left for a person.
# Start it detached from the tree; it takes the queue's paths from logs/phase1-20261002/queue.env:
#   setsid nohup scripts/experiments/phase1_requeue_20261003.sh >/dev/null 2>&1 </dev/null &
set -u
cd "${OPENWAM_ROOT:-/root/evan/Fastest-WAM-evan}" || exit 1
log_dir=logs/phase1-20261002
queue=scripts/experiments/phase1_queue_20261002.sh

status() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$log_dir/status.txt"; }

test -f "$log_dir/queue.env" || {
  echo "no $log_dir/queue.env" >&2
  exit 2
}
status "WAIT requeue: rerun the queue once the running one exits"
# The running queue holds queue.lock for its whole life.
until flock -n "$log_dir/queue.lock" true; do sleep 300; done
if ! grep -qx train_r5 "$log_dir/done.txt" 2> /dev/null; then
  status "SKIP requeue: the queue stopped before train_r5"
  exit 1
fi
set -a
# shellcheck disable=SC1091
source "$log_dir/queue.env"
set +a
status "RUN requeue"
exec "$queue" all
