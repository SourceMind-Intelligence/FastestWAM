#!/usr/bin/env bash
# Maintain a reviewable local archive until the resource deadline; no push.
set -u
cd "$(dirname "$0")/../.."
DEADLINE="${EVIDENCE_DEADLINE_EPOCH:-$(date -d '2026-10-18 23:40:00 +0800' +%s)}"
while [[ $(date +%s) -lt "$DEADLINE" ]]; do
  /root/tengfei/envs/openwam/bin/python scripts/experiments/collect_last4h_evidence.py \
    > outputs/logs/last4h_collection.log 2>&1
  sleep 60
done
/root/tengfei/envs/openwam/bin/python scripts/experiments/collect_last4h_evidence.py \
  > outputs/logs/last4h_collection.log 2>&1
