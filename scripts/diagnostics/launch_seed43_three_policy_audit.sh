#!/usr/bin/env bash
# CPU-only raw-result audit after both seed-43 policy launchers finish.
set -euo pipefail

EVAN=/root/evan/Fastest-WAM-evan
OUT="$EVAN/outputs/libero-plus"
PY=/root/tengfei/envs/openwam/bin/python

echo "[$(date -Is)] Waiting for A/C and released seed-43 launchers."
while kill -0 394715 2>/dev/null || kill -0 416214 2>/dev/null; do sleep 30; done
"$PY" "$EVAN/scripts/diagnostics/aggregate_visual_shift_screen.py" \
  --a "$OUT/sf_decision_A_visual_shift_seed43_20260928" \
  --c "$OUT/sf_decision_C_visual_shift_seed43_20260928" \
  --released "$OUT/released_openwam_valid_visual_shift_seed43_20260928" \
  --sample-plan "$OUT/sample_plan_seed43_20260928.json" \
  --output "$OUT/paired_A_C_released_seed43_20260928.json"
echo "[$(date -Is)] Seed-43 three-policy raw-result audit complete."
