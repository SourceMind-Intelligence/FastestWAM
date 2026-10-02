#!/usr/bin/env bash
# Audit the independent seed-43 A/C visual-shift screen after both runs finish.
set -euo pipefail

EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
OUT="$EVAN/outputs/libero-plus"

echo "[$(date -Is)] Waiting for independent A/C seed-43 screen."
while kill -0 394715 2>/dev/null; do sleep 30; done
"$PY" "$EVAN/scripts/diagnostics/aggregate_visual_shift_screen.py" \
  --a "$OUT/sf_decision_A_visual_shift_seed43_20260928" \
  --c "$OUT/sf_decision_C_visual_shift_seed43_20260928" \
  --sample-plan "$OUT/sample_plan_seed43_20260928.json" \
  --output "$OUT/paired_A_C_visual_shift_seed43_20260928.json"
echo "[$(date -Is)] Saved verified seed-43 paired screen."
