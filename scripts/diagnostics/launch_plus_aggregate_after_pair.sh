#!/usr/bin/env bash
# Verify raw 103-task A/C LIBERO-Plus screens, then include released reference.
set -euo pipefail

EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
OUT="$EVAN/outputs/libero-plus"
AUDIT="$EVAN/scripts/diagnostics/aggregate_visual_shift_screen.py"
PLAN="$OUT/sample_plan_20260927.json"
A="$OUT/sf_decision_A_visual_shift_screen_20260927"
C="$OUT/sf_decision_C_visual_shift_screen_20260927"
RELEASED="$OUT/released_openwam_visual_shift_screen_20260927"

echo "[$(date -Is)] Waiting for matched A/C LIBERO-Plus screens."
while kill -0 4098380 2>/dev/null; do sleep 30; done
"$PY" "$AUDIT" --a "$A" --c "$C" --sample-plan "$PLAN" \
  --output "$OUT/paired_A_C_visual_shift_screen_20260927.json"
echo "[$(date -Is)] Saved paired A/C screen audit."

echo "[$(date -Is)] Waiting for released-policy reference."
while kill -0 4102524 2>/dev/null; do sleep 30; done
if [[ -f "$RELEASED/summary.json" ]]; then
  "$PY" "$AUDIT" --a "$A" --c "$C" --released "$RELEASED" \
    --sample-plan "$PLAN" \
    --output "$OUT/paired_A_C_released_visual_shift_screen_20260927.json"
  echo "[$(date -Is)] Saved three-policy screen audit."
else
  echo "Released-policy summary is missing; keeping the verified A/C audit." >&2
fi
