#!/usr/bin/env bash
# Run the verified paired 40-task audit immediately after C finishes.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
ROOT="$REPO/outputs/libero"
A="$ROOT/sf_decision_A_fm_2step_all_suites_step42730_20260927"
C="$ROOT/sf_decision_C_mip_mixed_long_suite_step42730_20260927"
OUT="$EVAN/outputs/paired_A_C_full_suite_summary_20260927.json"

echo "[$(date -Is)] Waiting for matched C scheduler."
while kill -0 4066526 2>/dev/null; do sleep 30; done
"$PY" - "$C/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert p.is_file(), p
x = json.loads(p.read_text())
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (40, 40, 2000), o
assert not x['missing_runs'], x['missing_runs']
PY
"$PY" "$EVAN/scripts/diagnostics/aggregate_paired_full_suite.py" \
  "$A" "$C" --output "$OUT"
echo "[$(date -Is)] Saved $OUT."
