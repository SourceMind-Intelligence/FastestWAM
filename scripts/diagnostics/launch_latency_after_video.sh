#!/usr/bin/env bash
# Use GPU 3 after the Object 03 diagnostic releases it.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
C_STANDARD="$REPO/outputs/libero/sf_decision_C_mip_mixed_long_suite_step42730_20260927"

while kill -0 4099506 2>/dev/null; do sleep 30; done
"$PY" - "$C_STANDARD/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert p.is_file(), p
x = json.loads(p.read_text())
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (40, 40, 2000), o
assert not x['missing_runs'], x['missing_runs']
PY
for attempt in $(seq 1 40); do
  used=$(nvidia-smi --id=3 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -lt 2000 ]]; then break; fi
  if [[ "$attempt" == 40 ]]; then
    echo "GPU 3 did not become free within 20 minutes." >&2
    exit 1
  fi
  sleep 30
done
cd "$REPO"
"$PY" "$EVAN/scripts/diagnostics/probe_paired_server_latency.py"
