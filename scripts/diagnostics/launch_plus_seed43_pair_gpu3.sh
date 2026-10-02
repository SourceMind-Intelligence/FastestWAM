#!/usr/bin/env bash
# Independent 1% LIBERO-Plus A/C sample on the free healthy GPU 3.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
PLUS=/root/evan/libero-plus-eval-20260926/LIBERO-plus
PLUS_PY=/root/evan/libero-plus-eval-20260926/env/bin/python
CKPT_ROOT="$REPO/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16"
OUT="$EVAN/outputs/libero-plus"

cd "$REPO"
common=(
  benchmarks/libero-plus/run_all_suites.py
  --ckpt-name checkpoint_step_42730.safetensors
  --server-python "$PY"
  --libero-python "$PLUS_PY"
  --libero-path "$PLUS"
  --policy-config "$REPO/benchmarks/libero-plus/policy_config.yml"
  --gpus 3
  --replicas-per-gpu 1
  --render-gpus 3
  --base-port 8990
  --compile-enabled false
  --task-sample-ratio 0.01
  --task-sample-seed 43
)

"$PY" "${common[@]}" \
  --ckpt-dir "$CKPT_ROOT/A_fm/2026-09-25_00-54-12" \
  --denoise-steps 2 --dry-run > "$OUT/seed43_pair_dry_run_20260928.log"
grep -q '103 pending request(s)' "$OUT/seed43_pair_dry_run_20260928.log"
echo "[$(date -Is)] Starting A on independent seed-43 103-task screen."
"$PY" "${common[@]}" \
  --ckpt-dir "$CKPT_ROOT/A_fm/2026-09-25_00-54-12" \
  --denoise-steps 2 \
  --output-dir "$OUT/sf_decision_A_visual_shift_seed43_20260928"
"$PY" - "$OUT/sf_decision_A_visual_shift_seed43_20260928/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
assert not x['missing_runs'], x['missing_runs']
PY

echo "[$(date -Is)] Starting C on the same seed-43 task set."
"$PY" "${common[@]}" \
  --ckpt-dir "$CKPT_ROOT/C_mip_mixed/2026-09-25_00-54-11" \
  --denoise-steps 10 \
  --output-dir "$OUT/sf_decision_C_visual_shift_seed43_20260928"
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
signatures = []
for label in ('A', 'C'):
    run = root / f'sf_decision_{label}_visual_shift_seed43_20260928'
    s = json.loads((run / 'summary.json').read_text())
    o = s['overall']
    assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
    assert not s['missing_runs'], s['missing_runs']
    signatures.append(json.loads((run / 'manifest.json').read_text())['resume_signature'])
for field in ('protocol_version', 'policy_config_sha256', 'effective_seed',
              'mujoco_version', 'trial_start', 'num_trials', 'jobs_sha256', 'jobs_count'):
    assert signatures[0][field] == signatures[1][field], field
print('Verified matched A/C seed-43 screens.')
PY
echo "[$(date -Is)] Seed-43 paired visual-shift screens finished."
