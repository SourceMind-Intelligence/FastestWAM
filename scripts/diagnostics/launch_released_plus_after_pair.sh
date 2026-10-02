#!/usr/bin/env bash
# One-time released-checkpoint reference after the matched A/C shift screens.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
PLUS=/root/evan/libero-plus-eval-20260926/LIBERO-plus
PLUS_PY=/root/evan/libero-plus-eval-20260926/env/bin/python
OUT="$EVAN/outputs/libero-plus"

echo "[$(date -Is)] Waiting for paired A/C visual-shift screens."
while kill -0 4098380 2>/dev/null; do sleep 30; done
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
signatures = []
for label in ('A', 'C'):
    run = root / f'sf_decision_{label}_visual_shift_screen_20260927'
    s = json.loads((run / 'summary.json').read_text())
    o = s['overall']
    assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
    assert not s['missing_runs'], s['missing_runs']
    m = json.loads((run / 'manifest.json').read_text())
    signatures.append(m['resume_signature'])
for key in ('jobs_sha256', 'jobs_count', 'policy_config_sha256', 'protocol_version',
            'mujoco_version', 'trial_start', 'num_trials', 'effective_seed'):
    assert signatures[0][key] == signatures[1][key], key
print('Verified matched 103-task A/C screens.')
PY

for attempt in $(seq 1 40); do
  if nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
     awk -F, '($1+0==0 || $1+0==1 || $1+0==2 || $1+0==4 || $1+0==5 || $1+0==7) && $2+0>=2000 {busy=1} END {exit busy}'; then
    break
  fi
  if [[ "$attempt" == 40 ]]; then
    echo "Required GPUs did not become free within 20 minutes." >&2
    exit 1
  fi
  sleep 30
done

cd "$REPO"
echo "[$(date -Is)] Starting released-checkpoint 103-task reference screen."
"$PY" benchmarks/libero-plus/run_all_suites.py \
  --ckpt-dir "$EVAN/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO" \
  --ckpt-name checkpoint_step_10690.safetensors \
  --server-python "$PY" --libero-python "$PLUS_PY" --libero-path "$PLUS" \
  --policy-config "$REPO/benchmarks/libero-plus/policy_config.yml" \
  --gpus 0,1,2 --replicas-per-gpu 1 --render-gpus 4,5,7 \
  --base-port 8970 --denoise-steps 2 --compile-enabled false \
  --task-sample-ratio 0.01 --task-sample-seed 42 \
  --output-dir "$OUT/released_openwam_visual_shift_screen_20260927"
"$PY" - "$OUT/released_openwam_visual_shift_screen_20260927/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
assert not x['missing_runs'], x['missing_runs']
print('Verified released policy 103/103 visual-shift screen.')
PY
echo "[$(date -Is)] Released reference screen finished."
