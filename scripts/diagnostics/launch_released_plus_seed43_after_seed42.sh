#!/usr/bin/env bash
# Use healthy GPUs 1–2 for an independent released-policy reference after seed 42.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
PLUS=/root/evan/libero-plus-eval-20260926/LIBERO-plus
PLUS_PY=/root/evan/libero-plus-eval-20260926/env/bin/python
CKPT="$REPO/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
OUT="$EVAN/outputs/libero-plus"
PREVIOUS="$OUT/released_openwam_valid_visual_shift_screen_20260928"
SCREEN="$OUT/released_openwam_valid_visual_shift_seed43_20260928"

echo "[$(date -Is)] Waiting for completed released seed-42 reference."
while kill -0 391389 2>/dev/null; do sleep 30; done
"$PY" - "$PREVIOUS/summary.json" "$OUT/sf_decision_A_visual_shift_seed43_20260928/manifest.json" <<'PY'
import json, sys
previous = json.load(open(sys.argv[1]))
o = previous['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
assert not previous['missing_runs'], previous['missing_runs']
a_manifest = json.load(open(sys.argv[2]))
a = a_manifest['resume_signature']
assert a['jobs_count'] == 103 and a_manifest['task_sample_seed'] == 43, a_manifest
PY

for attempt in $(seq 1 40); do
  busy=0
  for gpu in 1 2 5 7; do
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$used" -ge 2000 ]]; then busy=1; fi
  done
  if [[ "$busy" == 0 ]]; then break; fi
  if [[ "$attempt" == 40 ]]; then echo "Required healthy GPUs did not become free." >&2; exit 1; fi
  sleep 30
done

cd "$REPO"
common=(
  benchmarks/libero-plus/run_all_suites.py
  --ckpt-dir "$CKPT"
  --ckpt-name checkpoint_step_10690.safetensors
  --server-python "$PY"
  --libero-python "$PLUS_PY"
  --libero-path "$PLUS"
  --policy-config "$REPO/benchmarks/libero-plus/policy_config.yml"
  --gpus 1,2
  --replicas-per-gpu 1
  --render-gpus 5,7
  --base-port 9001
  --denoise-steps 2
  --compile-enabled false
  --task-sample-ratio 0.01
  --task-sample-seed 43
)
"$PY" "${common[@]}" --dry-run > "$OUT/released_seed43_dry_run_20260928.log"
grep -q '103 pending request(s)' "$OUT/released_seed43_dry_run_20260928.log"
echo "[$(date -Is)] Starting released seed-43 reference on GPUs 1–2."
"$PY" "${common[@]}" --output-dir "$SCREEN"
"$PY" - "$SCREEN" "$OUT/sf_decision_A_visual_shift_seed43_20260928/manifest.json" <<'PY'
import json, sys
from pathlib import Path
screen = Path(sys.argv[1])
s = json.loads((screen / 'summary.json').read_text())
o = s['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
assert not s['missing_runs'], s['missing_runs']
r = json.loads((screen / 'manifest.json').read_text())['resume_signature']
a = json.load(open(sys.argv[2]))['resume_signature']
for field in ('protocol_version', 'policy_config_sha256', 'effective_seed',
              'mujoco_version', 'trial_start', 'num_trials', 'jobs_sha256', 'jobs_count'):
    assert r[field] == a[field], field
print('Verified released seed-43 reference against A task list and protocol.')
PY
echo "[$(date -Is)] Released seed-43 reference complete."
