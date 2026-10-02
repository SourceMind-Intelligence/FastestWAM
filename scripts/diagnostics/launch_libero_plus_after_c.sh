#!/usr/bin/env bash
# One-time H100 handoff: wait for the matched C run, then evaluate visual shifts.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PLUS=/root/evan/libero-plus-eval-20260926/LIBERO-plus
PLUS_PY=/root/evan/libero-plus-eval-20260926/env/bin/python
PY=/root/tengfei/envs/openwam/bin/python
EVAL="$REPO/benchmarks/libero-plus/run_all_suites.py"
CONFIG="$REPO/benchmarks/libero-plus/policy_config.yml"
CKPT_ROOT="$REPO/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16"
C_STANDARD="$REPO/outputs/libero/sf_decision_C_mip_mixed_long_suite_step42730_20260927"
OUT="$EVAN/outputs/libero-plus"
mkdir -p "$OUT"

echo "[$(date -Is)] Waiting for the paired standard-LIBERO C scheduler."
while kill -0 4066526 2>/dev/null; do sleep 30; done

# Never consume the released GPUs if the matched run failed or stopped early.
"$PY" - "$C_STANDARD/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert p.is_file(), f"Missing full-suite C summary: {p}"
x = json.loads(p.read_text())
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (40, 40, 2000), o
assert not x['missing_runs'], x['missing_runs']
print('Verified C 40/40 and 2000 trials.')
PY

# Allow server/client cleanup to release policy and rendering memory.
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

common=(
  "$EVAL"
  --ckpt-name checkpoint_step_42730.safetensors
  --server-python "$PY"
  --libero-python "$PLUS_PY"
  --libero-path "$PLUS"
  --policy-config "$CONFIG"
  --replicas-per-gpu 1
  --compile-enabled false
)
A_CKPT="$CKPT_ROOT/A_fm/2026-09-25_00-54-12"
C_CKPT="$CKPT_ROOT/C_mip_mixed/2026-09-25_00-54-11"

cd "$REPO"
echo "[$(date -Is)] Starting one-rollout EGL and policy smoke."
"$PY" "${common[@]}" --ckpt-dir "$A_CKPT" --gpus 0 --render-gpus 4 \
  --base-port 8970 --denoise-steps 2 --suites spatial --smoke \
  --output-dir "$OUT/sf_decision_A_egl_smoke_20260927"

"$PY" - "$OUT/sf_decision_A_egl_smoke_20260927/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (1, 1, 1), o
assert not x['missing_runs'], x['missing_runs']
print('Verified one-rollout EGL/policy smoke.')
PY

for label in A C; do
  if [[ "$label" == A ]]; then
    ckpt="$A_CKPT"
    denoise=2
  else
    ckpt="$C_CKPT"
    denoise=10 # The C MIP path executes two fixed action passes.
  fi
  result="$OUT/sf_decision_${label}_visual_shift_screen_20260927"
  echo "[$(date -Is)] Starting $label paired 103-task visual-shift screen."
  "$PY" "${common[@]}" --ckpt-dir "$ckpt" --gpus 0,1,2 \
    --render-gpus 4,5,7 --base-port 8970 --denoise-steps "$denoise" \
    --task-sample-ratio 0.01 --task-sample-seed 42 --output-dir "$result"
  "$PY" - "$result/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
assert not x['missing_runs'], x['missing_runs']
print('Verified 103/103 visual-shift screen.')
PY
done
echo "[$(date -Is)] Both paired visual-shift screens finished."
