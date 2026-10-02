#!/usr/bin/env bash
# Retry the released reference from the validated Tengfei checkpoint, not the
# incomplete Evan transfer. Keep the failed attempt intact for diagnosis.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
PLUS=/root/evan/libero-plus-eval-20260926/LIBERO-plus
PLUS_PY=/root/evan/libero-plus-eval-20260926/env/bin/python
CKPT="$REPO/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
OUT="$EVAN/outputs/libero-plus"
AUDIT="$EVAN/scripts/diagnostics/aggregate_visual_shift_screen.py"
PLAN="$OUT/sample_plan_20260927.json"
SMOKE="$OUT/released_openwam_valid_smoke_20260928"
SCREEN="$OUT/released_openwam_valid_visual_shift_screen_20260928"

"$PY" - "$CKPT/checkpoint_step_10690.safetensors" <<'PY'
import sys
from pathlib import Path
from safetensors import safe_open
p = Path(sys.argv[1])
assert p.stat().st_size == 24813767464, p
with safe_open(str(p), framework='pt', device='cpu') as handle:
    assert len(handle.keys()) == 2089
print('Validated released checkpoint header and size.')
PY

cd "$REPO"
common=(
  benchmarks/libero-plus/run_all_suites.py
  --ckpt-dir "$CKPT"
  --ckpt-name checkpoint_step_10690.safetensors
  --server-python "$PY"
  --libero-python "$PLUS_PY"
  --libero-path "$PLUS"
  --policy-config "$REPO/benchmarks/libero-plus/policy_config.yml"
  --replicas-per-gpu 1
  --denoise-steps 2
  --compile-enabled false
)

echo "[$(date -Is)] Running released-checkpoint EGL/policy smoke."
"$PY" "${common[@]}" --gpus 0 --render-gpus 4 --base-port 8970 \
  --suites spatial --smoke --output-dir "$SMOKE"
"$PY" - "$SMOKE/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (1, 1, 1), o
assert not x['missing_runs'], x['missing_runs']
print('Validated released-checkpoint smoke.')
PY

echo "[$(date -Is)] Running released-checkpoint 103-task visual-shift reference."
"$PY" "${common[@]}" --gpus 0,1,2 --render-gpus 4,5,7 \
  --base-port 8970 --task-sample-ratio 0.01 --task-sample-seed 42 \
  --output-dir "$SCREEN"

"$PY" "$AUDIT" \
  --a "$OUT/sf_decision_A_visual_shift_screen_20260927" \
  --c "$OUT/sf_decision_C_visual_shift_screen_20260927" \
  --released "$SCREEN" --sample-plan "$PLAN" \
  --output "$OUT/paired_A_C_released_valid_visual_shift_screen_20260928.json"
echo "[$(date -Is)] Released reference and three-policy audit finished."
