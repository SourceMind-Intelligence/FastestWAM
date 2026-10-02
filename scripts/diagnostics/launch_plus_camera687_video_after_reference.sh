#!/usr/bin/env bash
# Replay one post hoc diagnostic task after the released 103-task screen frees GPU 0.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
PLUS=/root/evan/libero-plus-eval-20260926/LIBERO-plus
PLUS_PY=/root/evan/libero-plus-eval-20260926/env/bin/python
CKPT_ROOT="$REPO/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16"
SCREEN="$EVAN/outputs/libero-plus/released_openwam_valid_visual_shift_screen_20260928"
OUT="$EVAN/outputs/libero-plus/camera687_video_20260928"
mkdir -p "$OUT"

echo "[$(date -Is)] Waiting for released reference PID 391389."
while kill -0 391389 2>/dev/null; do sleep 30; done
"$PY" - "$SCREEN/summary.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
o = x['overall']
assert (o['tasks_complete'], o['tasks_expected'], o['trials']) == (103, 103, 103), o
assert not x['missing_runs'], x['missing_runs']
PY
for attempt in $(seq 1 40); do
  used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$used" -lt 2000 ]]; then break; fi
  if [[ "$attempt" == 40 ]]; then echo "GPU 0 did not become free." >&2; exit 1; fi
  sleep 30
done

cd "$REPO"
server_pid=''
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=''
  fi
}
trap stop_server EXIT
for label in A C Released; do
  if [[ "$label" == A ]]; then
    ckpt="$CKPT_ROOT/A_fm/2026-09-25_00-54-12"
    name=checkpoint_step_42730.safetensors
    denoise=2
  elif [[ "$label" == C ]]; then
    ckpt="$CKPT_ROOT/C_mip_mixed/2026-09-25_00-54-11"
    name=checkpoint_step_42730.safetensors
    denoise=10 # C uses a fixed two-pass MIP action path.
  else
    ckpt="$REPO/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO"
    name=checkpoint_step_10690.safetensors
    denoise=2
  fi
  result="$OUT/$label"
  mkdir -p "$result"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" scripts/deploy.py \
    --ckpt-dir "$ckpt" --ckpt-name "$name" \
    --device cuda:0 --host 127.0.0.1 --port 8981 \
    --denoise-steps "$denoise" --denoise-mode sync --inference-mode sync \
    --inference-horizon 10 --compile-enabled false \
    > "$result/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for attempt in $(seq 1 240); do
    if "$PY" - <<'PY'
import json
from websockets.sync.client import connect
try:
    with connect('ws://127.0.0.1:8981', open_timeout=1, close_timeout=1,
                 ping_interval=None, proxy=None) as ws:
        ws.send(json.dumps({'type':'ping'}))
        assert json.loads(ws.recv(timeout=1)).get('type') == 'pong'
except Exception:
    raise SystemExit(1)
PY
    then ready=1; break; fi
    kill -0 "$server_pid" 2>/dev/null || { tail -n 30 "$result/server.log"; exit 1; }
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo "Policy server did not become ready." >&2; exit 1; }
  echo "[$(date -Is)] Recording camera-viewpoint task 687 for $label."
  PYTHONPATH="$PLUS:$REPO/benchmarks/libero-plus:${PYTHONPATH:-}" \
  LIBERO_PLUS_PATH="$PLUS" LIBERO_PLUS_CONFIG_ROOT="$OUT/libero-config" \
  LIBERO_CONFIG_PATH="$OUT/libero-config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 PYTHONUNBUFFERED=1 \
  WAM_PLUS_SINGLE_EVAL="$REPO/benchmarks/libero-plus/single_eval.py" \
  WAM_VIDEO_OUT="$result/head_wrist.mp4" \
  "$PLUS_PY" "$EVAN/scripts/diagnostics/record_plus_single_eval.py" \
    --config "$REPO/benchmarks/libero-plus/policy_config.yml" \
    --suite libero_spatial --task-id 687 --host 127.0.0.1 --port 8981 \
    --trial-start 0 --num-trials 1 --seed 10000 --result-dir "$result" \
    > "$result/client.log" 2>&1
  stop_server
  "$PY" - "$result" "$label" <<'PY'
import json, sys
from pathlib import Path
root, label = Path(sys.argv[1]), sys.argv[2]
x = json.loads((root / 'results.json').read_text())
assert (x['suite'], x['task_id'], x['trial_start'], x['trial_stop']) == ('libero_spatial', 687, 0, 1)
assert len(x['trials']) == 1
assert (root / 'head_wrist.mp4').stat().st_size > 100_000
print(label, x['trials'][0]['success'], x['trials'][0]['policy_steps'])
PY
done
echo "[$(date -Is)] Camera task 687 videos and outcome records complete."
