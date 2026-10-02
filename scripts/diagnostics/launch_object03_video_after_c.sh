#!/usr/bin/env bash
# One-time diagnostic of the matched Object 03 divergence on GPU 3.
set -euo pipefail

REPO=/root/tengfei/FastestWAM
EVAN=/root/evan/Fastest-WAM-evan
PY=/root/tengfei/envs/openwam/bin/python
LIBERO_PY=/root/fasteval-libero/envs/libero/bin/python
LIBERO=/root/fasteval-libero/LIBERO
CKPT_ROOT="$REPO/outputs/experiments/libero_sf_suite_2026-09-25_00-52-16"
C_STANDARD="$REPO/outputs/libero/sf_decision_C_mip_mixed_long_suite_step42730_20260927"
OUT="$EVAN/outputs/libero-object03-video-20260927"
mkdir -p "$OUT"

echo "[$(date -Is)] Waiting for paired C completion before Object 03 video capture."
while kill -0 4066526 2>/dev/null; do sleep 30; done
"$PY" - "$C_STANDARD/summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert p.is_file(), f"Missing C summary: {p}"
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

cp "$REPO/benchmarks/libero/policy_config.yml" "$OUT/policy_config_video.yml"
printf '\nsave_videos: true\nvideo_fps: 10\n' >> "$OUT/policy_config_video.yml"
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
for label in A C; do
  if [[ "$label" == A ]]; then
    ckpt="$CKPT_ROOT/A_fm/2026-09-25_00-54-12"
    denoise=2
  else
    ckpt="$CKPT_ROOT/C_mip_mixed/2026-09-25_00-54-11"
    denoise=10 # C MIP executes two fixed action passes.
  fi
  result="$OUT/${label}_trials_000_001"
  mkdir -p "$result"
  CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 "$PY" scripts/deploy.py \
    --ckpt-dir "$ckpt" --ckpt-name checkpoint_step_42730.safetensors \
    --device cuda:0 --host 127.0.0.1 --port 8980 \
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
    with connect('ws://127.0.0.1:8980', open_timeout=1, close_timeout=1,
                 ping_interval=None, proxy=None) as ws:
        ws.send(json.dumps({'type':'ping'}))
        assert json.loads(ws.recv(timeout=1)).get('type') == 'pong'
except Exception:
    raise SystemExit(1)
PY
    then
      ready=1
      break
    fi
    kill -0 "$server_pid" 2>/dev/null || { tail -n 30 "$result/server.log"; exit 1; }
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo "Policy server did not become ready." >&2; exit 1; }
  echo "[$(date -Is)] Recording Object 03 trial 0 then trial 1 for $label."
  PYTHONPATH="$LIBERO:$REPO/benchmarks/libero:${PYTHONPATH:-}" \
  LIBERO_PATH="$LIBERO" LIBERO_CONFIG_ROOT="$OUT/libero-config" \
  LIBERO_CONFIG_PATH="$OUT/libero-config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 PYTHONUNBUFFERED=1 \
  "$LIBERO_PY" benchmarks/libero/single_eval.py \
    --config "$OUT/policy_config_video.yml" --suite libero_object --task-id 3 \
    --host 127.0.0.1 --port 8980 --trial-start 0 --num-trials 2 --seed 42 \
    --result-dir "$result" > "$result/client.log" 2>&1
  stop_server
  "$PY" - "$result" "$label" <<'PY'
import json, sys
from pathlib import Path
root, label = Path(sys.argv[1]), sys.argv[2]
x = json.loads((root / 'results.json').read_text())
rows = x['trials']
assert [r['trial'] for r in rows] == [0, 1], rows
assert x['save_videos'] is True, x
assert len(list(root.glob('*.mp4'))) >= 2, root
if label == 'A':
    assert [r['success'] for r in rows] == [True, False], (
        'A failure did not reproduce; stop before interpreting a C comparison', rows
    )
print(label, [r['success'] for r in rows], root / 'results.json')
PY
done
echo "[$(date -Is)] Matched Object 03 video reruns finished."
