#!/usr/bin/env bash
# Record several MIP rollouts of libero_10/task06 (originally 0/50).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${ROOT}/outputs/experiments/libero_alpha_mip_2026-09-18_21-58-14/2026-09-18_22-00-15"
CKPT_NAME="checkpoint_step_21370.safetensors"
OUT="${ROOT}/outputs/libero/mip_final_21370_videos/task06_rerun"
POLICY_CFG="${ROOT}/benchmarks/libero/policy_config_video.yml"
SERVER_PYTHON="${SERVER_PYTHON:-/root/tengfei/envs/openwam/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/fasteval-libero/envs/libero/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-/root/fasteval-libero/LIBERO}"
POLICY_GPU="${POLICY_GPU:-0}"
RENDER_GPU="${RENDER_GPU:-4}"
PORT="${PORT:-8942}"
# Original eval failed all 50; re-record a spread of init states.
TRIALS="${TRIALS:-0 2 3 4 5 7 8 9 10 12}"

mkdir -p "${OUT}"
export LIBERO_PATH

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] MIP task06 rerun trials=[${TRIALS}] policy_gpu=${POLICY_GPU} render_gpu=${RENDER_GPU}"
CUDA_VISIBLE_DEVICES="${POLICY_GPU}" "${SERVER_PYTHON}" "${ROOT}/scripts/deploy.py" \
  --ckpt-dir "${CKPT_DIR}" \
  --ckpt-name "${CKPT_NAME}" \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --denoise-mode sync \
  --inference-mode sync \
  --inference-horizon 10 \
  --compile-enabled false \
  >"${OUT}/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 120); do
  grep -q "Inference engine ready" "${OUT}/server.log" 2>/dev/null && break
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "server died; see ${OUT}/server.log" >&2
    exit 1
  fi
  sleep 2
done
grep -q "Inference engine ready" "${OUT}/server.log" || {
  echo "server not ready; see ${OUT}/server.log" >&2
  exit 1
}

run_one() {
  local trial="$1"
  local tag="libero_10_task06_trial$(printf '%03d' "${trial}")"
  local result_dir="${OUT}/${tag}"
  mkdir -p "${result_dir}"
  echo "[$(date -Is)] recording ${tag}"
  CUDA_VISIBLE_DEVICES="${RENDER_GPU}" \
  MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  PYTHONPATH="${ROOT}:${LIBERO_PATH}:${ROOT}/benchmarks/libero${PYTHONPATH:+:${PYTHONPATH}}" \
  LIBERO_CONFIG_ROOT="${OUT}/libero_configs/${tag}" \
  LIBERO_CONFIG_PATH="${OUT}/libero_configs/${tag}" \
  "${LIBERO_PYTHON}" "${ROOT}/benchmarks/libero/single_eval.py" \
    --config "${POLICY_CFG}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --suite libero_10 \
    --task-id 6 \
    --trial-start "${trial}" \
    --num-trials 1 \
    --result-dir "${result_dir}" \
    | tee "${result_dir}/client.log"
}

for trial in ${TRIALS}; do
  run_one "${trial}"
done

echo "[$(date -Is)] done"
echo "==== summary ===="
/root/tengfei/envs/openwam/bin/python - <<PY
import json
from pathlib import Path
root = Path("${OUT}")
rows=[]
for p in sorted(root.glob("libero_10_task06_trial*/results.json")):
    d=json.loads(p.read_text())
    for t in d["trials"]:
        rows.append((t["trial"], "OK" if t["success"] else "FAIL", t["policy_steps"], Path(t.get("video","")).name))
print(f"{'trial':>5} {'result':>6} {'steps':>5}  video")
for trial, res, steps, vid in rows:
    print(f"{trial:5d} {res:>6} {steps:5d}  {vid}")
print("successes", sum(1 for _,r,_,_ in rows if r=="OK"), "/", len(rows))
for p in sorted(root.rglob("*.mp4")):
    print(p)
PY
