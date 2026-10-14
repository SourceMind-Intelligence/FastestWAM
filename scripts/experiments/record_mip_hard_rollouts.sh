#!/usr/bin/env bash
# Re-run a few hard MIP LIBERO trials with agentview mp4 dumps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${ROOT}/outputs/experiments/libero_alpha_mip_2026-09-18_21-58-14/2026-09-18_22-00-15"
CKPT_NAME="checkpoint_step_21370.safetensors"
OUT="${ROOT}/outputs/libero/mip_final_21370_videos"
POLICY_CFG="${ROOT}/benchmarks/libero/policy_config_video.yml"
SERVER_PYTHON="${SERVER_PYTHON:-/root/tengfei/envs/openwam/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/fasteval-libero/envs/libero/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-/root/fasteval-libero/LIBERO}"
POLICY_GPU="${POLICY_GPU:-0}"
RENDER_GPU="${RENDER_GPU:-4}"
PORT="${PORT:-8940}"

mkdir -p "${OUT}"
export LIBERO_PATH
export LIBERO_CONFIG_ROOT="${OUT}/libero_configs"
export CUDA_VISIBLE_DEVICES="${POLICY_GPU}"
export MUJOCO_GL=egl

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] starting MIP server on policy_gpu=${POLICY_GPU} port=${PORT}"
"${SERVER_PYTHON}" "${ROOT}/scripts/deploy.py" \
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

echo "[$(date -Is)] waiting for server readiness"
for i in $(seq 1 120); do
  if grep -q "Inference engine ready" "${OUT}/server.log" 2>/dev/null; then
    break
  fi
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
  local suite="$1" task_id="$2" trial="$3"
  local tag="${suite}_task$(printf '%02d' "${task_id}")_trial$(printf '%03d' "${trial}")"
  local result_dir="${OUT}/${tag}"
  mkdir -p "${result_dir}"
  echo "[$(date -Is)] recording ${tag}"
  CUDA_VISIBLE_DEVICES="${RENDER_GPU}" \
  MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  PYTHONPATH="${LIBERO_PATH}:${ROOT}/benchmarks/libero${PYTHONPATH:+:${PYTHONPATH}}" \
  LIBERO_CONFIG_ROOT="${OUT}/libero_configs/${tag}" \
  LIBERO_CONFIG_PATH="${OUT}/libero_configs/${tag}" \
  "${LIBERO_PYTHON}" "${ROOT}/benchmarks/libero/single_eval.py" \
    --config "${POLICY_CFG}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --suite "${suite}" \
    --task-id "${task_id}" \
    --trial-start "${trial}" \
    --num-trials 1 \
    --result-dir "${result_dir}" \
    | tee "${result_dir}/client.log"
}

# Worst task first: task06 is 0/50. Then mixed task04 fail/success. Then goal/spatial fails.
run_one libero_10 6 0
run_one libero_10 6 1
run_one libero_10 4 0
run_one libero_10 4 11
run_one libero_10 4 18
run_one libero_goal 2 14
run_one libero_goal 2 0
run_one libero_spatial 7 9
run_one libero_spatial 7 0

echo "[$(date -Is)] done. videos under ${OUT}"
find "${OUT}" -name '*.mp4' | sort
