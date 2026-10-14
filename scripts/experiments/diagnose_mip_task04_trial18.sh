#!/usr/bin/env bash
# Diagnose LIBERO On() predicates during MIP rollout for task04 trial18.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${ROOT}/outputs/experiments/libero_alpha_mip_2026-09-18_21-58-14/2026-09-18_22-00-15"
OUT="${ROOT}/outputs/libero/mip_final_21370_videos/libero_10_task04_trial018_diag"
SERVER_PYTHON="${SERVER_PYTHON:-/root/tengfei/envs/openwam/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/fasteval-libero/envs/libero/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-/root/fasteval-libero/LIBERO}"
POLICY_GPU="${POLICY_GPU:-0}"
RENDER_GPU="${RENDER_GPU:-4}"
PORT="${PORT:-8941}"
BENCH="${LIBERO_PATH}/libero/libero"

mkdir -p "${OUT}/libero_configs"
# Must exist BEFORE importing libero, otherwise it interactively prompts.
cat > "${OUT}/libero_configs/config.yaml" <<EOF
assets: ${BENCH}/assets
bddl_files: ${BENCH}/bddl_files
benchmark_root: ${BENCH}
datasets: ${LIBERO_PATH}/datasets
init_states: ${BENCH}/init_files
EOF

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] starting MIP server"
CUDA_VISIBLE_DEVICES="${POLICY_GPU}" "${SERVER_PYTHON}" "${ROOT}/scripts/deploy.py" \
  --ckpt-dir "${CKPT_DIR}" \
  --ckpt-name checkpoint_step_21370.safetensors \
  --device cuda:0 --host 127.0.0.1 --port "${PORT}" \
  --denoise-mode sync --inference-mode sync --inference-horizon 10 \
  --compile-enabled false >"${OUT}/server.log" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 120); do
  grep -q "Inference engine ready" "${OUT}/server.log" 2>/dev/null && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "server died"; tail -40 "${OUT}/server.log"; exit 1; }
  sleep 2
done
grep -q "Inference engine ready" "${OUT}/server.log"
echo "[$(date -Is)] server ready; running predicate trace"

CUDA_VISIBLE_DEVICES="${RENDER_GPU}" \
MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="${ROOT}:${LIBERO_PATH}:${ROOT}/benchmarks/libero${PYTHONPATH:+:${PYTHONPATH}}" \
LIBERO_PATH="${LIBERO_PATH}" \
LIBERO_CONFIG_PATH="${OUT}/libero_configs" \
LIBERO_CONFIG_ROOT="${OUT}/libero_configs" \
"${LIBERO_PYTHON}" "${ROOT}/scripts/experiments/diagnose_mip_task04_trial18.py" \
  --root "${ROOT}" \
  --out "${OUT}" \
  --port "${PORT}"
