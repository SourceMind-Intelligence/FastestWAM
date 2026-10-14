#!/usr/bin/env bash
# Parallel MIP libero_10/task06 video dump across 4 policy + 4 render GPUs.
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
BASE_PORT="${BASE_PORT:-8942}"

# Prefer nearly-empty GPUs for policy; use remaining cards for EGL (1:1).
POLICY_GPUS=(${POLICY_GPUS:-0 1 3 6})
RENDER_GPUS=(${RENDER_GPUS:-2 4 5 7})
# 10 trials sharded across 4 workers
ALL_TRIALS=(${ALL_TRIALS:-0 2 3 4 5 7 8 9 10 12})

mkdir -p "${OUT}"
export LIBERO_PATH

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT

n_workers="${#POLICY_GPUS[@]}"
if [[ "${#RENDER_GPUS[@]}" -ne "${n_workers}" ]]; then
  echo "POLICY_GPUS and RENDER_GPUS must have the same length" >&2
  exit 2
fi

echo "[$(date -Is)] starting ${n_workers} MIP servers"
for i in $(seq 0 $((n_workers - 1))); do
  pg="${POLICY_GPUS[$i]}"
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES="${pg}" "${SERVER_PYTHON}" "${ROOT}/scripts/deploy.py" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-name "${CKPT_NAME}" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port "${port}" \
    --denoise-mode sync \
    --inference-mode sync \
    --inference-horizon 10 \
    --compile-enabled false \
    >"${OUT}/server_gpu${pg}_port${port}.log" 2>&1 &
  SERVER_PIDS+=($!)
  echo "  server gpu=${pg} port=${port} pid=${SERVER_PIDS[-1]}"
done

echo "[$(date -Is)] waiting for servers"
for i in $(seq 0 $((n_workers - 1))); do
  pg="${POLICY_GPUS[$i]}"
  port=$((BASE_PORT + i))
  log="${OUT}/server_gpu${pg}_port${port}.log"
  for _ in $(seq 1 150); do
    if grep -q "Inference engine ready" "${log}" 2>/dev/null; then
      echo "  ready gpu=${pg} port=${port}"
      break
    fi
    if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
      echo "server died gpu=${pg}; log=${log}" >&2
      tail -40 "${log}" >&2 || true
      exit 1
    fi
    sleep 2
  done
  grep -q "Inference engine ready" "${log}" || {
    echo "server not ready gpu=${pg}; log=${log}" >&2
    exit 1
  }
done

run_worker() {
  local worker_id="$1"
  local pg="$2"
  local rg="$3"
  local port="$4"
  shift 4
  local trials=("$@")
  local wlog="${OUT}/worker${worker_id}_gpu${pg}.log"
  {
    echo "[$(date -Is)] worker=${worker_id} policy=${pg} render=${rg} port=${port} trials=${trials[*]}"
    for trial in "${trials[@]}"; do
      local tag="libero_10_task06_trial$(printf '%03d' "${trial}")"
      local result_dir="${OUT}/${tag}"
      # Skip if already finished by a previous run.
      if [[ -f "${result_dir}/results.json" ]]; then
        echo "[$(date -Is)] skip existing ${tag}"
        continue
      fi
      mkdir -p "${result_dir}"
      echo "[$(date -Is)] recording ${tag}"
      CUDA_VISIBLE_DEVICES="${rg}" \
      MUJOCO_EGL_DEVICE_ID="${rg}" \
      MUJOCO_GL=egl \
      PYOPENGL_PLATFORM=egl \
      PYTHONPATH="${ROOT}:${LIBERO_PATH}:${ROOT}/benchmarks/libero${PYTHONPATH:+:${PYTHONPATH}}" \
      LIBERO_CONFIG_ROOT="${OUT}/libero_configs/${tag}" \
      LIBERO_CONFIG_PATH="${OUT}/libero_configs/${tag}" \
      "${LIBERO_PYTHON}" "${ROOT}/benchmarks/libero/single_eval.py" \
        --config "${POLICY_CFG}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --suite libero_10 \
        --task-id 6 \
        --trial-start "${trial}" \
        --num-trials 1 \
        --result-dir "${result_dir}" \
        | tee "${result_dir}/client.log"
    done
    echo "[$(date -Is)] worker=${worker_id} done"
  } >"${wlog}" 2>&1
}

# Shard trials round-robin across workers.
declare -a shards
for i in $(seq 0 $((n_workers - 1))); do
  shards[$i]=""
done
idx=0
for trial in "${ALL_TRIALS[@]}"; do
  w=$((idx % n_workers))
  shards[$w]+="${trial} "
  idx=$((idx + 1))
done

echo "[$(date -Is)] launching workers"
for i in $(seq 0 $((n_workers - 1))); do
  # shellcheck disable=SC2206
  trials=(${shards[$i]})
  [[ "${#trials[@]}" -eq 0 ]] && continue
  run_worker "${i}" "${POLICY_GPUS[$i]}" "${RENDER_GPUS[$i]}" "$((BASE_PORT + i))" "${trials[@]}" &
  WORKER_PIDS+=($!)
  echo "  worker ${i}: policy=${POLICY_GPUS[$i]} render=${RENDER_GPUS[$i]} trials=${trials[*]} pid=${WORKER_PIDS[-1]}"
done

ec=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    ec=1
  fi
done

echo "[$(date -Is)] all workers finished ec=${ec}"
/root/tengfei/envs/openwam/bin/python - <<PY
import json
from pathlib import Path
root = Path("${OUT}")
rows=[]
for p in sorted(root.glob("libero_10_task06_trial*/results.json")):
    d=json.loads(p.read_text())
    for t in d["trials"]:
        rows.append((t["trial"], "OK" if t["success"] else "FAIL", t["policy_steps"], Path(t.get("video","")).name if t.get("video") else ""))
print(f"{'trial':>5} {'result':>6} {'steps':>5}  video")
for trial, res, steps, vid in rows:
    print(f"{trial:5d} {res:>6} {steps:5d}  {vid}")
print("successes", sum(1 for _,r,_,_ in rows if r=="OK"), "/", len(rows))
for p in sorted(root.rglob("*.mp4")):
    print(p)
PY
exit "${ec}"
