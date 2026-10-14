#!/usr/bin/env bash
# 3-hour MIP hard-task video dump: task06 / task04 / goal02 / spatial07 / task00.
# 4 policy GPUs + 4 render GPUs; skips trials that already have results.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${ROOT}/outputs/experiments/libero_alpha_mip_2026-09-18_21-58-14/2026-09-18_22-00-15"
CKPT_NAME="checkpoint_step_21370.safetensors"
OUT="${ROOT}/outputs/libero/mip_final_21370_videos/hard_rerun"
POLICY_CFG="${ROOT}/benchmarks/libero/policy_config_video.yml"
SERVER_PYTHON="${SERVER_PYTHON:-/root/tengfei/envs/openwam/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/fasteval-libero/envs/libero/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-/root/fasteval-libero/LIBERO}"
BASE_PORT="${BASE_PORT:-8950}"
POLICY_GPUS=(${POLICY_GPUS:-0 1 3 6})
RENDER_GPUS=(${RENDER_GPUS:-2 4 5 7})

mkdir -p "${OUT}"
export LIBERO_PATH

# Also symlink / copy pointers from earlier task06_rerun so we don't redo them.
LEGACY="${ROOT}/outputs/libero/mip_final_21370_videos"
for d in "${LEGACY}"/task06_rerun/libero_10_task06_trial* \
         "${LEGACY}"/libero_10_task06_trial* \
         "${LEGACY}"/libero_10_task04_trial* \
         "${LEGACY}"/libero_goal_task02_trial* \
         "${LEGACY}"/libero_spatial_task07_trial*; do
  [[ -d "${d}" ]] || continue
  base="$(basename "${d}")"
  if [[ -f "${d}/results.json" && ! -e "${OUT}/${base}" ]]; then
    ln -s "${d}" "${OUT}/${base}" 2>/dev/null || true
  fi
done

# Job list: suite task_id trial
# Prefer original-eval failures; add a few OK controls per suite.
JOBS=(
  # libero_10/task06 — original 0/50 (record a dense set)
  libero_10 6 0
  libero_10 6 1
  libero_10 6 5
  libero_10 6 6
  libero_10 6 11
  libero_10 6 13
  libero_10 6 14
  libero_10 6 15
  libero_10 6 16
  libero_10 6 17
  libero_10 6 18
  libero_10 6 19
  libero_10 6 20
  libero_10 6 21
  libero_10 6 22
  libero_10 6 23
  libero_10 6 24
  libero_10 6 25
  libero_10 6 26
  libero_10 6 27
  libero_10 6 28
  libero_10 6 29
  libero_10 6 30
  libero_10 6 31
  libero_10 6 32
  libero_10 6 33
  libero_10 6 34
  libero_10 6 35
  libero_10 6 36
  libero_10 6 37
  libero_10 6 38
  libero_10 6 39
  libero_10 6 40
  libero_10 6 41
  libero_10 6 42
  libero_10 6 43
  libero_10 6 44
  libero_10 6 45
  libero_10 6 46
  libero_10 6 47
  libero_10 6 48
  libero_10 6 49
  # keep a few of the earlier shard indices too if missing
  libero_10 6 2
  libero_10 6 3
  libero_10 6 4
  libero_10 6 7
  libero_10 6 8
  libero_10 6 9
  libero_10 6 10
  libero_10 6 12
  # libero_10/task04 — original fails + controls
  libero_10 4 0
  libero_10 4 1
  libero_10 4 2
  libero_10 4 3
  libero_10 4 4
  libero_10 4 5
  libero_10 4 6
  libero_10 4 7
  libero_10 4 8
  libero_10 4 9
  libero_10 4 10
  libero_10 4 18
  libero_10 4 20
  libero_10 4 39
  libero_10 4 42
  libero_10 4 11
  libero_10 4 12
  libero_10 4 15
  # libero_goal/task02
  libero_goal 2 14
  libero_goal 2 19
  libero_goal 2 33
  libero_goal 2 45
  libero_goal 2 0
  libero_goal 2 1
  # libero_spatial/task07
  libero_spatial 7 9
  libero_spatial 7 17
  libero_spatial 7 46
  libero_spatial 7 0
  libero_spatial 7 1
  # libero_10/task00
  libero_10 0 38
  libero_10 0 45
  libero_10 0 0
)

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT

n_workers="${#POLICY_GPUS[@]}"
echo "[$(date -Is)] hard-rerun: ${#JOBS[@]} job-slots across ${n_workers} workers -> ${OUT}"

for i in $(seq 0 $((n_workers - 1))); do
  pg="${POLICY_GPUS[$i]}"
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES="${pg}" "${SERVER_PYTHON}" "${ROOT}/scripts/deploy.py" \
    --ckpt-dir "${CKPT_DIR}" --ckpt-name "${CKPT_NAME}" \
    --device cuda:0 --host 127.0.0.1 --port "${port}" \
    --denoise-mode sync --inference-mode sync --inference-horizon 10 \
    --compile-enabled false \
    >"${OUT}/server_gpu${pg}_port${port}.log" 2>&1 &
  SERVER_PIDS+=($!)
  echo "  server gpu=${pg} port=${port} pid=${SERVER_PIDS[-1]}"
done

for i in $(seq 0 $((n_workers - 1))); do
  pg="${POLICY_GPUS[$i]}"; port=$((BASE_PORT + i))
  log="${OUT}/server_gpu${pg}_port${port}.log"
  for _ in $(seq 1 150); do
    grep -q "Inference engine ready" "${log}" 2>/dev/null && break
    kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null || { echo "server died ${log}"; tail -30 "${log}"; exit 1; }
    sleep 2
  done
  grep -q "Inference engine ready" "${log}" || { echo "not ready ${log}"; exit 1; }
  echo "  ready gpu=${pg} port=${port}"
done

# Flatten jobs into triples, skip finished, round-robin shard.
declare -a flat=()
n_jobs=$((${#JOBS[@]} / 3))
for j in $(seq 0 $((n_jobs - 1))); do
  suite="${JOBS[$((j*3))]}"
  tid="${JOBS[$((j*3+1))]}"
  trial="${JOBS[$((j*3+2))]}"
  tag="${suite}_task$(printf '%02d' "${tid}")_trial$(printf '%03d' "${trial}")"
  if [[ -f "${OUT}/${tag}/results.json" ]]; then
    echo "[skip] ${tag}"
    continue
  fi
  flat+=("${suite}:${tid}:${trial}")
done
echo "[$(date -Is)] pending jobs: ${#flat[@]}"

declare -a shards
for i in $(seq 0 $((n_workers - 1))); do shards[$i]=""; done
idx=0
for item in "${flat[@]}"; do
  w=$((idx % n_workers))
  shards[$w]+="${item} "
  idx=$((idx + 1))
done

run_worker() {
  local wid="$1" pg="$2" rg="$3" port="$4"
  shift 4
  local items=("$@")
  local wlog="${OUT}/worker${wid}_gpu${pg}.log"
  {
    echo "[$(date -Is)] worker=${wid} policy=${pg} render=${rg} port=${port} n=${#items[@]}"
    for item in "${items[@]}"; do
      IFS=':' read -r suite tid trial <<<"${item}"
      local tag="${suite}_task$(printf '%02d' "${tid}")_trial$(printf '%03d' "${trial}")"
      local result_dir="${OUT}/${tag}"
      if [[ -f "${result_dir}/results.json" ]]; then
        echo "[$(date -Is)] skip ${tag}"
        continue
      fi
      mkdir -p "${result_dir}"
      echo "[$(date -Is)] recording ${tag}"
      CUDA_VISIBLE_DEVICES="${rg}" MUJOCO_EGL_DEVICE_ID="${rg}" \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      PYTHONPATH="${ROOT}:${LIBERO_PATH}:${ROOT}/benchmarks/libero${PYTHONPATH:+:${PYTHONPATH}}" \
      LIBERO_CONFIG_ROOT="${OUT}/libero_configs/${tag}" \
      LIBERO_CONFIG_PATH="${OUT}/libero_configs/${tag}" \
      "${LIBERO_PYTHON}" "${ROOT}/benchmarks/libero/single_eval.py" \
        --config "${POLICY_CFG}" \
        --host 127.0.0.1 --port "${port}" \
        --suite "${suite}" --task-id "${tid}" \
        --trial-start "${trial}" --num-trials 1 \
        --result-dir "${result_dir}" \
        | tee "${result_dir}/client.log"
    done
    echo "[$(date -Is)] worker=${wid} done"
  } >"${wlog}" 2>&1
}

for i in $(seq 0 $((n_workers - 1))); do
  # shellcheck disable=SC2206
  items=(${shards[$i]})
  [[ "${#items[@]}" -eq 0 ]] && continue
  run_worker "${i}" "${POLICY_GPUS[$i]}" "${RENDER_GPUS[$i]}" "$((BASE_PORT + i))" "${items[@]}" &
  WORKER_PIDS+=($!)
  echo "  worker ${i}: n=${#items[@]} pid=${WORKER_PIDS[-1]}"
done

ec=0
for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}" || ec=1
done

echo "[$(date -Is)] finished ec=${ec}"
/root/tengfei/envs/openwam/bin/python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
root = Path("/root/tengfei/FastestWAM/outputs/libero/mip_final_21370_videos/hard_rerun")
by = defaultdict(lambda: [0, 0])
rows = []
for p in sorted(root.glob("*_trial*/results.json")):
    # resolve symlink
    p = p.resolve()
    d = json.loads(p.read_text())
    for t in d["trials"]:
        key = f"{d['suite']}/task{d['task_id']:02d}"
        by[key][1] += 1
        by[key][0] += int(t["success"])
        rows.append((key, t["trial"], "OK" if t["success"] else "FAIL", t["policy_steps"], t.get("video")))
print(f"{'task':24} {'result':>6} {'trial':>5} {'steps':>5}")
for key, trial, res, steps, vid in rows:
    print(f"{key:24} {res:>6} {trial:5d} {steps:5d}")
print("\n=== per-task recorded ===")
for k in sorted(by):
    ok, n = by[k]
    print(f"{k:24} {ok}/{n}")
print("mp4 count", len(list(root.rglob("*.mp4"))))
PY
exit "${ec}"
