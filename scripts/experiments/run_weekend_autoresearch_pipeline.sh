#!/usr/bin/env bash
# Weekend AutoResearch pipeline (keep 8×H100 busy through Sunday).
#
# Chain (skips arms whose summary.json already exists):
#   1) wait for MIP-gt-from-FM eval (if running)
#   2) DoT-8 FM 4ep from foundation   — capacity ablation
#   3) eval DoT-8 FM
#   4) DoT-4 MIP-gt 4ep from foundation — fair MIP vs FM (no Mixed, no FM warmstart)
#   5) eval DoT-4 MIP-gt-from-foundation
#
# Usage:
#   tmux new -s weekend 'bash scripts/experiments/run_weekend_autoresearch_pipeline.sh'
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/scripts/experiments/_load_env.sh"

export PATH="/root/tengfei/envs/openwam/bin:$PATH"
export PYTHONUNBUFFERED=1
export NCCL_NVLS_ENABLE=0 NCCL_RAS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export SERVER_PYTHON="${SERVER_PYTHON:-/root/tengfei/envs/openwam/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/root/fasteval-libero/envs/libero/bin/python}"
export LIBERO_PATH="${LIBERO_PATH:-/root/fasteval-libero/LIBERO}"

FOUNDATION="${FOUNDATION_CKPT:-${ROOT}/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model}"
STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
SUITE="${SUITE_ROOT:-${ROOT}/outputs/experiments/libero_weekend_autoresearch_${STAMP}}"
mkdir -p "${SUITE}" "${ROOT}/outputs/logs"
echo "suite=${SUITE}" | tee "${SUITE}/RUN_INFO.txt"

wait_gpus_free() {
  echo "[$(date -Is)] waiting for GPUs free (no deploy.py, GPU0 mem<2GiB)..."
  while true; do
    local mem0 n
    mem0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    n=$(pgrep -c -f 'scripts/deploy.py' || true)
    echo "  mem0=${mem0} deploy=${n}"
    if [[ "${mem0}" -lt 2000 && "${n}" -eq 0 ]]; then
      sleep 5
      mem0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
      n=$(pgrep -c -f 'scripts/deploy.py' || true)
      [[ "${mem0}" -lt 2000 && "${n}" -eq 0 ]] && break
    fi
    sleep 60
  done
}

run_eval() {
  local ckpt_dir="$1" ckpt_name="$2" out="$3" port="$4"
  mkdir -p "${out}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "[skip eval] ${out}/summary.json exists"
    return 0
  fi
  wait_gpus_free
  unset CUDA_VISIBLE_DEVICES
  export GPUS=0,1,2,3 REPLICAS_PER_GPU=1 BASE_PORT="${port}" OUTPUT_DIR="${out}"
  bash benchmarks/libero/run_eval.sh "${ckpt_dir}" "${ckpt_name}" \
    --compile-enabled false --render-gpus 4,5,6,7 \
    --denoise-steps 2 --denoise-mode sync \
    2>&1 | tee "${ROOT}/outputs/logs/$(basename "${out}").log"
}

train_and_wait() {
  local name="$1" master_port="$2"; shift 2
  local out="${SUITE}/${name}"
  local log="${ROOT}/outputs/logs/weekend_${name}_${STAMP}.log"
  mkdir -p "${out}"
  if compgen -G "${out}/2026-*/checkpoint_step_*.safetensors" > /dev/null; then
    echo "[skip train] checkpoint already under ${out}"
    return 0
  fi
  wait_gpus_free
  echo "[$(date -Is)] train ${name}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 MASTER_PORT="${master_port}" \
    bash scripts/train.sh "$@" \
    "training.output_path=${out}" \
    "hydra.run.dir=${out}/hydra" \
    training.debug=false project.wandb.project=null \
    2>&1 | tee "${log}"
}

DISK_COMMON=(
  training.mixed_precision=bf16
  training.zero_stage=2
  training.use_gradient_checkpointing=true
  training.offload_optimizer_device=none
  training.batch_size=8
  training.gradient_accumulation_steps=2
  training.max_steps=null
  training.save_steps=4000
  training.keep_last_k_ckpts=1
  training.max_full_ckpts=0
  training.save_full_states_for_resume=false
  'training.milestone_save_steps=[4000]'
  training.max_milestone_ckpts=1
  training.disk_guard_warn_gb=600
  training.disk_guard_critical_gb=400
  training.measure_flops=false
  project.seed=42
  hydra.job.chdir=false
  dataloader=libero
  model=dual_system_dot
  model.architecture.variant=joint_cross_attn
)

# --- P0: wait current MIP-gt eval if any ---
if [[ ! -f outputs/libero/dot_MIP_gt_from_FM_2step/summary.json ]]; then
  echo "[$(date -Is)] waiting for MIP-gt-from-FM eval summary..."
  while [[ ! -f outputs/libero/dot_MIP_gt_from_FM_2step/summary.json ]]; do
    sleep 120
    echo "  still waiting MIP-gt eval..."
  done
fi
python3 - <<'PY'
import json
p="outputs/libero/dot_MIP_gt_from_FM_2step/summary.json"
s=json.load(open(p)); tot=[0,0]
for r in s["task_results"]: tot[0]+=r["successes"]; tot[1]+=r["trials"]
print(f"[P0] MIP-gt-from-FM overall {tot[0]}/{tot[1]}={100*tot[0]/tot[1]:.2f}%")
PY

# --- P1: DoT-8 FM ---
train_and_wait DOT8_FM_flow 29801 \
  "${DISK_COMMON[@]}" \
  "training.finetune_ckpt_path=${FOUNDATION}" \
  training.finetune_strict_load=false \
  training.num_epochs=4 \
  model.architecture.dot_num_layers=8 \
  model.architecture.action_objective=flow

DOT8_RUN=$(ls -dt "${SUITE}/DOT8_FM_flow"/2026-* 2>/dev/null | head -1 || true)
if [[ -n "${DOT8_RUN}" ]]; then
  CKPT=$(ls "${DOT8_RUN}"/checkpoint_step_*.safetensors | sort -V | tail -1)
  run_eval "$(dirname "${CKPT}")" "$(basename "${CKPT}")" \
    "${ROOT}/outputs/libero/dot8_FM_flow_2step" 8770
fi

# --- P2: DoT-4 MIP-gt from foundation ---
train_and_wait DOT4_MIP_gt_from_foundation 29802 \
  "${DISK_COMMON[@]}" \
  "training.finetune_ckpt_path=${FOUNDATION}" \
  training.finetune_strict_load=false \
  training.num_epochs=4 \
  model.architecture.dot_num_layers=4 \
  model.architecture.action_objective=mip \
  model.architecture.mip_t_star=0.9 \
  model.architecture.mip_num_steps=2 \
  model.architecture.mip_refine_mode=gt \
  model.architecture.detach_bridge=false

MIP_RUN=$(ls -dt "${SUITE}/DOT4_MIP_gt_from_foundation"/2026-* 2>/dev/null | head -1 || true)
if [[ -n "${MIP_RUN}" ]]; then
  CKPT=$(ls "${MIP_RUN}"/checkpoint_step_*.safetensors | sort -V | tail -1)
  run_eval "$(dirname "${CKPT}")" "$(basename "${CKPT}")" \
    "${ROOT}/outputs/libero/dot4_MIP_gt_from_foundation_2step" 8780
fi

echo "[$(date -Is)] weekend pipeline finished → ${SUITE}"
python3 - <<'PY'
import json, os
from collections import defaultdict
paths={
 "DoT-FM 4L":"outputs/libero/dot_FM_flow_final_2step/summary.json",
 "FM→MIP-gt":"outputs/libero/dot_MIP_gt_from_FM_2step/summary.json",
 "DoT-8 FM":"outputs/libero/dot8_FM_flow_2step/summary.json",
 "DoT-4 MIP-gt foundation":"outputs/libero/dot4_MIP_gt_from_foundation_2step/summary.json",
}
print(f"{'Run':<28} {'Overall':>8}")
for n,p in paths.items():
  if not os.path.exists(p):
    print(f"{n:<28} {'pending':>8}"); continue
  s=json.load(open(p)); a=b=0
  for r in s["task_results"]: a+=r["successes"]; b+=r["trials"]
  print(f"{n:<28} {100*a/b:7.2f}%")
PY
