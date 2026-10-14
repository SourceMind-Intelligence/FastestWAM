#!/usr/bin/env bash
# Stable LIBERO eval: one policy replica per GPU, dedicated EGL render GPUs.
# 8 clients on one render GPU previously SIGABRT'd (exit -6) in mjr_readPixels.
#
# Usage:
#   bash scripts/experiments/run_libero_eval_stable.sh CKPT_DIR CKPT_NAME OUTPUT_DIR
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT_DIR="${1:?ckpt dir}"
CKPT_NAME="${2:?ckpt filename}"
OUTPUT_DIR="${3:?output dir}"
shift 3 || true

unset CUDA_VISIBLE_DEVICES
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"

# Policy on 0,1,2; EGL on 4,6,7 (1:1). GPU 5 may have leftover occupancy.
export SERVER_PYTHON="${SERVER_PYTHON:-/root/tengfei/envs/openwam/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/root/fasteval-libero/envs/libero/bin/python}"
export LIBERO_PATH="${LIBERO_PATH:-/root/fasteval-libero/LIBERO}"
export GPUS="${GPUS:-0,1,2}"
export REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
export OUTPUT_DIR

echo "[$(date -Is)] stable eval ckpt=${CKPT_DIR}/${CKPT_NAME}"
echo "[$(date -Is)] gpus=${GPUS} replicas_per_gpu=${REPLICAS_PER_GPU} render=${RENDER_GPUS:-4,6,7}"
echo "[$(date -Is)] output=${OUTPUT_DIR}"

exec bash "${ROOT}/benchmarks/libero/run_eval.sh" \
    "${CKPT_DIR}" "${CKPT_NAME}" \
    --compile-enabled false \
    --render-gpus "${RENDER_GPUS:-4,6,7}" \
    "$@"
