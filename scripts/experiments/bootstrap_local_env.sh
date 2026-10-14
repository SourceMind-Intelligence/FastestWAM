#!/usr/bin/env bash
# Create ~/tengfei/envs/openwam by cloning the working fastwam conda env,
# then pip-install this checkout. Safe to re-run.
set -euo pipefail
PREFIX="${HOME}/tengfei/envs/openwam"
SRC="${CONDA_CLONE_SRC:-/opt/conda/envs/fastwam}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$(dirname "$PREFIX")"

echo "[$(date -Is)] prefix=${PREFIX} src=${SRC}"
if [[ ! -x "${PREFIX}/bin/python" ]]; then
    conda create --prefix "${PREFIX}" --clone "${SRC}" -y
else
    echo "[$(date -Is)] ${PREFIX} already exists — skipping clone"
fi

"${PREFIX}/bin/python" -m pip install -e "${ROOT}"
"${PREFIX}/bin/python" -m pip install matplotlib pandas pytest
echo "[$(date -Is)] verifying imports"
"${PREFIX}/bin/python" - <<'PY'
import torch, openwam, wandb, hydra, accelerate, deepspeed
print("python_ok", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("openwam", openwam.__file__)
print("wandb", wandb.__version__)
PY
echo "[$(date -Is)] env ready: conda activate ${PREFIX}"
