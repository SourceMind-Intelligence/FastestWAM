#!/usr/bin/env bash
# Source repo .env and apply machine-split routing:
#   Hugging Face  -> SSH HTTP proxy from .env (direct huggingface.co times out)
#   WandB         -> DIRECT (proxy SSL-breaks api.wandb.ai)
# Secrets stay in .env; this file never prints them.
# shellcheck disable=SC1091
if [[ -f "${ROOT}/.env" ]]; then
    set -a
    source "${ROOT}/.env"
    set +a
elif [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# huggingface_hub reads this name; .env only has HF_TOKEN.
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi

# ~/.bashrc sets HF_ENDPOINT=hf-mirror.com, which 302s large files to the US
# XET CDN. With the proxy we want official huggingface.co.
unset HF_ENDPOINT

# Keep loopback + WandB off the proxy even if .env NO_PROXY is incomplete.
_EXTRA_NO_PROXY="localhost,127.0.0.1,::1,wandb.ai,api.wandb.ai,.wandb.ai"
if [[ -z "${NO_PROXY:-}${no_proxy:-}" ]]; then
    export NO_PROXY="${_EXTRA_NO_PROXY}"
else
    export NO_PROXY="${NO_PROXY:-${no_proxy}},${_EXTRA_NO_PROXY}"
fi
export no_proxy="${NO_PROXY}"

# Sensible WandB defaults for this checkout if .env omitted them.
export WANDB_ENTITY="${WANDB_ENTITY:-tengfeima13206}"
export WANDB_PROJECT="${WANDB_PROJECT:-fastestwam}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

# Physical GPU 0 is occupied by another job; train on 1-7 unless overridden.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES}"
    export NPROC_PER_NODE="${#_gpus[@]}"
fi

# Prefer a local prefix env under ~/tengfei/envs when it exists.
_LOCAL_PY="${HOME}/tengfei/envs/openwam/bin"
if [[ -x "${_LOCAL_PY}/python" ]]; then
    export PATH="${_LOCAL_PY}:${PATH}"
    export CONDA_PREFIX="${HOME}/tengfei/envs/openwam"
fi
