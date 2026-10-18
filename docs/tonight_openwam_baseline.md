> Historical setup log, archived 2026-10-18. Pending statuses below describe that earlier session, not current experiment status. See `arxiv_readiness_20261018.md`.

# Tonight OpenWAM LIBERO baseline audit

Reproducible 8-GPU OpenWAM-Alpha → LIBERO DualSystem / Wan2.2-TI2V-5B flow-matching baseline.

## Git identity

| Field | Value |
|---|---|
| Remote | `git@github.com:SourceMind-Intelligence/FastestWAM.git` |
| Branch | `tengfei` |
| SHA | `a8992b613fdb5b1c4649f53b3a914379c49e3892` |
| Status at audit | Clean relative to HEAD except local RoboTwin submodule add (staged) and eval-path convenience edits (unstaged). Training code/configs used tonight are unmodified. |

## Environment

| Field | Value |
|---|---|
| Python (training env `fastwam`) | 3.10.21 (`/opt/conda/envs/fastwam/bin/python`) |
| PyTorch | 2.7.1+cu128 (matches this checkout's README recommendation) |
| CUDA (driver) | 12.8 (nvidia-smi) |
| CUDA (torch) | 12.8 |
| Driver | 570.211.01 |
| GPUs | 8 × NVIDIA H100 80GB HBM3 (81559 MiB each) |
| CPU | 128 logical cores |
| RAM | 2.0 TiB |

GPU 6 reported `Uncorr. ECC = 1` at audit time. GPU 0 had ~25 GiB allocated to an idle `/root/openwam-eval` RoboDojo policy server (`CUDA_VISIBLE_DEVICES=0`, PID 550755, 0% util). Debug/overnight launches still target all 8 GPUs; if rank-0 OOMs that occupancy is the first thing to revisit.

Install check: `python -c "import openwam, torch; print(openwam.__file__, torch.__version__, torch.cuda.device_count())"` in `fastwam`. Cheap tests: `tests/test_finetune_config_override.py` — 11 passed. RoboTwin / LIBERO simulator were **not** installed (offline training only).

Local training prefix env (cloned from `fastwam`):

```bash
conda activate /root/tengfei/envs/openwam
```

Python 3.10, torch 2.7.1+cu128, 8 visible GPUs, `openwam` editable from this checkout. Launchers prefer this prefix when it exists, otherwise `/opt/conda/envs/fastwam`.

## Storage (Phase 1)

Repo, home, `/tmp`, Hugging Face cache, and planned `assets/` / `outputs/` all sit on the same overlay filesystem:

| Mount | Size | Used | Avail | Use% |
|---|---|---|---|---|
| `/` (overlay, also repo/home/tmp) | 7.0T | 1.8T | **4.9T** | 26–27% |

Recorded immediately before downloads (2026-09-17T01:42:25+08:00). No cache relocation: the large filesystem already hosts the checkout. `HF_HOME` remains the default `~/.cache/huggingface` on the same 4.9T volume.

Estimated download:

- Official LIBERO training data (`OpenWAM/LIBERO`): ~1.9 GB — **done**
- OpenWAM-Alpha foundation checkpoint: ~24.8 GB — **in progress** (see Phase 4)
- Wan2.2-TI2V-5B **not downloaded**: this checkout's finetune loader builds the architecture from the foundation checkpoint's self-contained `config.yaml` + weights; `model.video_backbone.model_path` does not need to exist on the training host (`openwam/train/utils/ckpt_model_loader.py`). A local copy already exists at `/root/OpenWAM-work/OpenWAM/assets/video_backbone_ckpt/Wan2.2-TI2V-5B` if needed later.

## Exact commands expected by THIS commit

Native install (README):

```bash
conda create -n openwam python=3.10
conda activate openwam
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Tonight reuses the existing `fastwam` env, which already has the recommended torch/CUDA stack, then `pip install -e .` from this checkout.

Asset downloaders (non-interactive flags exist):

```bash
python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes
python scripts/download_assets/download_openwam_checkpoints.py \
  --family alpha --name OpenWAM-Alpha-Pretrain-Foundation-Model --yes
```

Official α → LIBERO fine-tune path (`assets/openwam_usage_docs/openwam-alpha-finetuning.md` + `configs/train.yaml`):

```bash
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=libero \
  training.finetune_ckpt_path=<FOUNDATION_CKPT> \
  training.debug=true
```

Tonight's wrappers (do not change architecture / losses):

```bash
NPROC_PER_NODE=8 bash scripts/experiments/run_libero_alpha_flow_baseline.sh debug
NPROC_PER_NODE=8 bash scripts/experiments/run_libero_alpha_flow_baseline.sh overnight
```

Notes that differ from generic docs:

- Checkpoints are written under `training.output_path/<timestamp>[_debug]`, **not** `project.output_dir`.
- Debug mode (this trainer) hard-sets `max_steps=20`, `save_steps=10`, constant LR, and disables wandb.
- Overnight run must be **step-driven**: `training.num_epochs=null` plus `training.max_steps=5000` (both unset is a hard error).
- `training.batch_size` in this checkout is **16 per GPU** (docs snippet still says 24).
- Architecture tonight: DualSystem `joint_self_attn`, Wan2.2-TI2V-5B, `action_dim=80`, `state_dim=80`, `attention_mask_mode=mutual`.
- LIBERO reader: `num_frames=33` (action horizon 32), `video_stride=4` (9 video frames), `normalize_mode=min-max`, `unify_action_map=["0-9"]`. Color jitter is off.
- Verified Hydra / trainer keys used overnight: `training.max_steps`, `training.save_steps`, `training.keep_last_k_ckpts`, `training.save_full_states_for_resume`, `training.mixed_precision=bf16`, `training.zero_stage=2`, `training.use_gradient_checkpointing=true`, `project.seed=42`.

## Scientific control

Unchanged vs upstream: ActionDiT, VideoDiT, attention pattern, action dimensions, proprio representation, action normalization, augmentation, video/action flow-matching objectives, `lambda_video` / `lambda_action`.

## Phase 3 — LIBERO data

- Downloader: `python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes`
- Path: `/root/tengfei/FastestWAM/assets/benchmark_data/libero`
- Disk: **1.9G**
- `configs/dataloader/libero.yaml` `dataset_dir` points at that path (official downloader rewrote it)
- `meta/conversion.json`: `native_delta_eef10`, 1693 episodes, 273465 frames, 74 videos, 40 tasks, fps=10, source `lerobot/libero@a1aaacb7`
- Normalization stats: `meta/libero_normalization_stats.npy` present (`stats_exists: true`)
- Action representation **not** changed (native 10-D EEF scattered into 80-D slots 0–9, min-max, identity-pinned rot6d)

### Sample 0 / batch shapes (data-only smoke)

From `outputs/logs/libero_data_smoke.json` (4 batches, 0.85s, RSS 687→699 MB, no growth):

| Field | Shape / type | Notes |
|---|---|---|
| video | list of 9 PIL RGB, 320×384 | `num_frames=33`, `video_stride=4` → 9 frames |
| first_frame_image | list of 1 PIL RGB, 320×384 | |
| action | `(32, 80)` float32 | finite; horizon = num_frames−1 |
| action_mask | `(32, 80)` bool | 0/1 |
| proprio | `(1, 80)` float32 | finite |
| proprio_mask | `(1, 80)` bool | 0/1 |
| video_mask | `(9,)` bool | present |
| prompt | str | e.g. LIBERO language instruction |

Smoke command:

```bash
/opt/conda/envs/fastwam/bin/python scripts/experiments/smoke_libero_dataloader.py
```

## Phase 4 — OpenWAM-Alpha foundation

Target directory (self-contained):

```
/root/tengfei/FastestWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model
```

Already on disk before the weight file finished:

- `config.yaml` — DualSystem `joint_self_attn`, `wan22_ti2v_5b`, `action_dim=80`, `state_dim=80`, freeze VAE/text encoder
- `tokenizer/google/umt5-xxl/`
- `README.md`

Weights file: `checkpoint_step_154000.safetensors` (HF size **24813767464** bytes ≈ 23.11 GiB / 24.8 GB).

Direct `huggingface.co` from this host times out. `hf-mirror.com` 302s the payload to `us.aws.cdn.hf.co` (XET). Active download is `aria2c -x 16` through the local SSH proxy `http://127.0.0.1:17897`, resume-enabled. Observed throughput ≈ **1.1–1.3 MiB/s**, multi-hour ETA. Sparse `stat` size can already show 22G while allocated bytes are still growing — completeness requires **no `.aria2` sidecar** and allocated size ≥ 20 GB.

Wan2.2-TI2V-5B was **not** downloaded separately (self-contained finetune checkpoint).

## Phase 5 — Data-only smoke

**Passed.** See tensor table above. No corrupted samples in 4 shuffled batches; cameras present as 9-frame RGB canvases; actions/proprio finite; masks valid; RSS stable.

## Phase 6–10 — 8-GPU debug + overnight

Not started until the foundation weight file is complete. A tmux watchdog will:

1. Wait until `checkpoint_step_154000.safetensors` is fully allocated and parseable as safetensors
2. Run `NPROC_PER_NODE=8 bash scripts/experiments/run_libero_alpha_flow_baseline.sh debug`
3. Require 20 finite debug steps, 8 ranks, finite video/action/total/grad, and a written checkpoint
4. If that passes, launch overnight:

```bash
NPROC_PER_NODE=8 MASTER_PORT=29512 bash scripts/experiments/run_libero_alpha_flow_baseline.sh overnight
```

which expands to (keys verified against this checkout):

```bash
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=libero \
  model=dual_system \
  training.finetune_ckpt_path=/root/tengfei/FastestWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
  training.mixed_precision=bf16 \
  training.zero_stage=2 \
  training.use_gradient_checkpointing=true \
  project.seed=42 \
  training.debug=false \
  training.num_epochs=null \
  training.max_steps=5000 \
  training.save_steps=500 \
  training.keep_last_k_ckpts=3 \
  training.save_full_states_for_resume=true \
  training.output_path=outputs/experiments/libero_alpha_flow_baseline_<timestamp>/
```

Attach: `tmux attach -t tonight_baseline`

- window `foundation`: weight download
- window `pipeline`: wait → debug → overnight

Logs:

- dataset smoke: `outputs/logs/libero_data_smoke.json`
- LIBERO download: `outputs/logs/download_libero.log`
- foundation download: `outputs/logs/download_foundation.log`
- pipeline: `outputs/logs/pipeline_<timestamp>.log`
- debug / overnight: `outputs/logs/libero_alpha_flow_{debug,baseline}_<timestamp>.log`

`RUN_INFO.txt` is written under the overnight output parent after launch.

WandB / Hugging Face routing (from repo `.env`, secrets never logged):

- HF token is valid (`hf auth whoami` → `tengfeima-ai`). Large-file downloads must use the SSH HTTP proxy; direct `huggingface.co` times out. `HF_ENDPOINT=hf-mirror.com` from `~/.bashrc` is unset by the launcher because the mirror 302s to the US XET CDN.
- WandB key is valid as user `tengfeima1320613206` / entity `tengfeima13206`. Existing projects: `fastestwam`, `wan-wm`. Overnight logs to **`tengfeima13206/fastestwam`**.
- **Do not send WandB through the proxy**: `api.wandb.ai` SSL-breaks on `127.0.0.1:17897`. Launchers put `wandb.ai` / `api.wandb.ai` in `NO_PROXY` so wandb goes direct.
- Debug mode still disables wandb (upstream trainer). Overnight uses online wandb.
- If `WANDB_API_KEY` were missing, the launcher would set `WANDB_MODE=disabled` so login cannot block training.

## Fill-in after debug / overnight start

| Item | Value |
|---|---|
| Debug result | pending foundation download |
| Debug last loss / video / action / grad | pending |
| Overnight command | see Phase 6–10 |
| Overnight current step / steps/s | pending |
| GPU util / memory | pending (GPU0 idle eval server ~25 GiB) |
| Checkpoint dir | `outputs/experiments/libero_alpha_flow_baseline_<timestamp>/<run_ts>/` |
