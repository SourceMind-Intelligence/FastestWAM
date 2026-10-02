"""One-time per-step FLOPs measurement for the OpenWAM dashboard.

Builds the configured model and one real training batch on a single GPU,
measures forward + backward FLOPs with torch.profiler (with_flops=True,
same backward path as training — gradient-checkpoint recompute included),
and writes dashboard/flops.json. dashboard/serve.py turns that into
cumulative FLOPs as  per_step_per_gpu_flops * global_step * num_gpus.

Run it while GPUs are free (needs one full GPU), with the training env:

    PATH=/root/openwam-eval/env/bin:$PATH HF_HUB_OFFLINE=1 \
        python scripts/profile_step_flops.py dataloader=robotwin

On CUDA OOM the batch size is halved and the result is scaled linearly back
to the configured training batch size (recorded as scaled_linearly: true).
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLOPS_OUT = PROJECT_ROOT / "dashboard" / "flops.json"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _sum_flops(prof) -> float:
    total = 0.0
    for evt in prof.key_averages():
        flops = getattr(evt, "flops", 0) or 0
        if flops > 0:
            total += flops
    return total


def _measure(model, dataloader, batch_size: int) -> float:
    batch = next(iter(dataloader))
    if not isinstance(batch, list):
        batch = [batch]

    def fwd_bwd():
        inputs = model.prepare_inputs(batch)
        result = model.compute_loss(
            **inputs,
            lambda_video=1.0,
            lambda_action=1.0,
        )
        loss = result["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)
        return float(loss.detach())

    torch.cuda.synchronize()
    t0 = time.monotonic()
    warm_loss = fwd_bwd()  # warm up kernels/autotune outside the profile
    torch.cuda.synchronize()
    logger.info("warmup fwd+bwd ok: loss=%.4f (%.1fs, batch=%d)", warm_loss, time.monotonic() - t0, batch_size)

    torch.cuda.synchronize()
    t0 = time.monotonic()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        fwd_bwd()
    torch.cuda.synchronize()
    logger.info("profiled fwd+bwd in %.1fs", time.monotonic() - t0)
    return _sum_flops(prof)


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from openwam.dataloader.registry import build_dataset
    from openwam.model import build_architecture, resolve_architecture_config

    t = cfg.training
    target_batch = int(t.batch_size)
    num_frames = getattr(cfg.dataloader, "num_frames", None)

    logger.info("building dataset (%s) ...", getattr(cfg.dataloader, "type", "?"))
    dataset = build_dataset(cfg.dataloader, split="train")

    logger.info("building architecture ...")
    resolved = resolve_architecture_config(cfg.model)
    model = build_architecture(resolved.registry_name, resolved.params)
    for name in model.freeze_modules(list(getattr(cfg.model, "freeze", []))):
        logger.info("frozen: %s", name)
    model.init_training_schedulers(1000)
    model.set_training_runtime(
        use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
        use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
        max_timestep_boundary=float(t.max_timestep_boundary),
        min_timestep_boundary=float(t.min_timestep_boundary),
    )
    model.set_dtype_device(model.dtype, torch.device("cuda:0"))
    model.train()

    flops = None
    measured_batch = None
    for batch_size in [b for b in (target_batch, 8, 4, 2, 1) if b <= target_batch]:
        measured_batch = batch_size
        try:
            dl = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                collate_fn=list, pin_memory=True,
            )
            flops = _measure(model, dl, batch_size)
            break
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM at batch_size=%d, retrying smaller", batch_size)
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
    if flops is None:
        raise RuntimeError("could not measure FLOPs even at batch_size=1")

    scaled = measured_batch != target_batch
    per_step = flops * (target_batch / measured_batch)
    num_gpus = torch.cuda.device_count()
    payload = {
        "per_step_per_gpu_flops": per_step,
        "measured_flops": flops,
        "measured_batch_size": measured_batch,
        "config_batch_size": target_batch,
        "scaled_linearly": scaled,
        "num_gpus": num_gpus,
        "num_frames": num_frames,
        "dataloader": getattr(cfg.dataloader, "type", None),
        "model": f"{resolved.canonical.framework}/{resolved.canonical.variant}",
        "method": "torch.profiler with_flops=True; bf16; one fwd+bwd micro-batch per GPU; "
                  "cumulative = per_step_per_gpu_flops * global_step * num_gpus",
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    FLOPS_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = FLOPS_OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, FLOPS_OUT)
    print(f"per-step per-GPU FLOPs @ batch {target_batch}: {per_step:.3e}"
          + (f" (scaled from measured batch {measured_batch})" if scaled else ""))
    print(f"all {num_gpus} GPUs per step: {per_step * num_gpus:.3e}")
    print(f"wrote {FLOPS_OUT}")


if __name__ == "__main__":
    main()
