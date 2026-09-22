"""Stateless trainer helpers.

Pure compute / IO with no training state: config access, parameter reporting,
LR scheduling, wandb, cross-rank reduction, debug-CSV writing.
"""

import json
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


def cfg_get(cfg, key: str, default=None):
    """Read ``key`` from a dict or attr-style config, with default."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def log_parameter_counts(architecture, *, is_main: bool) -> None:
    """Print per-backbone total/trainable param counts (rank-0 only).

    Uses print() not logger so it survives Hydra's default logging filter;
    the is_main gate keeps it correct regardless of the launcher's stdout
    suppression on non-main ranks.
    """
    if not is_main:
        return

    def _count(module):
        if module is None:
            return 0, 0
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    bb_counts = {name: _count(module) for name, module in architecture.backbones.items()}
    extra_counts = {}
    for name, module in architecture.named_children():
        if name in bb_counts:
            continue
        total, trainable = _count(module)
        if total:
            extra_counts[name] = (total, trainable)
    arch_total = sum(total for total, _ in bb_counts.values()) + sum(total for total, _ in extra_counts.values())
    arch_train = sum(train for _, train in bb_counts.values()) + sum(train for _, train in extra_counts.values())
    print("=" * 60)
    print("Parameter counts")
    for name, (total, trainable) in {**bb_counts, **extra_counts}.items():
        print(f"  {name:<15}: total={total / 1e6:7.1f}M  trainable={trainable / 1e6:7.1f}M")
    print(f"  Architecture  : total={arch_total / 1e6:7.1f}M  trainable={arch_train / 1e6:7.1f}M")
    print("=" * 60, flush=True)


def build_cosine_scheduler(optimizer, *, total_opt_steps: int, cfg, num_processes: int = 1):
    """Linear-warmup + cosine-anneal LR schedule.

    total_opt_steps is in per-process optimizer steps; prepare() wraps the scheduler in
    AcceleratedScheduler which advances it num_processes times per opt step, so scale the
    horizons by num_processes to cancel that out.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    t = cfg.training
    lr = float(t.learning_rate)
    warmup_ratio = float(getattr(t, "warmup_ratio", 0.05))
    lr_min_ratio = float(getattr(t, "lr_min_ratio", 0.01))
    n = max(int(num_processes), 1)
    warmup_steps = int(total_opt_steps * warmup_ratio)
    cosine_steps = max(total_opt_steps - warmup_steps, 1)
    warmup_iters = warmup_steps * n
    warmup_sched = LinearLR(optimizer, start_factor=1.0 / max(warmup_iters, 1), total_iters=warmup_iters)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps * n, eta_min=lr * lr_min_ratio)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_iters])
    logger.info(
        "LR scheduler: cosine | total_opt_steps=%d warmup=%d eta_min=%.2e",
        total_opt_steps,
        warmup_steps,
        lr * lr_min_ratio,
    )
    return scheduler


def init_wandb(cfg):
    """Init a wandb run from cfg.project.wandb. Returns the run or None."""
    wandb_cfg = cfg.project.get("wandb", None)
    if wandb_cfg is None:
        return None
    project = getattr(wandb_cfg, "project", None)
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None

    run_name = getattr(wandb_cfg, "run_name", None)
    entity = getattr(wandb_cfg, "entity", None)
    from omegaconf import OmegaConf

    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info("wandb initialized: %s/%s", project, run.name)
    return run


def reduce_step_metrics(accelerator, losses: dict, grad_norm) -> dict:
    """Reduce loss/grad_norm across ranks (mean); single-process fast path."""
    loss = losses["total"]

    def _f(v):
        return v.item() if isinstance(v, torch.Tensor) else float(v)

    extra_pairs = []
    if "mip_t0" in losses:
        extra_pairs.append(("loss_mip_t0", _f(losses["mip_t0"])))
        extra_pairs.append(("loss_mip_t09", _f(losses["mip_t09"])))

    values = [
        loss.detach().float().item(),
        _f(losses["video"]),
        _f(losses["action"]),
        grad_norm.item(),
        *[v for _, v in extra_pairs],
    ]
    if accelerator is not None and accelerator.num_processes > 1:
        local = torch.tensor(values, device=loss.device, dtype=torch.float32).reshape(1, -1)
        g = accelerator.gather(local).mean(dim=0)
        reduced = {
            "loss_total": g[0].item(),
            "loss_video": g[1].item(),
            "loss_action": g[2].item(),
            "grad_norm": g[3].item(),
        }
        for i, (key, _) in enumerate(extra_pairs):
            reduced[key] = g[4 + i].item()
        return reduced
    reduced = {
        "loss_total": loss.detach().item(),
        "loss_video": _f(losses["video"]),
        "loss_action": _f(losses["action"]),
        "grad_norm": grad_norm.item(),
    }
    for key, value in extra_pairs:
        reduced[key] = value
    return reduced


def write_scaling_metrics_row(
    output_path,
    *,
    metrics,
    global_step,
    opt_step,
    epoch,
    lr,
    steps_per_sec,
    batch_size,
    num_procs,
) -> None:
    """Append one JSONL row for later loss-vs-compute / scaling plots.

    Does not change the training objective; rank-0 IO only, best-effort.
    """
    if not output_path:
        return
    samples_seen = int(global_step) * int(batch_size) * int(num_procs)
    row = {
        "ts": time.time(),
        "step": int(global_step),
        "opt_step": int(opt_step),
        "epoch": int(epoch) if epoch is not None else None,
        "loss": float(metrics["loss_total"]),
        "loss_video": float(metrics["loss_video"]),
        "loss_action": float(metrics.get("loss_action", 0.0)),
        "loss_mip_t0": float(metrics["loss_mip_t0"]) if "loss_mip_t0" in metrics else None,
        "loss_mip_t09": float(metrics["loss_mip_t09"]) if "loss_mip_t09" in metrics else None,
        "grad_norm": float(metrics["grad_norm"]),
        "lr": float(lr),
        "steps_per_sec": float(steps_per_sec),
        "samples_per_sec": float(steps_per_sec) * int(batch_size) * int(num_procs),
        "batch_size_per_gpu": int(batch_size),
        "num_procs": int(num_procs),
        "samples_seen": samples_seen,
        "global_batch": int(batch_size) * int(num_procs),
    }
    path = os.path.join(output_path, "scaling_metrics.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as exc:
        logger.warning("failed to write scaling_metrics.jsonl: %s", exc)


def write_debug_loss_row(
    output_path,
    *,
    labels,
    metrics,
    global_step,
    opt_step,
    epoch,
    lr,
    steps_per_sec,
) -> None:
    """Append one row to debug_loss_history.csv (writes the header on first call).

    Loss columns follow ``labels`` = ``[(display_name, metrics_key)]``.
    """
    loss_log_path = os.path.join(output_path, "debug_loss_history.csv")
    write_header = not os.path.exists(loss_log_path)
    label_cols = ",".join(f"loss_{name}" for name, _ in labels)
    label_vals = ",".join(f"{metrics[key]:.10g}" for _, key in labels)
    with open(loss_log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(f"step,opt_step,epoch,loss,loss_video,{label_cols},grad_norm,lr,steps_per_sec\n")
        f.write(
            f"{global_step},{opt_step},{epoch},{metrics['loss_total']:.10g},{metrics['loss_video']:.10g},{label_vals},"
            f"{metrics['grad_norm']:.10g},{lr:.10g},{steps_per_sec:.10g}\n"
        )
