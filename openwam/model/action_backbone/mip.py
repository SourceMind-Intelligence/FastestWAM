"""MIP (Minimal Iterative Policy) helpers for the action stream.

MIP data-time ``t ∈ [0, 1]`` uses ``t=1`` as clean action and ``t=0`` as the
start of the trajectory. OpenWAM's ActionDiT was pretrained with flow-matching
where the embedder sees ``timestep = σ * num_train_timesteps`` and

    noisy = (1 - σ) * A + σ * z    # σ=1 noise, σ=0 clean

so MIP mix ``t * A + (1-t) * z`` is the same interpolation with ``σ = 1 - t``.
Feeding the raw MIP value ``0.9`` (or ``900``) would tell a pretrained FM head
the sample is noisy when it is 90% clean.

Train and inference share one grid ``linspace(0, mip_t_star, mip_num_steps)``.
With the defaults ``mip_num_steps=2, mip_t_star=0.9`` this recovers the original
two-point MVP ``{0.0, 0.9}``.

Refinement training modes (``mip_refine_mode``, MIP-2 only):

- ``gt``: original — refine on ``t*GT + (1-t)*noise``
- ``mixed``: self-forcing — refine on ``t*(0.5*GT + 0.5*pred0.detach()) + (1-t)*noise``
- ``pred``: full self-forcing — refine on ``t*pred0.detach() + (1-t)*noise``

Inference is always the original MIP-2 chain (zeros → pred0 → ``t*pred0`` → pred1).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

MIP_T_STAR = 0.9
MIP_NUM_STEPS = 2
MIP_REFINE_MODES = ("gt", "mixed", "pred")


def mip_t_to_openwam_timestep(t_mip: float, num_train_timesteps: int = 1000) -> float:
    """Map MIP data-time ``t`` onto the OpenWAM ActionDiT timestep axis.

    ``t_mip=0.0`` → 1000 (maximum uncertainty) and ``t_mip=0.9`` → 100
    (σ=0.1, matching ``I = 0.9 A + 0.1 z``).
    """
    t = float(t_mip)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"MIP t must be in [0, 1], got {t_mip!r}")
    steps = int(num_train_timesteps)
    if steps <= 0:
        raise ValueError(f"num_train_timesteps must be positive, got {num_train_timesteps!r}")
    return (1.0 - t) * float(steps)


def mip_t_grid(num_steps: int, t_star: float = MIP_T_STAR) -> list[float]:
    """Shared train/infer MIP data-time grid.

    ``num_steps=2, t_star=0.9`` → ``[0.0, 0.9]``.
    ``num_steps=4, t_star=0.9`` → ``[0.0, 0.3, 0.6, 0.9]``.
    """
    n = int(num_steps)
    if n < 2:
        raise ValueError(f"mip_num_steps must be >= 2, got {num_steps!r}")
    t_hi = float(t_star)
    if not 0.0 < t_hi <= 1.0:
        raise ValueError(f"mip_t_star must be in (0, 1], got {t_star!r}")
    if n == 2:
        # Exact endpoints avoid float drift away from the original MVP pair.
        return [0.0, t_hi]
    return [i * t_hi / (n - 1) for i in range(n)]


def mip_action_timesteps(
    t_mip: float,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_train_timesteps: int = 1000,
) -> Tensor:
    """Batch of OpenWAM timesteps for a single MIP data-time."""
    value = mip_t_to_openwam_timestep(t_mip, num_train_timesteps=num_train_timesteps)
    return torch.full((int(batch_size),), value, device=device, dtype=dtype)


def mip_branch_input(actions: Tensor, t_mip: float, noise: Tensor | None = None) -> Tensor:
    """Build the MIP action input at data-time ``t_mip``.

    Training uses ``I(0)=0`` and ``I(t)=t A + (1-t) z`` for ``t>0``.
    Inference uses the same mix with ``z = 0`` (no extra Gaussian).
    """
    t = float(t_mip)
    if t == 0.0:
        return torch.zeros_like(actions)
    if noise is None:
        noise = torch.zeros_like(actions)
    return t * actions + (1.0 - t) * noise


def normalize_mip_refine_mode(mode: str | None) -> str:
    """Validate ``mip_refine_mode`` (default ``gt`` = original MIP)."""
    value = "gt" if mode is None else str(mode).strip().lower()
    if value not in MIP_REFINE_MODES:
        raise ValueError(f"mip_refine_mode must be one of {MIP_REFINE_MODES}, got {mode!r}")
    return value


def mip_refine_anchor(gt_action: Tensor, pred0: Tensor, mode: str) -> Tensor:
    """Build the clean-action anchor used by the MIP refine branch.

    ``pred0`` must already be detached by the caller when ``mode`` is
    ``mixed`` / ``pred`` so refine loss does not backprop into the coarse step
    through the refine input.
    """
    mode = normalize_mip_refine_mode(mode)
    if mode == "gt":
        return gt_action
    if mode == "mixed":
        return 0.5 * gt_action + 0.5 * pred0
    return pred0


def validate_mip_grid(grid: Sequence[float]) -> list[float]:
    """Normalize and validate a caller-supplied MIP grid."""
    values = [float(t) for t in grid]
    if len(values) < 2:
        raise ValueError(f"MIP grid must have >= 2 points, got {values!r}")
    if values[0] != 0.0:
        raise ValueError(f"MIP grid must start at 0.0, got {values!r}")
    for t in values:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"MIP grid values must be in [0, 1], got {values!r}")
    for a, b in zip(values, values[1:]):
        if b <= a:
            raise ValueError(f"MIP grid must be strictly increasing, got {values!r}")
    return values
