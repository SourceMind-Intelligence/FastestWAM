"""MIP (Mean Inverse Prediction) helpers for the action stream.

MIP data-time ``t ∈ [0, 1]`` uses ``t=1`` as clean action and ``t=0`` as the
start of the trajectory. OpenWAM's ActionDiT was pretrained with flow-matching
where the embedder sees ``timestep = σ * num_train_timesteps`` and

    noisy = (1 - σ) * A + σ * z    # σ=1 noise, σ=0 clean

so MIP mix ``t * A + (1-t) * z`` is the same interpolation with ``σ = 1 - t``.
Feeding the raw MIP value ``0.9`` (or ``900``) would tell a pretrained FM head
the sample is noisy when it is 90% clean.
"""

from __future__ import annotations

import torch
from torch import Tensor

MIP_T_STAR = 0.9


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

    Training uses ``I0 = 0`` and ``I(t*) = t* A + (1-t*) z``. Inference uses
    the same mix with ``z = 0`` (no extra Gaussian).
    """
    t = float(t_mip)
    if t == 0.0:
        return torch.zeros_like(actions)
    if noise is None:
        noise = torch.zeros_like(actions)
    return t * actions + (1.0 - t) * noise
