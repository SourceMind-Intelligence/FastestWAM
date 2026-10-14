"""Architecture-side helpers shared across WAM frameworks.

Bridge-layer resolution, latent-mask downsampling, and the video
tokens-per-frame derivation used by the MoT drivers and the SingleSystem
mask builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from openwam.model.video_backbone.base import BlockLoopState

# Wan VAE encodes every 4 video frames into 1 latent time step.
VAE_TEMPORAL_FACTOR = 4


def compute_video_tokens_per_frame(vstate: "BlockLoopState", driver_name: str) -> int:
    """Derive video tokens-per-frame from the spatial dims populated on ``vstate``.

    Used by every MoT driver to build the v↔v block of the joint attention mask.
    The video backbone's ``prepare()`` must populate ``grid_height`` and
    ``grid_width`` on the ``BlockLoopState``; if not, raise a clear error
    attributable to the calling driver via ``driver_name``.
    """
    h = int(getattr(vstate, "grid_height", 0))
    w = int(getattr(vstate, "grid_width", 0))
    if h <= 0 or w <= 0:
        raise ValueError(
            f"{driver_name}: cannot derive video_tokens_per_frame from vstate "
            f"(grid_height={h}, grid_width={w}). The video backbone's prepare() must populate them."
        )
    return h * w


def downsample_video_mask_to_latent(
    video_is_pad: torch.Tensor,
    *,
    temporal_factor: int = VAE_TEMPORAL_FACTOR,
    skip_first: bool = True,
) -> torch.Tensor:
    """Downsample frame-level padding mask to VAE latent temporal dimension.

    Wan2pt1-family causal VAEs encode frame 0 into latent[0] alone, then
    group the remaining tail frames by ``temporal_factor``. A latent step is
    padded only if ALL frames in the group are padded.

    Modes:
      - ``skip_first=True`` (FastWAM / Wan TI2V): latent[0] is the conditioning
        frame and excluded from loss; the returned mask covers tail latent
        steps only (shape ``T_latent_tail = ceil((T_video - 1) / k)``). This
        matches the Wan loss path which trims pred/target via ``[:, :, 1:]``.
      - ``skip_first=False`` (Cosmos T2V, no first-frame conditioning): all
        latents — including latent[0] — are predicted, so the mask must
        include frame 0 too (shape ``T_latent = 1 + T_latent_tail``).

    Args:
        video_is_pad: (..., T_video) bool, True=padded.
        temporal_factor: VAE temporal compression factor for the tail.
        skip_first: whether latent[0] is excluded from the loss (default True
            for Wan compatibility).

    Returns:
        (..., T_latent_out) bool mask, where ``T_latent_out`` is either
        ``T_latent_tail`` (skip_first=True) or ``1 + T_latent_tail``.
    """
    T = video_is_pad.shape[-1]
    leading_shape = video_is_pad.shape[:-1]
    if T == 0:
        return video_is_pad

    first_latent_mask = video_is_pad[..., 0:1]  # (..., 1)

    if T == 1:
        if skip_first:
            return torch.zeros((*leading_shape, 0), dtype=torch.bool, device=video_is_pad.device)
        return first_latent_mask

    tail_is_pad = video_is_pad[..., 1:]
    T_tail = tail_is_pad.shape[-1]
    pad_len = (temporal_factor - T_tail % temporal_factor) % temporal_factor
    if pad_len > 0:
        pad_block = torch.ones((*leading_shape, pad_len), dtype=torch.bool, device=tail_is_pad.device)
        tail_is_pad = torch.cat([tail_is_pad, pad_block], dim=-1)
    grouped = tail_is_pad.view(*tail_is_pad.shape[:-1], -1, temporal_factor)
    tail_latent_mask = grouped.all(dim=-1)

    if skip_first:
        return tail_latent_mask
    return torch.cat([first_latent_mask, tail_latent_mask], dim=-1)


def _cfg_field(cfg: Any, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def evenly_spaced_bridge_layers(num_layers: int, count: int) -> tuple:
    """Evenly spaced video-layer indices in ``[0, num_layers)``, always including the last.

    Used by DoT (``dot_num_layers``): a shallow ActionDiT cross-attends this many
    video bridges. ``count=4, num_layers=30`` → ``(0, 10, 19, 29)``.
    """
    num_layers = int(num_layers)
    count = int(count)
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if count < 1:
        raise ValueError(f"dot_num_layers / count must be >= 1, got {count}")
    if count > num_layers:
        raise ValueError(f"dot_num_layers={count} exceeds video num_layers={num_layers}")
    if count == 1:
        return (num_layers - 1,)
    last = num_layers - 1
    return tuple(int(round(i * last / (count - 1))) for i in range(count))


def resolve_bridge_layers(cfg: Any, *, num_layers: Optional[int] = None) -> tuple:
    """Parse ``bridge_layers`` indices from an architecture config.

    Input modes (first match wins):
      - Explicit: ``cfg.bridge_layers`` is a list / tuple / comma-separated str.
      - DoT: ``cfg.bridge_layers`` is None and ``cfg.dot_num_layers`` is set;
        evenly spaced indices via :func:`evenly_spaced_bridge_layers`.
      - Interval: ``cfg.bridge_layers`` is None and ``cfg.bridge_interval`` is set;
        the indices are ``range(0, num_layers, bridge_interval)``.

    ``num_layers`` should be passed by the caller (typically the video backbone's
    ``num_layers``). It falls back to ``cfg.num_dit_layers`` for backward
    compatibility.

    The output is always sorted with no duplicates.
    """
    bl_raw = _cfg_field(cfg, "bridge_layers", None)

    if bl_raw is None:
        if num_layers is None:
            num_layers = _cfg_field(cfg, "num_dit_layers", None)
        if num_layers is None:
            raise ValueError(
                "num_layers must be provided (or num_dit_layers set on cfg) when bridge_layers is null"
            )
        num_layers = int(num_layers)

        dot_n = _cfg_field(cfg, "dot_num_layers", None)
        if dot_n is not None:
            bl = evenly_spaced_bridge_layers(num_layers, int(dot_n))
        else:
            interval_raw = _cfg_field(cfg, "bridge_interval", None)
            if interval_raw is None:
                raise ValueError(
                    "bridge_layers is null but neither bridge_interval nor dot_num_layers is set"
                )
            interval = int(interval_raw)
            assert interval >= 1, f"bridge_interval must be >= 1, got {interval}"
            bl = tuple(range(0, num_layers, interval))
    elif isinstance(bl_raw, str):
        bl = tuple(int(x) for x in bl_raw.split(","))
    elif isinstance(bl_raw, tuple):
        bl = bl_raw
    else:
        bl = tuple(bl_raw)

    bl = tuple(sorted(bl))
    assert len(set(bl)) == len(bl), f"bridge_layers must be unique, got {bl}"
    return bl


__all__ = [
    "VAE_TEMPORAL_FACTOR",
    "compute_video_tokens_per_frame",
    "downsample_video_mask_to_latent",
    "evenly_spaced_bridge_layers",
    "resolve_bridge_layers",
]
