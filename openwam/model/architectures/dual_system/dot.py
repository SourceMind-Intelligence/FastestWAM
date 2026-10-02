"""DualSystem DoT architecture: a shallow action head docked on the video DiT.

DoT ("Dock of Transformer", Faster-WAM, arXiv 2608.02365) treats the video DiT
as a representation hub. The video stream runs its own layer loop and never
attends to the action. At every docked video layer the clean-prefix tokens'
keys (video 3D RoPE undone) and values are captured; :class:`DoTKVFusion`
remaps their channels into the action head's attention space and mixes them
across layers, per head and per action layer. Each action layer then attends to
``[its own K/V, the fused video K/V]`` before its text/proprio cross-attention
and FFN. The paper's head is one layer.

Why the clean prefix: under ``first_frame_causal`` the prefix (the first frame
plus any history frames) attends only to itself and runs at t=0, so its K/V are
the same whether or not noisy future frames follow. Training keeps the video
flow-matching loss on the future frames; inference runs the video DiT once on
the observed frames and then only the action head per denoising step.

MIP on DoT shares that single video pass between its two branches, so its
training step costs about one flow-matching step instead of two joint forwards.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from openwam.model.action_backbone.components import RMSNorm
from openwam.model.action_backbone.mip import (
    mip_action_timesteps,
    mip_branch_input,
    mip_refine_anchor,
)
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import compute_video_tokens_per_frame
from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    build_cross_modal_attention_mask,
    set_video_attention_mask_mode,
)
from openwam.model.video_backbone.wan.models.dit import rope_apply

logger = logging.getLogger(__name__)

_REQUIRED_VIDEO_MASK_MODE = "first_frame_causal"


def _attention(q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor], num_heads: int) -> Tensor:
    """SDPA over ``(B, S, H*D)`` inputs; ``attn_mask`` is bool, True = keep."""
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return rearrange(out, "b n s d -> b s (n d)", n=num_heads)


def _undo_video_rope(k: Tensor, freqs: Tensor, num_heads: int) -> Tensor:
    """Rotate Wan keys back to their pre-RoPE frame (multiply by conj(freqs))."""
    if not torch.is_complex(freqs):
        raise NotImplementedError(
            "DoT undoes the video RoPE with Wan's complex rotary table; this video backbone "
            f"provides rope_freqs of dtype {freqs.dtype}."
        )
    return rope_apply(k, freqs.conj(), num_heads)


class DoTKVFusion(nn.Module):
    """KV-Fusion: channel remap, then per-head layer mixing (Faster-WAM Eq. 2-3).

    Keys and values arrive as ``(L_v, B, S, video_attn_dim)`` with the video RoPE
    undone. ``k_proj`` / ``v_proj`` remap channels into the action head's
    attention space (identity at init when the widths match). ``layer_mix`` of
    shape ``(L_a, heads, L_v)`` then mixes the video layers per head for each
    action layer (uniform at init). Fused keys are RMS-normalised to the scale
    of the action keys. They enter the head at 1D position 0, so the action RoPE
    leaves them unrotated.
    """

    def __init__(
        self,
        *,
        num_video_layers: int,
        num_action_layers: int,
        video_attn_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        attn_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(video_attn_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(video_attn_dim, attn_dim, bias=False)
        if video_attn_dim == attn_dim:
            with torch.no_grad():
                self.k_proj.weight.copy_(torch.eye(attn_dim))
                self.v_proj.weight.copy_(torch.eye(attn_dim))
        self.layer_mix = nn.Parameter(
            torch.full((num_action_layers, num_heads, num_video_layers), 1.0 / num_video_layers)
        )
        self.norm_k = RMSNorm(attn_dim, eps=eps)

    def forward(self, keys: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        """Return fused ``(L_a, B, S, attn_dim)`` keys and values."""
        k = rearrange(self.k_proj(keys), "l b s (n d) -> l b s n d", n=self.num_heads)
        v = rearrange(self.v_proj(values), "l b s (n d) -> l b s n d", n=self.num_heads)
        mix = self.layer_mix.to(dtype=k.dtype)
        fused_k = rearrange(torch.einsum("anl,lbsnd->absnd", mix, k), "a b s n d -> a b s (n d)")
        fused_v = rearrange(torch.einsum("anl,lbsnd->absnd", mix, v), "a b s n d -> a b s (n d)")
        return self.norm_k(fused_k), fused_v


def _dot_video_layers(cfg, num_video_layers: int) -> Tuple[int, ...]:
    layers = cfg.get("dot_video_layers")
    if layers is None:
        return tuple(range(num_video_layers))
    layers = tuple(sorted({int(i) for i in layers}))
    if not layers or any(not 0 <= i < num_video_layers for i in layers):
        raise ValueError(f"dot_video_layers must list video layers in [0, {num_video_layers}), got {layers}")
    return layers


@register_architecture(
    "dual_system_dot",
    status="supported",
    note="DualSystem DoT: shallow action head on K/V fused from the video layers (Faster-WAM).",
    framework="dual_system",
    variant="dot",
)
class DualSystemDoTArchitecture(BaseWAMArchitecture):
    """Shallow action head that reads fused clean-prefix K/V from the video DiT."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.dot_fusion: DoTKVFusion | None = None
        self._dot_video_layers: Tuple[int, ...] = ()
        self._video_attention_mask_mode = _REQUIRED_VIDEO_MASK_MODE
        self._mask_mode_applied_to: int | None = None
        if cfg is None:
            return
        cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
        if self.video_backbone is not None:
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        mode = cfg.get("video_attention_mask_mode") or _REQUIRED_VIDEO_MASK_MODE
        if mode != _REQUIRED_VIDEO_MASK_MODE:
            raise ValueError(
                f"variant=dot needs video_attention_mask_mode={_REQUIRED_VIDEO_MASK_MODE!r} so the clean "
                f"prefix K/V the action head reads do not depend on noisy future frames; got {mode!r}."
            )
        num_video_layers = int(cfg.get("num_dit_layers", 30))
        self._dot_video_layers = _dot_video_layers(cfg, num_video_layers)
        num_action_layers = int(cfg.get("dot_num_action_layers", 1))
        if num_action_layers < 1:
            raise ValueError(f"dot_num_action_layers must be >= 1, got {num_action_layers}")

        video_dim = self._resolve_video_dim(cfg)
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)

        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))
        video_num_heads = int(getattr(self.video_backbone, "num_heads", num_heads))
        video_head_dim = int(getattr(self.video_backbone, "head_dim", attn_head_dim))

        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=int(cfg.get("ffn_dim", 4 * action_dim_hidden)),
            num_heads=num_heads,
            num_layers=num_action_layers,
            video_dim=video_dim,
            # Informational for the MoT-style block API; DoT reads every docked layer.
            bridge_layers=tuple(range(num_video_layers - num_action_layers, num_video_layers)),
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )
        self.dot_fusion = DoTKVFusion(
            num_video_layers=len(self._dot_video_layers),
            num_action_layers=num_action_layers,
            video_attn_dim=video_num_heads * video_head_dim,
            num_heads=num_heads,
            head_dim=attn_head_dim,
        )

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Move the backbones and the KV-Fusion (not a backbone, so base skips it)."""
        super().set_dtype_device(dtype, device)
        if self.dot_fusion is not None:
            self.dot_fusion.to(dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def adapt_checkpoint_state_dict(self, state_dict: dict) -> dict:
        """Warm-start the video DiT from a non-DoT checkpoint; the head starts fresh.

        A DoT checkpoint (it carries ``dot_fusion.*``) loads unchanged. Any other
        checkpoint (the OpenWAM foundation model, a MoT run) supplies everything
        except the action head: its ``action_backbone.*`` keys are dropped and this
        model's freshly initialised head and KV-Fusion fill those slots, as in
        Faster-WAM, so the strict load still checks every video key.
        """
        if any(key.startswith("dot_fusion.") for key in state_dict):
            return state_dict
        own = self.state_dict()
        adapted = {k: v for k, v in state_dict.items() if not k.startswith("action_backbone.")}
        dropped = len(state_dict) - len(adapted)
        for key, value in own.items():
            if key.startswith(("action_backbone.", "dot_fusion.")):
                adapted[key] = value
        logger.info(
            "DoT warm start: video weights from the checkpoint; action head and KV-Fusion start fresh "
            "(%d checkpoint action keys dropped).",
            dropped,
        )
        return adapted

    # ------------------------------------------------------------------
    # Video pass with K/V capture
    # ------------------------------------------------------------------

    def _apply_video_mask_mode(self) -> None:
        vb = self.video_backbone
        if self._mask_mode_applied_to == id(vb):
            return
        set_video_attention_mask_mode(vb, self._video_attention_mask_mode)
        if getattr(vb, "video_attention_mask_mode", None) != _REQUIRED_VIDEO_MASK_MODE:
            raise ValueError(f"variant=dot needs a video backbone running {_REQUIRED_VIDEO_MASK_MODE!r} attention.")
        if self._dot_video_layers[-1] >= vb.num_layers:
            raise ValueError(
                f"dot_video_layers {self._dot_video_layers} do not fit a video backbone with {vb.num_layers} layers."
            )
        self._mask_mode_applied_to = id(vb)

    def _video_layer(
        self,
        layer_id: int,
        vstate,
        attn_mask: Optional[Tensor],
        prefix_tokens: int,
        capture: bool,
    ):
        """One video layer (pre-attn, masked self-attention, post-attn), optionally capturing prefix K/V."""
        vb = self.video_backbone
        q, k, v, post = vb.pre_attn_at_layer(layer_id, vstate)
        attn = _attention(q, k, v, attn_mask, vb.num_heads)
        vstate = vb.post_attn_at_layer(layer_id, vstate, attn.contiguous(), post)
        if not capture:
            return vstate, None, None
        k_prefix = _undo_video_rope(k[:, :prefix_tokens], vstate.rope_freqs[:prefix_tokens], vb.num_heads)
        return vstate, k_prefix, v[:, :prefix_tokens]

    def _video_layer_checkpointed(
        self,
        layer_id: int,
        vstate,
        attn_mask: Optional[Tensor],
        prefix_tokens: int,
        capture: bool,
        offload: bool,
    ):
        """:meth:`_video_layer` under ``torch.utils.checkpoint``.

        Same pattern as ``DualSystemMoTDriver._step_checkpointed``: the closure
        works on a shallow copy of ``vstate`` so the backward recompute never
        rewrites the outer state's ``hidden_states``.
        """

        def _run(hidden_states: Tensor):
            local = copy.copy(vstate)
            local.hidden_states = hidden_states
            local, k_prefix, v_prefix = self._video_layer(layer_id, local, attn_mask, prefix_tokens, capture)
            if capture:
                return local.hidden_states, k_prefix, v_prefix
            return (local.hidden_states,)

        if offload:
            with torch.autograd.graph.save_on_cpu():
                out = torch.utils.checkpoint.checkpoint(_run, vstate.hidden_states, use_reentrant=False)
        else:
            out = torch.utils.checkpoint.checkpoint(_run, vstate.hidden_states, use_reentrant=False)
        vstate.hidden_states = out[0]
        if capture:
            return vstate, out[1], out[2]
        return vstate, None, None

    def _video_loop(
        self,
        vstate,
        *,
        capture: bool,
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
    ):
        vb = self.video_backbone
        if int(getattr(vstate, "prefix_kv_len", 0) or 0):
            raise NotImplementedError("variant=dot does not support video backbones with prefix K/V streams.")
        tokens_per_frame = compute_video_tokens_per_frame(vstate, "DualSystemDoTArchitecture")
        s_video = int(vstate.grid_frames) * tokens_per_frame
        prefix_frames = int((vstate.extras or {}).get("clean_prefix_frames", 1))
        prefix_tokens = min(prefix_frames * tokens_per_frame, s_video)
        attn_mask = build_cross_modal_attention_mask(
            vb,
            s_video=s_video,
            s_action=0,
            video_tokens_per_frame=tokens_per_frame,
            mode=ACTION_SEES_VIDEO,
            device=vstate.hidden_states.device,
            clean_prefix_frames=prefix_frames,
        )
        checkpointed = use_gradient_checkpointing and self.training
        docked = frozenset(self._dot_video_layers)
        keys, values = [], []
        for layer_id in range(vb.num_layers):
            want = capture and layer_id in docked
            if checkpointed:
                vstate, k_prefix, v_prefix = self._video_layer_checkpointed(
                    layer_id, vstate, attn_mask, prefix_tokens, want, use_gradient_checkpointing_offload
                )
            else:
                vstate, k_prefix, v_prefix = self._video_layer(layer_id, vstate, attn_mask, prefix_tokens, want)
            if want:
                keys.append(k_prefix)
                values.append(v_prefix)
        return vstate, keys, values

    def _encode(
        self,
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        capture: bool = True,
        **pipeline_inputs,
    ):
        """Run the video DiT once. Returns ``(video_pred, fused_k, fused_v, context, context_mask)``."""
        vb = self.video_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )
        self._apply_video_mask_mode()
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Per-token t_mod with t=0 on the clean prefix, as in joint_self_attn.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        vstate, keys, values = self._video_loop(
            vstate,
            capture=capture,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        video_pred = vb.finalize(vstate)
        if not capture:
            return video_pred, None, None, action_context, action_context_mask
        fused_k, fused_v = self.dot_fusion(torch.stack(keys), torch.stack(values))
        return video_pred, fused_k, fused_v, action_context, action_context_mask

    # ------------------------------------------------------------------
    # Action head
    # ------------------------------------------------------------------

    def _predict_actions(
        self,
        noisy_actions: Tensor,
        action_timestep: Tensor,
        fused_k: Tensor,
        fused_v: Tensor,
        *,
        context: Optional[Tensor],
        context_mask: Optional[Tensor],
    ) -> Tensor:
        """Run the action head on cached fused video K/V."""
        ab = self.action_backbone
        astate = ab.prepare_state(noisy_actions, action_timestep, context=context, context_mask=context_mask)
        for idx in range(ab.num_layers):
            q_a, k_a, v_a, post = ab.pre_attn_at_layer(idx, astate)
            k_cat = torch.cat([k_a, fused_k[idx].to(dtype=k_a.dtype)], dim=1)
            v_cat = torch.cat([v_a, fused_v[idx].to(dtype=v_a.dtype)], dim=1)
            attn = _attention(q_a, k_cat, v_cat, None, ab.num_heads)
            astate = ab.post_attn_at_layer(idx, astate, attn.contiguous(), post)
        return ab.extract_prediction(astate)

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        want_actions = noisy_actions is not None and self.action_backbone is not None
        video_pred, fused_k, fused_v, context, context_mask = self._encode(
            proprio=proprio,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            capture=want_actions,
            **pipeline_inputs,
        )
        if not want_actions:
            return video_pred, None
        action_pred = self._predict_actions(
            noisy_actions, action_timestep, fused_k, fused_v, context=context, context_mask=context_mask
        )
        return video_pred, action_pred

    # ------------------------------------------------------------------
    # MIP: both branches share one video pass
    # ------------------------------------------------------------------

    def _compute_mip_action_loss(
        self,
        *,
        actions: Tensor,
        video_timesteps: Tensor,
        video_target: Tensor,
        video_timestep_ids: Tensor,
        inputs: dict,
        lambda_video: float,
        lambda_action: float,
        device: torch.device,
        dtype: torch.dtype,
        accelerator=None,  # noqa: ARG002 — one ordinary backward suffices here
    ) -> dict:
        """MIP on DoT: one video pass feeds both MIP passes of the action head.

        The video stream never sees the action, so the two branches share the
        video forward and its loss; only the head runs twice. The trainer runs a
        single ordinary backward (``backward_done=False``), as for flow matching.
        """
        B = actions.shape[0]
        num_ts = int(getattr(self.action_scheduler, "num_train_timesteps", 1000))
        t_star = self._mip_t_star
        t0 = mip_action_timesteps(0.0, B, device=device, dtype=dtype, num_train_timesteps=num_ts)
        t09 = mip_action_timesteps(t_star, B, device=device, dtype=dtype, num_train_timesteps=num_ts)

        video_pred, fused_k, fused_v, context, context_mask = self._encode(
            **self._loss_forward_kwargs(inputs), timestep=video_timesteps
        )
        loss_video = self._compute_video_loss(video_pred, video_target, video_timestep_ids, inputs, device)

        pred0 = self._predict_actions(
            mip_branch_input(actions, 0.0), t0, fused_k, fused_v, context=context, context_mask=context_mask
        )
        loss_t0 = self._masked_action_mse(pred0, actions, inputs)
        z = torch.randn_like(actions)
        anchor = mip_refine_anchor(actions, pred0, self._mip_refine_mode)
        pred1 = self._predict_actions(
            mip_branch_input(anchor, t_star, z), t09, fused_k, fused_v, context=context, context_mask=context_mask
        )
        loss_t09 = self._masked_action_mse(pred1, actions, inputs)

        loss_action = 0.5 * (loss_t0 + loss_t09)
        if lambda_video == 0:
            loss = lambda_action * loss_action
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action
        return {
            "loss": loss,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
            "loss_mip_t0": loss_t0.detach(),
            "loss_mip_t09": loss_t09.detach(),
            "backward_done": False,
        }

    # ------------------------------------------------------------------
    # Inference: one video pass, then the head per action step
    # ------------------------------------------------------------------

    def _generate_from_inputs(
        self,
        inputs_shared: dict,
        schedule,
        *,
        action_num_frames: int,
        decode_video: bool,
        active_action_mask: Optional[Tensor],
        cfg_scale_f: float,
        seed: int,
        profile: bool,
        t0: float,
    ) -> dict:
        if cfg_scale_f > 1.0:
            raise NotImplementedError("variant=dot does not support classifier-free guidance yet.")
        prefix = inputs_shared.get("first_frame_latents")
        if prefix is None:
            raise ValueError("variant=dot needs a clean conditioning frame (first_frame_latents) at inference.")
        device = self.device
        dtype = self.dtype

        # The observed frames only: no future latents, every token clean (t=0).
        encode_inputs = dict(inputs_shared)
        encode_inputs["latents"] = prefix.clone()
        encode_inputs["num_clean_prefix_frames"] = int(prefix.shape[2])
        proprio = encode_inputs.pop("proprio", None)
        v_value = float(schedule[0][0]) if schedule is not None and len(schedule) > 0 else 1000.0
        v_timestep = torch.tensor([v_value], dtype=dtype, device=device)

        t_loop = time.time()
        with torch.no_grad():
            _, fused_k, fused_v, context, context_mask = self._encode(
                proprio=proprio, **encode_inputs, timestep=v_timestep
            )
            inactive_action_dims = self._resolve_inactive_action_dims(active_action_mask, device)
            if self._action_objective == "mip":
                action_latents = self._dot_mip_actions(
                    action_num_frames, fused_k, fused_v, context, context_mask, inactive_action_dims
                )
            else:
                action_latents = self._dot_flow_actions(
                    schedule, action_num_frames, seed, fused_k, fused_v, context, context_mask, inactive_action_dims
                )

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] dot_infer: %.3fs (total %.3fs)", time.time() - t_loop, time.time() - t0)
        if decode_video:
            logger.info("variant=dot generates no future video; returning video=None.")

        actions = action_latents.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self, "normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)
        return {"video": None, "actions": actions}

    def _dot_mip_actions(self, action_num_frames, fused_k, fused_v, context, context_mask, inactive_action_dims):
        device, dtype = self.device, self.dtype
        num_ts = int(getattr(self.action_scheduler, "num_train_timesteps", 1000))
        t_star = self._mip_t_star
        a_t0 = mip_action_timesteps(0.0, 1, device=device, dtype=dtype, num_train_timesteps=num_ts)
        a_t1 = mip_action_timesteps(t_star, 1, device=device, dtype=dtype, num_train_timesteps=num_ts)
        zeros = torch.zeros(1, action_num_frames - 1, self.action_dim, device=device, dtype=dtype)
        pred0 = self._predict_actions(zeros, a_t0, fused_k, fused_v, context=context, context_mask=context_mask)
        if inactive_action_dims is not None:
            pred0 = pred0.clone()
            pred0[..., inactive_action_dims] = 0
        pred1 = self._predict_actions(
            mip_branch_input(pred0, t_star, noise=None),
            a_t1,
            fused_k,
            fused_v,
            context=context,
            context_mask=context_mask,
        )
        if inactive_action_dims is not None:
            pred1 = pred1.clone()
            pred1[..., inactive_action_dims] = 0
        return pred1

    def _dot_flow_actions(
        self, schedule, action_num_frames, seed, fused_k, fused_v, context, context_mask, inactive_action_dims
    ):
        device, dtype = self.device, self.dtype
        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        num_train_ts_a = float(self.action_scheduler.num_train_timesteps)
        inactive_action_noise = None
        for i in range(len(schedule) - 1):
            t_a = schedule[i][1]
            t_a_next = schedule[i + 1][1]
            sigma_a = t_a / num_train_ts_a
            sigma_a_next = t_a_next / num_train_ts_a
            if sigma_a == sigma_a_next:
                continue
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device)
            pred = self._predict_actions(
                action_latents, a_timestep, fused_k, fused_v, context=context, context_mask=context_mask
            )
            if inactive_action_dims is not None and inactive_action_noise is None:
                sigma_a_f = float(sigma_a)
                if sigma_a_f <= 0.0:
                    raise ValueError("Cannot initialize inactive action noise from a non-positive sigma.")
                inactive_action_noise = action_latents[..., inactive_action_dims].detach().clone() / sigma_a_f
            action_latents = self.action_scheduler.flow_step(pred, sigma_a, sigma_a_next, action_latents)
            if inactive_action_dims is not None:
                action_latents[..., inactive_action_dims] = inactive_action_noise * float(sigma_a_next)
        return action_latents


__all__ = ["DoTKVFusion", "DualSystemDoTArchitecture"]
