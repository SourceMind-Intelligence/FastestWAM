"""DoT (Faster-WAM) action head: KV-Fusion over the clean prefix of the video DiT.

The mock video backbone keeps Wan's per-layer attention contract (RoPE'd q/k,
first_frame_causal mask, t=0 on the clean prefix) and is deterministic, so the
head's video inputs can be compared between training and inference.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from einops import rearrange

from openwam.model.action_backbone.mip import MIP_T_STAR
from openwam.model.architectures.dual_system.dot import DualSystemDoTArchitecture
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.wan.models.dit import rope_apply
from tests.test_openwam_trainer import _MockScheduler, _MockVideoBackbone

Z_DIM, DIM, HEADS, LAYERS, TEXT_DIM, ACTION_DIM, STATE_DIM = 4, 32, 4, 3, 16, 7, 5
H_LAT, W_LAT = 2, 3
TOKENS_PER_FRAME = H_LAT * W_LAT


class _WanLikeVideoBackbone(_MockVideoBackbone):
    """Deterministic stand-in with Wan's per-layer attention contract.

    Tokens embed the latents (patch size 1), clean-prefix tokens are conditioned
    on t=0, q/k carry a complex rotary rotation, and each layer's pre-RoPE keys
    are recorded so tests can check what the head reads.
    """

    def __init__(self):
        super().__init__(dim=DIM, num_layers=LAYERS, num_heads=HEADS)
        self.patch_embed = nn.Linear(Z_DIM, DIM)
        self.time_embed = nn.Linear(1, DIM)
        self.text_proj = nn.Linear(TEXT_DIM, DIM)
        self.q_proj = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(LAYERS)])
        self.k_proj = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(LAYERS)])
        self.v_proj = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(LAYERS)])
        self.o_proj = nn.ModuleList([nn.Linear(DIM, DIM) for _ in range(LAYERS)])
        self.out_head = nn.Linear(DIM, Z_DIM)
        self.pre_rope_keys: dict[int, torch.Tensor] = {}
        self.prepare_calls: list[dict] = []

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device, clean_prefix_frames=1):
        mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if self.video_attention_mask_mode == "first_frame_causal":
            prefix = min(video_tokens_per_frame * max(int(clean_prefix_frames), 1), video_seq_len)
            mask[:prefix, prefix:] = False
        return mask

    def prepare(self, **inputs) -> BlockLoopState:
        latents = inputs["latents"]
        B, _, F, H, W = latents.shape
        num_clean = max(int(inputs.get("num_clean_prefix_frames", 0) or 0), 1)
        self.prepare_calls.append({"frames": F, "num_clean": num_clean})
        x = self.patch_embed(rearrange(latents, "b c f h w -> b (f h w) c"))
        t = inputs["timestep"].to(x.dtype).reshape(-1).expand(B)
        token_t = t.view(B, 1, 1).expand(B, F, H * W).clone()
        token_t[:, :num_clean] = 0
        x = x + self.time_embed(token_t.reshape(B, -1, 1))
        context = inputs["context"]
        weights = inputs["context_mask"].to(x.dtype).unsqueeze(-1)
        text = (context * weights).sum(1) / weights.sum(1).clamp(min=1)
        x = x + self.text_proj(text).unsqueeze(1)
        seq = F * H * W
        half = self.head_dim // 2
        angles = torch.arange(seq, dtype=torch.float64)[:, None] * (0.5 ** torch.arange(half, dtype=torch.float64))
        freqs = torch.polar(torch.ones_like(angles), angles).unsqueeze(1)
        return BlockLoopState(
            hidden_states=x,
            time_mod=torch.zeros(B, seq, 6, DIM),
            rope_freqs=freqs,
            context=None,
            grid_frames=F,
            grid_height=H,
            grid_width=W,
            extras={"clean_prefix_frames": num_clean},
        )

    def pre_attn_at_layer(self, layer_id, state):
        x = state.hidden_states
        k = self.k_proj[layer_id](x)
        self.pre_rope_keys[layer_id] = k.detach()
        q = rope_apply(self.q_proj[layer_id](x), state.rope_freqs, self.num_heads)
        k = rope_apply(k, state.rope_freqs, self.num_heads)
        return q, k, self.v_proj[layer_id](x), {"residual": x}

    def post_attn_at_layer(self, layer_id, state, attn_out, post_state):
        hidden = post_state["residual"] + self.o_proj[layer_id](attn_out)
        state.hidden_states = hidden + torch.tanh(hidden)
        return state

    def finalize(self, state):
        out = self.out_head(state.hidden_states)
        return rearrange(out, "b (f h w) c -> b c f h w", f=state.grid_frames, h=state.grid_height, w=state.grid_width)


def _make_dot_arch(**overrides) -> DualSystemDoTArchitecture:
    cfg = {
        "framework": "dual_system",
        "variant": "dot",
        "action_dim": ACTION_DIM,
        "dim": DIM,
        "ffn_dim": 2 * DIM,
        "num_heads": HEADS,
        "attn_head_dim": DIM // HEADS,
        "num_dit_layers": LAYERS,
        "video_dim": DIM,
        "text_dim": TEXT_DIM,
        "video_attention_mask_mode": "first_frame_causal",
        **overrides,
    }
    arch = DualSystemDoTArchitecture(cfg=cfg)
    arch.video_backbone = _WanLikeVideoBackbone()
    arch.action_backbone.scheduler = _MockScheduler()
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _loss_inputs(B=2, *, frames=3, prefix=1, proprio=False) -> dict:
    latents = torch.randn(B, Z_DIM, frames, H_LAT, W_LAT)
    inputs = {
        "input_latents": latents,
        "latents": None,
        "context": torch.randn(B, 5, TEXT_DIM),
        "context_mask": torch.ones(B, 5, dtype=torch.bool),
        "first_frame_latents": latents[:, :, :prefix].clone(),
        # Wan's training preprocess marks a lone first frame with 0, history with K+1.
        "num_clean_prefix_frames": prefix if prefix > 1 else 0,
        "use_gradient_checkpointing": False,
        "use_gradient_checkpointing_offload": False,
        "max_timestep_boundary": 1.0,
        "min_timestep_boundary": 0.0,
    }
    if proprio:
        inputs["proprio"] = torch.randn(B, STATE_DIM)
    return inputs


def _clone_inputs(inputs: dict) -> dict:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}


def _spy_head(arch) -> list[dict]:
    """Record every action-head pass: its input, timestep and prediction."""
    calls = []
    orig = arch._predict_actions

    def _recording(noisy_actions, action_timestep, *args, **kwargs):
        pred = orig(noisy_actions, action_timestep, *args, **kwargs)
        calls.append(
            {
                "input": noisy_actions.detach().clone(),
                "input_requires_grad": noisy_actions.requires_grad,
                "t": action_timestep.detach().clone(),
                "pred": pred.detach().clone(),
            }
        )
        return pred

    arch._predict_actions = _recording
    return calls


def _attach_inference_inputs(arch, *, prefix=1, frames=3):
    vb = arch.video_backbone

    def preprocess_input_for_inference(self, **kw):
        gen = torch.Generator().manual_seed(123)
        return {
            "latents": torch.randn(1, Z_DIM, frames, H_LAT, W_LAT, generator=gen),
            "first_frame_latents": torch.randn(1, Z_DIM, prefix, H_LAT, W_LAT, generator=gen),
            "num_clean_prefix_frames": prefix if prefix > 1 else 0,
            "context": torch.randn(1, 5, TEXT_DIM, generator=gen),
            "context_mask": torch.ones(1, 5, dtype=torch.bool),
        }

    vb.preprocess_input_for_inference = preprocess_input_for_inference.__get__(vb, type(vb))


def _generate(arch, schedule, **kwargs):
    return arch.generate(
        schedule,
        prompt="stack the bowls",
        num_frames=9,
        action_num_frames=9,
        height=8 * H_LAT,
        width=8 * W_LAT,
        decode_video=False,
        seed=0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_dot_builds_a_one_layer_head_with_identity_uniform_fusion():
    arch = _make_dot_arch()
    assert arch.action_backbone.num_layers == 1
    fusion = arch.dot_fusion
    assert fusion.layer_mix.shape == (1, HEADS, LAYERS)
    assert torch.allclose(fusion.layer_mix, torch.full_like(fusion.layer_mix, 1.0 / LAYERS))
    assert torch.equal(fusion.k_proj.weight, torch.eye(DIM))
    assert torch.equal(fusion.v_proj.weight, torch.eye(DIM))


def test_dot_video_layer_selection_and_validation():
    arch = _make_dot_arch(dot_video_layers=[2, 0], dot_num_action_layers=2)
    assert arch._dot_video_layers == (0, 2)
    assert arch.action_backbone.num_layers == 2
    assert arch.dot_fusion.layer_mix.shape == (2, HEADS, 2)
    with pytest.raises(ValueError, match="dot_video_layers"):
        _make_dot_arch(dot_video_layers=[LAYERS])
    with pytest.raises(ValueError, match="dot_num_action_layers"):
        _make_dot_arch(dot_num_action_layers=0)


def test_dot_requires_first_frame_causal_video_attention():
    with pytest.raises(ValueError, match="first_frame_causal"):
        _make_dot_arch(video_attention_mask_mode="bidirectional")


def test_dot_variant_resolves_from_the_model_config():
    from types import SimpleNamespace

    from openwam.model import resolve_architecture_config

    model_cfg = SimpleNamespace(
        architecture={
            "framework": "dual_system",
            "variant": "dot",
            "action_dim": 80,
            "action_objective": "mip",
            "mip_refine_mode": "mixed",
            "dot_num_action_layers": 1,
        },
        action_backbone={"dim": 64, "ffn_dim": 128},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=64, num_dit_layers=2)
    assert resolved.registry_name == "dual_system_dot"
    assert resolved.params["mip_refine_mode"] == "mixed"


def test_dot_set_dtype_device_moves_the_fusion():
    arch = _make_dot_arch()
    arch.set_dtype_device(torch.float64, torch.device("cpu"))
    assert arch.dot_fusion.layer_mix.dtype == torch.float64
    assert arch.dot_fusion.k_proj.weight.dtype == torch.float64


# ---------------------------------------------------------------------------
# What the head reads
# ---------------------------------------------------------------------------


def test_dot_fusion_reads_pre_rope_keys_of_the_clean_prefix_only():
    torch.manual_seed(0)
    arch = _make_dot_arch()
    arch.eval()
    vb = arch.video_backbone
    seen = {}
    orig = arch.dot_fusion.forward

    def _spy(keys, values):
        seen["keys"], seen["values"] = keys.detach(), values.detach()
        return orig(keys, values)

    arch.dot_fusion.forward = _spy
    inputs = _loss_inputs()
    with torch.no_grad():
        _, fused_k, fused_v, _, _ = arch._encode(
            latents=inputs["input_latents"],
            timestep=torch.full((2,), 0.5),
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            first_frame_latents=inputs["first_frame_latents"],
        )

    expected_keys = torch.stack([vb.pre_rope_keys[i][:, :TOKENS_PER_FRAME] for i in range(LAYERS)])
    assert seen["keys"].shape == (LAYERS, 2, TOKENS_PER_FRAME, DIM)
    assert torch.allclose(seen["keys"], expected_keys, atol=1e-5)
    # Identity projections and a uniform layer mix: fused values are the layer mean.
    assert fused_v.shape == (1, 2, TOKENS_PER_FRAME, DIM)
    assert torch.allclose(fused_v[0], seen["values"].mean(dim=0), atol=1e-6)
    assert fused_k.shape == fused_v.shape


@pytest.mark.parametrize("prefix", [1, 2])
def test_dot_head_reads_the_same_video_kv_in_training_and_inference(prefix):
    torch.manual_seed(0)
    arch = _make_dot_arch()
    arch.eval()
    B, frames = 2, 4
    clean = torch.randn(B, Z_DIM, frames, H_LAT, W_LAT)
    common = {
        "context": torch.randn(B, 5, TEXT_DIM),
        "context_mask": torch.ones(B, 5, dtype=torch.bool),
        "first_frame_latents": clean[:, :, :prefix].clone(),
    }

    def fused(latents, timestep, num_clean):
        with torch.no_grad():
            _, k, v, _, _ = arch._encode(
                latents=latents, timestep=timestep, num_clean_prefix_frames=num_clean, **common
            )
        return k, v

    # Training: noisy future frames behind the clean prefix, at two noise levels.
    train_kv = []
    for sigma in (0.3, 0.9):
        noisy = clean.clone()
        noisy[:, :, prefix:] = torch.randn_like(noisy[:, :, prefix:])
        train_kv.append(fused(noisy, torch.full((B,), sigma), prefix if prefix > 1 else 0))
    # Inference: the observed frames alone.
    infer_k, infer_v = fused(clean[:, :, :prefix].clone(), torch.tensor([1000.0]), prefix)

    assert infer_k.shape == (1, B, prefix * TOKENS_PER_FRAME, DIM)
    for k, v in train_kv:
        assert torch.allclose(k, infer_k, atol=1e-5)
        assert torch.allclose(v, infer_v, atol=1e-5)


def test_dot_video_prediction_ignores_the_action_stream():
    torch.manual_seed(0)
    arch = _make_dot_arch()
    arch.eval()
    inputs = _loss_inputs()
    kw = {
        "latents": inputs["input_latents"],
        "timestep": torch.full((2,), 0.5),
        "context": inputs["context"],
        "context_mask": inputs["context_mask"],
        "first_frame_latents": inputs["first_frame_latents"],
    }
    t_action = torch.full((2,), 500.0)
    with torch.no_grad():
        video_only, no_actions = arch(None, None, **kw)
        video_1, actions_1 = arch(torch.randn(2, 5, ACTION_DIM), t_action, **kw)
        video_2, actions_2 = arch(torch.randn(2, 5, ACTION_DIM), t_action, **kw)

    assert no_actions is None
    assert actions_1.shape == (2, 5, ACTION_DIM)
    assert torch.allclose(video_only, video_1, atol=1e-6)
    assert torch.allclose(video_1, video_2, atol=1e-6)
    assert not torch.allclose(actions_1, actions_2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_dot_flow_action_loss_trains_the_head_the_fusion_and_the_video_dit():
    torch.manual_seed(0)
    arch = _make_dot_arch(use_proprioception=True, state_dim=STATE_DIM)
    arch.train()
    vb = arch.video_backbone
    result = arch.compute_loss(actions=torch.randn(2, 5, ACTION_DIM), lambda_video=0.0, **_loss_inputs(proprio=True))
    assert "loss_mip_t0" not in result
    result["loss"].backward()

    def grad_norm(p):
        return 0.0 if p.grad is None else float(p.grad.abs().sum())

    assert grad_norm(arch.action_backbone.action_decoder.weight) > 0
    assert grad_norm(arch.dot_fusion.layer_mix) > 0
    assert grad_norm(arch.dot_fusion.k_proj.weight) > 0
    assert grad_norm(arch.proprio_encoder.weight) > 0
    # With no video loss, the last video layer learns only through the fused K/V.
    assert grad_norm(vb.k_proj[LAYERS - 1].weight) > 0
    assert grad_norm(vb.out_head.weight) == 0


@pytest.mark.parametrize("mode", ["gt", "mixed"])
def test_dot_mip_runs_one_video_pass_and_two_head_passes(mode):
    torch.manual_seed(0)
    arch = _make_dot_arch(action_objective="mip", mip_refine_mode=mode)
    arch.train()
    vb = arch.video_backbone
    head = _spy_head(arch)
    actions = torch.randn(2, 5, ACTION_DIM)
    result = arch.compute_loss(actions=actions, **_loss_inputs())

    assert len(vb.prepare_calls) == 1
    assert len(head) == 2
    assert torch.equal(head[0]["input"], torch.zeros_like(actions))
    assert torch.allclose(head[0]["t"], torch.full((2,), 1000.0))
    assert torch.allclose(head[1]["t"], torch.full((2,), (1.0 - MIP_T_STAR) * 1000.0))
    assert result["backward_done"] is False
    for key in ("loss", "loss_video", "loss_action", "loss_mip_t0", "loss_mip_t09"):
        assert torch.isfinite(result[key]).all()
    result["loss"].backward()
    assert arch.action_backbone.action_decoder.weight.grad is not None
    assert arch.dot_fusion.layer_mix.grad is not None


def test_dot_mip_self_forcing_moves_the_second_head_input_by_the_detached_first_prediction():
    torch.manual_seed(0)
    arch = _make_dot_arch(action_objective="mip")
    arch.train()
    actions = torch.randn(2, 5, ACTION_DIM)
    inputs = _loss_inputs()

    runs = {}
    for mode in ("gt", "mixed"):
        arch._mip_refine_mode = mode
        head = _spy_head(arch)
        torch.manual_seed(1)
        arch.compute_loss(actions=actions.clone(), **_clone_inputs(inputs))
        del arch._predict_actions
        runs[mode] = head

    pred0 = runs["mixed"][0]["pred"]
    assert torch.allclose(runs["gt"][0]["pred"], pred0)
    shift = runs["mixed"][1]["input"] - runs["gt"][1]["input"]
    assert torch.allclose(shift, MIP_T_STAR * 0.5 * (pred0 - actions), atol=1e-5)
    assert not runs["mixed"][1]["input_requires_grad"]


def test_dot_mip_with_video_xm_probes_twice_then_trains_once():
    torch.manual_seed(0)
    arch = _make_dot_arch(action_objective="mip", mip_refine_mode="mixed")
    arch.train()
    vb = arch.video_backbone
    result = arch.compute_loss(actions=torch.randn(2, 5, ACTION_DIM), video_xm_k=2, video_xm_mix=1.0, **_loss_inputs())
    assert len(vb.prepare_calls) == 3
    for key in ("xm_alt_fraction", "xm_candidate0_loss", "xm_best_loss", "loss_mip_t0", "loss_mip_t09"):
        assert torch.isfinite(result[key]).all()
    result["loss"].backward()


@pytest.mark.parametrize("objective", ["flow", "mip"])
def test_dot_gradient_checkpointing_matches_the_plain_backward(objective):
    torch.manual_seed(0)
    inputs = _loss_inputs(prefix=2, frames=4)
    actions = torch.randn(2, 5, ACTION_DIM)
    losses, grads = {}, {}
    for checkpointed in (False, True):
        torch.manual_seed(0)
        arch = _make_dot_arch(action_objective=objective)
        arch.train()
        run_inputs = _clone_inputs(inputs)
        run_inputs["use_gradient_checkpointing"] = checkpointed
        torch.manual_seed(1)
        result = arch.compute_loss(actions=actions.clone(), **run_inputs)
        result["loss"].backward()
        losses[checkpointed] = result["loss"].detach()
        grads[checkpointed] = {n: p.grad.clone() for n, p in arch.named_parameters() if p.grad is not None}

    assert torch.allclose(losses[False], losses[True])
    assert grads[False].keys() == grads[True].keys()
    for name, grad in grads[False].items():
        assert torch.allclose(grad, grads[True][name], atol=1e-5), name


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_dot_warm_start_keeps_the_video_weights_and_starts_the_head_fresh():
    arch = _make_dot_arch()
    own = {k: v.clone() for k, v in arch.state_dict().items()}
    fresh = ("action_backbone.", "dot_fusion.")
    foreign = {k: v + 1.0 for k, v in own.items() if not k.startswith(fresh)}
    # A deep MoT expert's keys, including one whose shape does not fit the DoT head.
    foreign["action_backbone.blocks.29.self_attn.q.weight"] = torch.zeros(3, 3)
    foreign["action_backbone.action_encoder.weight"] = torch.zeros(3, 3)

    adapted = arch.adapt_checkpoint_state_dict(foreign)
    assert set(adapted) == set(own)
    for key, value in own.items():
        expected = value if key.startswith(fresh) else foreign[key]
        assert torch.equal(adapted[key], expected), key
    arch.load_state_dict(adapted, strict=True)

    dot_ckpt = arch.state_dict()
    assert arch.adapt_checkpoint_state_dict(dot_ckpt) is dot_ckpt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_dot_generate_runs_the_video_once_then_the_head_per_step():
    torch.manual_seed(0)
    arch = _make_dot_arch()
    _attach_inference_inputs(arch)
    vb = arch.video_backbone
    head = _spy_head(arch)
    out = _generate(arch, [(1000.0, 1000.0), (500.0, 500.0), (0.0, 0.0)])

    assert out["video"] is None
    assert out["actions"].shape == (8, ACTION_DIM)
    assert vb.prepare_calls == [{"frames": 1, "num_clean": 1}]
    assert [float(call["t"][0]) for call in head] == [1000.0, 500.0]


def test_dot_generate_with_history_encodes_only_the_observed_frames():
    torch.manual_seed(0)
    arch = _make_dot_arch()
    _attach_inference_inputs(arch, prefix=3, frames=5)
    vb = arch.video_backbone
    out = _generate(arch, [(1000.0, 1000.0), (0.0, 0.0)])
    assert out["actions"].shape == (8, ACTION_DIM)
    assert vb.prepare_calls == [{"frames": 3, "num_clean": 3}]


def test_dot_mip_generate_is_two_head_passes_from_zeros():
    torch.manual_seed(0)
    arch = _make_dot_arch(action_objective="mip", use_proprioception=True, state_dim=STATE_DIM)
    _attach_inference_inputs(arch)
    vb = arch.video_backbone
    head = _spy_head(arch)
    out = _generate(arch, [(1000.0, 1000.0), (0.0, 0.0)], proprio=torch.randn(1, STATE_DIM))

    assert out["actions"].shape == (8, ACTION_DIM)
    assert len(vb.prepare_calls) == 1
    assert len(head) == 2
    assert torch.equal(head[0]["input"], torch.zeros(1, 8, ACTION_DIM))
    assert [float(call["t"][0]) for call in head] == [1000.0, pytest.approx((1.0 - MIP_T_STAR) * 1000.0)]
    assert torch.allclose(head[1]["input"], MIP_T_STAR * head[0]["pred"])
    assert torch.allclose(torch.from_numpy(out["actions"]), head[1]["pred"][0])


@pytest.mark.parametrize(
    ("mode", "refine_mode", "scale"),
    [
        (None, "mixed", 0.5),
        (None, "gt", 0.0),
        ("zero", "mixed", 0.0),
        ("anchor", "mixed", 0.5),
        ("anchor", "gt", 0.0),
        ("keep", "mixed", 1.0),
    ],
)
def test_dot_mip_generate_feeds_unused_dims_to_the_second_pass_by_mode(monkeypatch, mode, refine_mode, scale):
    if mode is None:
        monkeypatch.delenv("OPENWAM_DOT_MIP_UNUSED_DIMS", raising=False)
    else:
        monkeypatch.setenv("OPENWAM_DOT_MIP_UNUSED_DIMS", mode)
    torch.manual_seed(0)
    arch = _make_dot_arch(action_objective="mip", mip_refine_mode=refine_mode)
    _attach_inference_inputs(arch)
    head = _spy_head(arch)
    active = torch.tensor([True, True, True, True, False, False, False])
    out = _generate(arch, [(1000.0, 1000.0), (0.0, 0.0)], active_action_mask=active)

    pred0 = head[0]["pred"]
    assert torch.allclose(head[1]["input"][..., active], MIP_T_STAR * pred0[..., active])
    assert torch.allclose(head[1]["input"][..., ~active], MIP_T_STAR * scale * pred0[..., ~active])
    assert pred0[..., ~active].abs().min() > 0
    assert torch.all(torch.from_numpy(out["actions"])[:, ~active] == 0)


def test_dot_mip_generate_rejects_an_unknown_unused_dims_mode(monkeypatch):
    monkeypatch.setenv("OPENWAM_DOT_MIP_UNUSED_DIMS", "half")
    arch = _make_dot_arch(action_objective="mip")
    _attach_inference_inputs(arch)
    with pytest.raises(ValueError, match="OPENWAM_DOT_MIP_UNUSED_DIMS"):
        _generate(arch, [(1000.0, 1000.0), (0.0, 0.0)], active_action_mask=torch.tensor([True] * 4 + [False] * 3))


def test_dot_generate_rejects_classifier_free_guidance():
    arch = _make_dot_arch()
    _attach_inference_inputs(arch)
    with pytest.raises(NotImplementedError, match="classifier-free guidance"):
        _generate(arch, [(1000.0, 1000.0), (0.0, 0.0)], cfg_scale=2.0)
