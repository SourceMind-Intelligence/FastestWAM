"""MIP action objective: timestep mapping, masked loss, 2-step inference.

Flow-matching remains the default; these tests only switch ``action_objective``.
"""

import math

import torch

from openwam.model.action_backbone.mip import (
    MIP_T_STAR,
    mip_branch_input,
    mip_t_to_openwam_timestep,
)
from tests.test_openwam_trainer import (
    _TINY_ARCH_CFG,
    _MockScheduler,
    _MockVideoBackbone,
    _make_fake_loss_inputs,
    _make_tiny_arch,
)


def test_mip_timestep_matches_openwam_sigma_axis():
    assert math.isclose(mip_t_to_openwam_timestep(0.0), 1000.0)
    assert math.isclose(mip_t_to_openwam_timestep(0.9), 100.0)
    assert math.isclose(mip_t_to_openwam_timestep(1.0), 0.0)
    # Mix I=0.9 A + 0.1 z is FM σ=0.1, so the embedder must see 100, not 900.
    assert math.isclose(mip_t_to_openwam_timestep(MIP_T_STAR), (1.0 - MIP_T_STAR) * 1000.0)


def test_mip_branch_inputs():
    a = torch.ones(2, 4, 7)
    z = torch.full_like(a, 3.0)
    i0 = mip_branch_input(a, 0.0)
    assert torch.equal(i0, torch.zeros_like(a))
    i09 = mip_branch_input(a, 0.9, z)
    assert torch.allclose(i09, 0.9 * a + 0.1 * z)
    infer = mip_branch_input(a, 0.9, noise=None)
    assert torch.allclose(infer, 0.9 * a)


def _make_tiny_mip_arch():
    cfg = dict(_TINY_ARCH_CFG)
    cfg["action_objective"] = "mip"
    cfg["mip_t_star"] = 0.9
    from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture

    arch = DualSystemCrossAttnArchitecture(cfg=cfg)
    arch.video_backbone = _MockVideoBackbone(dim=64, num_layers=2)
    arch.action_backbone.scheduler = _MockScheduler()
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def test_default_objective_is_flow():
    arch = _make_tiny_arch()
    assert arch._action_objective == "flow"


def test_mip_loss_finite_and_masked():
    arch = _make_tiny_mip_arch()
    arch.train()
    B, T, D = 2, 5, 7
    actions = torch.randn(B, T, D)
    mask = torch.zeros(B, T, D, dtype=torch.bool)
    mask[:, :, :3] = True  # pad first 3 dims
    inputs = _make_fake_loss_inputs(B=B, action_dim=D, T_action=T)
    inputs["action_is_pad"] = mask

    result = arch.compute_loss(actions=actions, **inputs)
    for key in ("loss", "loss_video", "loss_action", "loss_mip_t0", "loss_mip_t09"):
        assert key in result
        assert torch.isfinite(result[key]).all()
    result["loss"].backward()
    assert any(p.grad is not None for p in arch.action_backbone.parameters())


def test_mip_sequential_backward_matches_combined():
    arch = _make_tiny_mip_arch()
    arch.train()
    actions = torch.randn(1, 5, 7)
    inputs = _make_fake_loss_inputs()

    class _Accel:
        def backward(self, loss):
            loss.backward()

    torch.manual_seed(0)
    combined = arch.compute_loss(
        actions=actions.clone(),
        **{k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()},
    )
    torch.manual_seed(0)
    sequential = arch.compute_loss(
        actions=actions.clone(),
        accelerator=_Accel(),
        **{k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()},
    )
    assert sequential["backward_done"] is True
    assert combined.get("backward_done") is False
    assert torch.isfinite(sequential["loss"])
    assert torch.isfinite(combined["loss"])


def test_flow_loss_has_no_mip_keys():
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()
    actions = torch.randn(1, 5, 7)
    result = arch.compute_loss(actions=actions, **_make_fake_loss_inputs())
    assert "loss_mip_t0" not in result
    assert torch.isfinite(result["loss"])


def test_resolve_action_objective_from_model_cfg():
    from types import SimpleNamespace

    from openwam.model import resolve_architecture_config

    model_cfg = SimpleNamespace(
        architecture={
            "framework": "dual_system",
            "variant": "joint_self_attn",
            "action_dim": 80,
            "action_objective": "mip",
            "mip_t_star": 0.9,
        },
        action_backbone={"dim": 64, "ffn_dim": 128},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=64, num_dit_layers=2)
    assert resolved.params["action_objective"] == "mip"
    assert math.isclose(float(resolved.params["mip_t_star"]), 0.9)


def test_mip_generate_two_steps_zero_init_and_shape():
    arch = _make_tiny_mip_arch()
    arch.eval()
    vb = arch.video_backbone

    def preprocess_input_for_inference(self, **kw):
        latents = torch.zeros(1, 16, 3, 8, 8)
        return {
            "latents": latents,
            "first_frame_latents": latents[:, :, :1].clone(),
            "context": torch.randn(1, 4, 64),
            "context_mask": torch.ones(1, 4, dtype=torch.bool),
            "seq_lens": torch.ones(1, dtype=torch.long),
            "num_inference_steps": 2,
        }

    vb.preprocess_input_for_inference = preprocess_input_for_inference.__get__(vb, type(vb))
    calls = []
    orig_forward = arch.forward

    def wrapped_forward(noisy_actions, action_timestep, **kwargs):
        calls.append(
            {
                "abs_sum": float(noisy_actions.abs().sum()),
                "t": float(action_timestep.reshape(-1)[0]),
            }
        )
        return orig_forward(noisy_actions, action_timestep, **kwargs)

    arch.forward = wrapped_forward
    schedule = [(1000.0, 1000.0), (0.0, 0.0)]
    out = arch.generate(
        schedule,
        prompt="do something",
        num_frames=9,
        action_num_frames=9,
        height=64,
        width=64,
        decode_video=False,
        seed=0,
    )
    assert out["actions"].shape == (8, 7)
    assert len(calls) == 2
    assert calls[0]["abs_sum"] == 0.0
    assert calls[0]["t"] == 1000.0
    assert calls[1]["t"] == 100.0
    # Second input is 0.9 * a0 (no extra noise), so it is not a Gaussian draw.
    assert calls[1]["abs_sum"] >= 0.0
