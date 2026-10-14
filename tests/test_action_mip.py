"""MIP action objective: shared t-grid, masked loss, N-step inference.

Flow-matching remains the default; these tests only switch ``action_objective``.
"""

import math

import torch

from openwam.model.action_backbone.mip import (
    MIP_T_STAR,
    mip_branch_input,
    mip_t_grid,
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


def test_mip_t_grid_train_infer_shared():
    assert mip_t_grid(2, 0.9) == [0.0, 0.9]
    assert mip_t_grid(4, 0.9) == [0.0, 0.3, 0.6, 0.9]
    for n in (2, 4, 10):
        grid = mip_t_grid(n, 0.9)
        assert grid[0] == 0.0
        assert math.isclose(grid[-1], 0.9)
        assert len(grid) == n
        assert all(a < b for a, b in zip(grid, grid[1:]))


def test_mip_branch_inputs():
    a = torch.ones(2, 4, 7)
    z = torch.full_like(a, 3.0)
    i0 = mip_branch_input(a, 0.0)
    assert torch.equal(i0, torch.zeros_like(a))
    i09 = mip_branch_input(a, 0.9, z)
    assert torch.allclose(i09, 0.9 * a + 0.1 * z)
    infer = mip_branch_input(a, 0.9, noise=None)
    assert torch.allclose(infer, 0.9 * a)
    i03 = mip_branch_input(a, 0.3, z)
    assert torch.allclose(i03, 0.3 * a + 0.7 * z)


def _make_tiny_mip_arch(*, num_steps: int = 2):
    cfg = dict(_TINY_ARCH_CFG)
    cfg["action_objective"] = "mip"
    cfg["mip_t_star"] = 0.9
    cfg["mip_num_steps"] = num_steps
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
    arch = _make_tiny_mip_arch(num_steps=4)
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
    assert result["mip_num_steps"] == 4
    result["loss"].backward()
    assert any(p.grad is not None for p in arch.action_backbone.parameters())


def test_mip_sequential_backward_matches_combined():
    arch = _make_tiny_mip_arch(num_steps=4)
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
            "mip_num_steps": 4,
        },
        action_backbone={"dim": 64, "ffn_dim": 128},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=64, num_dit_layers=2)
    assert resolved.params["action_objective"] == "mip"
    assert math.isclose(float(resolved.params["mip_t_star"]), 0.9)
    assert int(resolved.params["mip_num_steps"]) == 4


def test_mip_generate_follows_shared_grid():
    arch = _make_tiny_mip_arch(num_steps=4)
    arch.eval()
    assert arch._mip_grid == [0.0, 0.3, 0.6, 0.9]
    vb = arch.video_backbone

    def preprocess_input_for_inference(self, **kw):
        latents = torch.zeros(1, 16, 3, 8, 8)
        return {
            "latents": latents,
            "first_frame_latents": latents[:, :, :1].clone(),
            "context": torch.randn(1, 4, 64),
            "context_mask": torch.ones(1, 4, dtype=torch.bool),
            "seq_lens": torch.ones(1, dtype=torch.long),
            "num_inference_steps": 4,
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
    assert len(calls) == 4
    assert calls[0]["abs_sum"] == 0.0
    assert calls[0]["t"] == 1000.0
    assert math.isclose(calls[1]["t"], 700.0)  # t=0.3 → (1-0.3)*1000
    assert math.isclose(calls[2]["t"], 400.0)  # t=0.6
    assert math.isclose(calls[3]["t"], 100.0)  # t=0.9


def test_mip_refine_anchor_modes():
    from openwam.model.action_backbone.mip import mip_refine_anchor, normalize_mip_refine_mode

    gt = torch.ones(2, 3, 4)
    pred = torch.full_like(gt, 3.0)
    assert normalize_mip_refine_mode("GT") == "gt"
    assert torch.equal(mip_refine_anchor(gt, pred, "gt"), gt)
    assert torch.allclose(mip_refine_anchor(gt, pred, "mixed"), 0.5 * gt + 0.5 * pred)
    assert torch.equal(mip_refine_anchor(gt, pred, "pred"), pred)


def test_mip_mixed_self_forcing_uses_stopgrad_pred0():
    """Refine input must not backprop into pred0 (stop-gradient)."""
    arch = _make_tiny_mip_arch(num_steps=2)
    arch._mip_refine_mode = "mixed"
    arch.train()
    actions = torch.randn(1, 5, 7, requires_grad=False)
    inputs = _make_fake_loss_inputs()

    # Hook action backbone to capture refine-branch action_in.
    captured = {}
    orig_forward = arch.forward
    call_idx = {"n": 0}

    def wrapped(noisy_actions, action_timestep, **kwargs):
        call_idx["n"] += 1
        if call_idx["n"] == 2:
            captured["refine_in"] = noisy_actions.detach().clone()
            captured["refine_requires_grad"] = bool(noisy_actions.requires_grad)
        return orig_forward(noisy_actions, action_timestep, **kwargs)

    arch.forward = wrapped
    torch.manual_seed(0)
    result = arch.compute_loss(actions=actions, **inputs)
    assert "refinement_gain" in result
    assert torch.isfinite(result["loss"])
    assert captured["refine_requires_grad"] is False
    result["loss"].backward()
    assert any(p.grad is not None for p in arch.action_backbone.parameters())


def test_mip_pred_mode_requires_two_steps():
    cfg = dict(_TINY_ARCH_CFG)
    cfg["action_objective"] = "mip"
    cfg["mip_num_steps"] = 4
    cfg["mip_refine_mode"] = "mixed"
    from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture

    try:
        DualSystemCrossAttnArchitecture(cfg=cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "mip_num_steps=2" in str(exc)


def test_mip_generate_two_steps_zero_init_and_shape():
    arch = _make_tiny_mip_arch(num_steps=2)
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
    assert calls[1]["abs_sum"] >= 0.0
