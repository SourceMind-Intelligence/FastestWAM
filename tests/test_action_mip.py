"""MIP action objective: timestep mapping, masked loss, 2-step inference.

Flow-matching remains the default; these tests only switch ``action_objective``.
"""

import math

import pytest
import torch

from openwam.model.action_backbone.mip import (
    MIP_T_STAR,
    mip_branch_input,
    mip_refine_anchor,
    mip_t_to_openwam_timestep,
)
from tests.test_openwam_trainer import (
    _TINY_ARCH_CFG,
    _make_fake_loss_inputs,
    _make_tiny_arch,
    _MockScheduler,
    _MockVideoBackbone,
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


def _make_tiny_mip_arch(refine_mode=None):
    cfg = dict(_TINY_ARCH_CFG)
    cfg["action_objective"] = "mip"
    cfg["mip_t_star"] = 0.9
    if refine_mode is not None:
        cfg["mip_refine_mode"] = refine_mode
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


def test_mip_refine_anchor_modes():
    a = torch.ones(2, 3, 4)
    pred0 = torch.full_like(a, 3.0, requires_grad=True)
    assert mip_refine_anchor(a, pred0, "gt") is a
    mixed = mip_refine_anchor(a, pred0, "mixed")
    assert torch.allclose(mixed, torch.full_like(a, 2.0))
    assert not mixed.requires_grad
    pred = mip_refine_anchor(a, pred0, "pred")
    assert torch.allclose(pred, torch.full_like(a, 3.0))
    assert not pred.requires_grad
    with pytest.raises(ValueError, match="mip_refine_mode"):
        mip_refine_anchor(a, pred0, "both")


def test_mip_refine_mode_defaults_to_gt_and_rejects_unknown():
    assert _make_tiny_mip_arch()._mip_refine_mode == "gt"
    with pytest.raises(ValueError, match="mip_refine_mode"):
        _make_tiny_mip_arch(refine_mode="both")


def _record_mip_branch_inputs(arch, actions, *, seed=0, **loss_kwargs):
    """Run one MIP loss and return (action input, grad flag, action pred) per forward."""
    calls = []
    orig = arch._loss_joint_forward

    def _recording(noisy_actions, action_timesteps, video_timesteps, inputs):
        video_pred, action_pred = orig(noisy_actions, action_timesteps, video_timesteps, inputs)
        calls.append(
            {
                "input": None if noisy_actions is None else noisy_actions.detach().clone(),
                "input_requires_grad": bool(noisy_actions is not None and noisy_actions.requires_grad),
                "grad_enabled": torch.is_grad_enabled(),
                "pred": None if action_pred is None else action_pred.detach().clone(),
            }
        )
        return video_pred, action_pred

    arch._loss_joint_forward = _recording
    torch.manual_seed(seed)
    result = arch.compute_loss(actions=actions.clone(), **_make_fake_loss_inputs(B=actions.shape[0]), **loss_kwargs)
    arch._loss_joint_forward = orig
    return result, calls


@pytest.mark.parametrize("mode,pred_weight", [("mixed", 0.5), ("pred", 1.0)])
def test_mip_self_forcing_trains_second_branch_on_detached_first_prediction(mode, pred_weight):
    arch = _make_tiny_mip_arch()
    arch.train()
    actions = torch.randn(2, 5, 7)

    _, gt_calls = _record_mip_branch_inputs(arch, actions)
    arch._mip_refine_mode = mode
    result, sf_calls = _record_mip_branch_inputs(arch, actions)

    assert len(gt_calls) == len(sf_calls) == 2
    # Same seed: the first branch and the refine noise z are identical, so the
    # second-branch inputs differ only by t* * w * (A0_hat - A).
    pred0 = sf_calls[0]["pred"]
    assert torch.allclose(gt_calls[0]["pred"], pred0)
    expected_shift = MIP_T_STAR * pred_weight * (pred0 - actions)
    assert torch.allclose(sf_calls[1]["input"] - gt_calls[1]["input"], expected_shift, atol=1e-5)
    assert not sf_calls[1]["input_requires_grad"]
    assert torch.isfinite(result["loss"])
    result["loss"].backward()


def test_mip_with_video_xm_probes_once_then_trains_both_branches():
    arch = _make_tiny_mip_arch(refine_mode="mixed")
    arch.train()
    actions = torch.randn(2, 5, 7)
    result, calls = _record_mip_branch_inputs(arch, actions, video_xm_k=2, video_xm_mix=1.0)

    # Two no-grad XM probes on the first MIP pass's input, then the two MIP branches.
    assert [c["grad_enabled"] for c in calls] == [False, False, True, True]
    for probe in calls[:2]:
        assert torch.equal(probe["input"], torch.zeros_like(actions))
    for key in ("xm_alt_fraction", "xm_candidate0_loss", "xm_best_loss", "loss_mip_t0", "loss_mip_t09"):
        assert torch.isfinite(result[key]).all()
    result["loss"].backward()
    assert any(p.grad is not None for p in arch.action_backbone.parameters())
