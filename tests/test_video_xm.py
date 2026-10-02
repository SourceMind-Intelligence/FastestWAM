"""Forward XM (K=2 video-noise selection) on the flow-matching action path."""

import pytest
import torch

from tests.test_openwam_trainer import _make_fake_loss_inputs, _make_tiny_arch, _MockScheduler


def _record_forwards(arch):
    calls = []
    orig = arch._loss_joint_forward

    def _recording(noisy_actions, action_timesteps, video_timesteps, inputs):
        calls.append({"grad_enabled": torch.is_grad_enabled(), "latents": inputs["latents"].detach().clone()})
        return orig(noisy_actions, action_timesteps, video_timesteps, inputs)

    arch._loss_joint_forward = _recording
    return calls


def test_flow_xm_probes_twice_then_replays_the_selected_noise():
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()
    arch.train()
    calls = _record_forwards(arch)
    torch.manual_seed(0)
    result = arch.compute_loss(
        actions=torch.randn(2, 5, 7), video_xm_k=2, video_xm_mix=1.0, **_make_fake_loss_inputs(B=2)
    )

    assert [c["grad_enabled"] for c in calls] == [False, False, True]
    # The replayed latents are, per sample, one of the two probed assignments.
    for b in range(2):
        replay = calls[2]["latents"][b]
        assert torch.equal(replay, calls[0]["latents"][b]) or torch.equal(replay, calls[1]["latents"][b])
    for key in ("xm_alt_fraction", "xm_candidate0_loss", "xm_best_loss"):
        assert torch.isfinite(result[key])
    assert float(result["xm_best_loss"]) <= float(result["xm_candidate0_loss"]) + 1e-6


def test_xm_rejects_bad_settings():
    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()
    with pytest.raises(ValueError, match="K=2"):
        arch.compute_loss(actions=torch.randn(1, 5, 7), video_xm_k=3, **_make_fake_loss_inputs())
    with pytest.raises(ValueError, match="video_xm_mix"):
        arch.compute_loss(actions=torch.randn(1, 5, 7), video_xm_k=2, video_xm_mix=1.5, **_make_fake_loss_inputs())
