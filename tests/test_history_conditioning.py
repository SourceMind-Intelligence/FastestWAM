"""History (memory) conditioning: past frames as a clean TI2V prefix."""

from __future__ import annotations

import types

import torch
from PIL import Image

from openwam.deploy.policy import WAMPolicy
from openwam.model.architectures.utils.mask_modes import MUTUAL, build_cross_modal_attention_mask
from openwam.model.video_backbone.wan import conditioning as wan_conditioning
from openwam.model.video_backbone.wan_backbone import WanBase


def _wan_mask_owner():
    owner = types.SimpleNamespace(video_attention_mask_mode="first_frame_causal")
    owner.build_video_to_video_mask = lambda **kw: WanBase.build_video_to_video_mask(owner, **kw)
    return owner


def test_single_clean_frame_mask_is_unchanged():
    owner = _wan_mask_owner()
    default = build_cross_modal_attention_mask(
        owner, s_video=12, s_action=3, video_tokens_per_frame=4, mode=MUTUAL, device=torch.device("cpu")
    )
    explicit = build_cross_modal_attention_mask(
        owner,
        s_video=12,
        s_action=3,
        video_tokens_per_frame=4,
        mode=MUTUAL,
        device=torch.device("cpu"),
        clean_prefix_frames=1,
    )
    assert torch.equal(default, explicit)
    assert not default[:4, 4:].any()  # first frame sees only itself
    assert default[4:12, :15].all()  # later video rows see everything


def test_clean_prefix_rows_see_only_the_prefix():
    tpf, prefix = 4, 3  # two history frames + the first frame
    owner = _wan_mask_owner()
    mask = build_cross_modal_attention_mask(
        owner,
        s_video=5 * tpf,
        s_action=3,
        video_tokens_per_frame=tpf,
        mode=MUTUAL,
        device=torch.device("cpu"),
        clean_prefix_frames=prefix,
    )
    p = prefix * tpf
    assert mask[:p, :p].all()
    assert not mask[:p, p:].any()  # no noisy video, no action
    assert mask[p : 5 * tpf, :].all()  # noisy video sees prefix, itself and action (mutual)
    assert mask[5 * tpf :, :].all()  # action sees everything


class _FakeEncoder:
    """Stands in for the VAE: one latent frame per one-frame clip, value = pixel mean."""

    def preprocess_video(self, frames):
        arr = torch.tensor([float(f.getpixel((0, 0))) for f in frames])
        return arr.view(1, 1, len(frames), 1, 1).expand(1, 2, len(frames), 2, 2).clone()

    def batch_encode(self, video):
        return video


def test_encode_history_latents_orders_frames_oldest_first():
    imgs = [[Image.new("L", (2, 2), c) for c in (10, 20, 30)], [Image.new("L", (2, 2), c) for c in (40, 50, 60)]]
    lat = wan_conditioning.encode_history_latents(
        imgs, encoder=_FakeEncoder(), vae=None, dtype=torch.float32, device=torch.device("cpu")
    )
    assert lat.shape == (2, 2, 3, 2, 2)
    assert lat[0, 0, :, 0, 0].tolist() == [10.0, 20.0, 30.0]
    assert lat[1, 0, :, 0, 0].tolist() == [40.0, 50.0, 60.0]


def test_prepend_history_extends_prefix_and_latents():
    first = torch.full((1, 2, 1, 2, 2), 7.0)
    noise = torch.randn(1, 2, 3, 2, 2)
    inputs = {"first_frame_latents": first, "latents": noise, "noise": noise, "num_clean_prefix_frames": 0}
    history = torch.full((1, 2, 2, 2, 2), 1.0)
    wan_conditioning.prepend_history_to_ti2v_inputs(inputs, history)
    assert inputs["num_clean_prefix_frames"] == 3
    assert inputs["first_frame_latents"].shape[2] == 3
    assert inputs["latents"].shape[2] == 5
    torch.testing.assert_close(inputs["latents"][:, :, 2:], noise)
    torch.testing.assert_close(inputs["first_frame_latents"][:, :, 2:], first)


class _RecordingEngine:
    def __init__(self):
        self.conditions = []

    def generate(self, conditions):
        self.conditions.append(conditions)
        return {"actions": torch.zeros(1, 4).numpy()}


def _cfg(num, stride, include_first_frame=False):
    return types.SimpleNamespace(
        dataloader=types.SimpleNamespace(
            history_num_frames=num, history_stride=stride, history_include_first_frame=include_first_frame
        )
    )


def test_policy_history_matches_training_sampling_and_resets():
    engine = _RecordingEngine()
    policy = WAMPolicy(engine, _cfg(2, 3), execution_config={"mode": "sync", "inference_horizon": 1})
    frames = [Image.new("L", (1, 1), i) for i in range(8)]
    for img in frames:
        policy.predict_action({"image": img, "prompt": "p"})

    def ids(cond):
        return [f.getpixel((0, 0)) for f in cond["history_images"]]

    # Step t sees frames max(t-6, 0) and max(t-3, 0), oldest first.
    assert ids(engine.conditions[0]) == [0, 0]
    assert ids(engine.conditions[4]) == [0, 1]
    assert ids(engine.conditions[7]) == [1, 4]

    policy.reset()
    policy.predict_action({"image": frames[5], "prompt": "p"})
    assert ids(engine.conditions[-1]) == [5, 5]


def test_policy_first_frame_slot_survives_the_bounded_buffer_and_resets():
    engine = _RecordingEngine()
    policy = WAMPolicy(
        engine, _cfg(2, 3, include_first_frame=True), execution_config={"mode": "sync", "inference_horizon": 1}
    )
    frames = [Image.new("L", (1, 1), i) for i in range(12)]
    for img in frames:
        policy.predict_action({"image": img, "prompt": "p"})

    def ids(cond):
        return [f.getpixel((0, 0)) for f in cond["history_images"]]

    # The buffer keeps 7 frames, so frame 0 is long gone from it by step 11;
    # the oldest slot still holds it and the other slot is frame t-3.
    assert ids(engine.conditions[0]) == [0, 0]
    assert ids(engine.conditions[4]) == [0, 1]
    assert ids(engine.conditions[11]) == [0, 8]

    policy.reset()
    policy.predict_action({"image": frames[5], "prompt": "p"})
    policy.predict_action({"image": frames[6], "prompt": "p"})
    assert ids(engine.conditions[-1]) == [5, 5]


def test_policy_without_history_sends_none():
    engine = _RecordingEngine()
    policy = WAMPolicy(engine, types.SimpleNamespace(), execution_config={"mode": "sync", "inference_horizon": 1})
    policy.predict_action({"image": Image.new("L", (1, 1), 0), "prompt": "p"})
    assert "history_images" not in engine.conditions[0]


def test_compute_loss_pins_the_whole_clean_prefix_and_skips_it_in_the_loss():
    from tests.test_openwam_trainer import _make_fake_loss_inputs, _make_tiny_arch, _MockScheduler

    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()
    inputs = _make_fake_loss_inputs(B=2)
    prefix = inputs["input_latents"][:, :, :2].clone()  # one history frame + the first frame
    inputs["first_frame_latents"] = prefix
    inputs["num_clean_prefix_frames"] = 2

    seen = {}
    original = arch._loss_joint_forward

    def spy(noisy_actions, action_timesteps, video_timesteps, loss_inputs):
        seen["latents"] = loss_inputs["latents"].clone()
        return original(noisy_actions, action_timesteps, video_timesteps, loss_inputs)

    arch._loss_joint_forward = spy
    result = arch.compute_loss(**inputs, actions=torch.randn(2, 5, 7))

    torch.testing.assert_close(seen["latents"][:, :, :2], prefix)
    assert not torch.allclose(seen["latents"][:, :, 2:], inputs["input_latents"][:, :, 2:])
    assert torch.isfinite(result["loss"])
