"""Shallow action expert: warm-starting fewer action blocks from a full-depth checkpoint."""

from __future__ import annotations

import types

import pytest
import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from tests.test_mot_driver import _make_action_dit


def _arch_with(action_backbone):
    # adapt_checkpoint_state_dict only reads ``self.action_backbone``.
    return types.SimpleNamespace(action_backbone=action_backbone)


def _full_depth_state_dict(num_layers: int) -> dict:
    deep = _make_action_dit(dim=32, num_heads=4, num_layers=num_layers)
    for i, block in enumerate(deep.blocks):
        torch.nn.init.constant_(block.self_attn.q.weight, float(i))
    return {f"action_backbone.{k}": v for k, v in deep.state_dict().items()}


def test_full_depth_checkpoint_maps_onto_shallow_blocks():
    shallow = _make_action_dit(dim=32, num_heads=4, bridge_layers=(1, 3))
    sd = _full_depth_state_dict(4)
    sd["video_backbone.some.weight"] = torch.ones(1)

    adapted = DualSystemSelfAttnArchitecture.adapt_checkpoint_state_dict(_arch_with(shallow), sd)

    assert adapted["video_backbone.some.weight"] is sd["video_backbone.some.weight"]
    assert not any(k.startswith("action_backbone.blocks.2.") for k in adapted)
    torch.testing.assert_close(
        adapted["action_backbone.blocks.0.self_attn.q.weight"], sd["action_backbone.blocks.1.self_attn.q.weight"]
    )
    torch.testing.assert_close(
        adapted["action_backbone.blocks.1.self_attn.q.weight"], sd["action_backbone.blocks.3.self_attn.q.weight"]
    )
    own = {k[len("action_backbone.") :]: v for k, v in adapted.items() if k.startswith("action_backbone.")}
    missing, unexpected = shallow.load_state_dict(own, strict=True)
    assert not missing and not unexpected


def test_shallow_checkpoint_loads_unchanged():
    shallow = _make_action_dit(dim=32, num_heads=4, bridge_layers=(1, 3))
    sd = {f"action_backbone.{k}": v for k, v in shallow.state_dict().items()}
    adapted = DualSystemSelfAttnArchitecture.adapt_checkpoint_state_dict(_arch_with(shallow), sd)
    assert adapted is sd


def test_checkpoint_without_needed_layer_is_rejected():
    shallow = _make_action_dit(dim=32, num_heads=4, bridge_layers=(1, 5))
    with pytest.raises(ValueError, match="bridge_layers"):
        DualSystemSelfAttnArchitecture.adapt_checkpoint_state_dict(_arch_with(shallow), _full_depth_state_dict(4))
