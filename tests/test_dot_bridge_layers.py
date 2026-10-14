"""DoT bridge selection: evenly spaced video layers for shallow ActionDiT."""

from openwam.model.architectures.utils.common import (
    evenly_spaced_bridge_layers,
    resolve_bridge_layers,
)


def test_evenly_spaced_includes_last():
    assert evenly_spaced_bridge_layers(30, 1) == (29,)
    assert evenly_spaced_bridge_layers(30, 4) == (0, 10, 19, 29)
    assert evenly_spaced_bridge_layers(30, 2) == (0, 29)


def test_resolve_dot_num_layers():
    bl = resolve_bridge_layers({"bridge_layers": None, "dot_num_layers": 4}, num_layers=30)
    assert bl == (0, 10, 19, 29)


def test_explicit_bridge_layers_wins_over_dot():
    bl = resolve_bridge_layers(
        {"bridge_layers": [7, 15, 23, 29], "dot_num_layers": 4},
        num_layers=30,
    )
    assert bl == (7, 15, 23, 29)


def test_dot_config_builds_cross_attn_action_depth():
    from openwam.model.architectures.dual_system.joint_cross_attn import DualSystemCrossAttnArchitecture
    from tests.test_openwam_trainer import _TINY_ARCH_CFG, _MockVideoBackbone

    cfg = dict(_TINY_ARCH_CFG)
    cfg["variant"] = "joint_cross_attn"
    cfg["bridge_layers"] = None
    cfg["dot_num_layers"] = 2
    cfg["num_dit_layers"] = 4
    arch = DualSystemCrossAttnArchitecture(cfg=cfg)
    arch.video_backbone = _MockVideoBackbone(dim=64, num_layers=4)
    # Re-resolve is not automatic after mock swap; check backbone built from cfg.
    assert arch.action_backbone.num_layers == 2
    assert arch.action_backbone.bridge_layers == (0, 3)
