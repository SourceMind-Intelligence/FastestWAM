"""A zero-input cross-attention policy needs positions beyond Q/K-only RoPE."""
import pytest
import torch
from openwam.model.action_backbone.separate_action_dit import ActionDiT


def make_model(layers=4, scale=0.0):
    return ActionDiT(action_dim=10, dim=32, ffn_dim=64, num_heads=4,
                     num_layers=layers, video_dim=32,
                     bridge_layers=tuple(range(layers)), text_dim=32,
                     action_position_scale=scale).eval()


@pytest.mark.parametrize('layers', [1, 4, 8])
def test_zero_input_and_deterministic_refinement_preserve_constant_sequence(layers):
    torch.manual_seed(42)
    model = make_model(layers)
    bridges = {i: torch.randn(2, 20, 32) for i in range(layers)}
    context = torch.randn(2, 5, 32)
    with torch.no_grad():
        coarse = model(torch.zeros(2, 32, 10), bridges, torch.full((2,),1000.), context=context)
        refined = model(0.9*coarse, bridges, torch.full((2,),100.), context=context)
    for x in (coarse, refined):
        torch.testing.assert_close(x, x[:, :1].expand_as(x), atol=2e-6, rtol=2e-6)


def test_opt_in_position_breaks_symmetry_and_preserves_checkpoint_keys():
    torch.manual_seed(42)
    original = make_model()
    repaired = make_model(scale=0.1)
    repaired.load_state_dict(original.state_dict(), strict=True)
    bridges = {i: torch.randn(1, 20, 32) for i in range(4)}
    zeros = torch.zeros(1, 32, 10)
    with torch.no_grad():
        y = repaired(zeros, bridges, torch.tensor([1000.]))
    assert (y-y[:, :1]).abs().max().item() > 1e-3
    default = make_model()
    default.load_state_dict(original.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(default(zeros, bridges, torch.tensor([1000.])),
                                   original(zeros, bridges, torch.tensor([1000.])), rtol=0, atol=0)


@pytest.mark.parametrize('scale', [-1, float('nan'), float('inf')])
def test_invalid_position_scale_rejected(scale):
    with pytest.raises(ValueError, match='action_position_scale'):
        make_model(scale=scale)
