"""Synthetic-only architectural and shape tests."""
import torch
from wss_mil import build_model
from wss_mil.architecture import PlainMaxPooling


def test_architecture_forward():
    model = build_model().eval()
    assert isinstance(model.attention_pooling, PlainMaxPooling)
    assert model.num_levels == 8
    with torch.no_grad():
        result = model(torch.randn(2, 1, 96))
    assert result["bag_pred"].shape == (2,)
    assert torch.isfinite(result["bag_pred"]).all()
    assert ((result["bag_pred"] >= 0) & (result["bag_pred"] <= 1)).all()


def test_all_ones_forward():
    model = build_model().eval()
    with torch.no_grad():
        result = model(torch.zeros(1, 1, 96))
    assert torch.isfinite(result["bag_pred"]).all()


def test_attention_is_named_for_actual_operation():
    from wss_mil.architecture import SelfAttentionBlock, LevelWiseSelfAttentionEncoder
    model = build_model()
    assert isinstance(model.self_attention, LevelWiseSelfAttentionEncoder)
    assert all(isinstance(block, SelfAttentionBlock) for block in model.self_attention.blocks)
    assert not hasattr(model, "graph_attention")
    assert any(key.startswith("self_attention.") for key in model.state_dict())
