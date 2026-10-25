"""Smoke test using synthetic tensors and untrained model parameters."""
import torch
from wss_mil import build_model

torch.manual_seed(7)
model = build_model().eval()
synthetic_windows = torch.randn(5, 1, 96)

with torch.no_grad():
    output = model(synthetic_windows)
    window_scores = output["bag_pred"]
    group_score = window_scores.mean()

print("Window score tensor shape:", tuple(window_scores.shape))
print("Group score tensor shape:", tuple(group_score.shape))
print("Untrained outputs; not predictions from a fitted model.")
