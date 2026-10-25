# WSS-MIL

**Weakly Supervised Self-attention Multiple Instance Learning for household-level EV identification**

A compact, model-only PyTorch reference implementation accompanying the WSS-MIL research paper.

## Architecture

The model processes fixed-length sequences through:

1. Convolutional temporal embedding with learned positional embeddings.
2. Hierarchical binary-tree max pooling.
3. Level-wise **full multi-head self-attention** (no graph attention or adjacency matrix).
4. Exponential-moving-average vector quantization.
5. Global max pooling across tree levels and a sigmoid classification head.

Predictions for multiple windows can be averaged into a single sequence-group score.

## Installation

```bash
python -m pip install -e .
```

Requires Python 3.10+ and PyTorch 2.x.

## Model-only smoke test

```bash
python examples/synthetic_demo.py
```

This demonstration uses randomly generated tensors. Outputs are from randomly initialized parameters and **have no predictive meaning**.

## Basic API

```python
import torch
from wss_mil import build_model

model = build_model().eval()
windows = torch.randn(3, 1, 96)  # synthetic windows (batch, channel, steps)

with torch.no_grad():
    outputs = model(windows)
    window_scores = outputs["bag_pred"]
    group_score = window_scores.mean()
```

The encoder classes are `SelfAttentionBlock` and `LevelWiseSelfAttentionEncoder`. The model attribute is `self_attention`; these names describe the actual operation, standard multi-head self-attention applied separately to nodes at each level.\n\nThe reference configuration is constructed by `build_model()`; additional ablation switches remain available via `ComponentAblationTreeMIL`.

### Outputs

`model(x)` returns `bag_pred`, `bag_logit`, `entropy_loss`, and `commitment_loss`.
Its input is a float tensor of shape `(batch, 1, 96)`. Each window can be normalized individually, with zero-variance windows handled by substituting a standard deviation of one.

### What this repository provides

Architecture definitions and synthetic-input demonstrations only. No trained parameters are distributed; running the demo **does not reproduce the reported research metrics**.

## Citation

If you use this implementation, please cite the accompanying WSS-MIL paper once its bibliographic details are final. Update `CITATION.cff` after publication.

## License

No license is assigned in this draft. Select a license only after verifying that you have authority to distribute the work and that any adapted third-party implementations are properly credited and licensed.

## Naming conventions

`self_attention` refers to the level-wise temporal encoder and is distinct from
`attention_pooling`. The publication configuration uses `PlainMaxPooling`,
not attention-based pooling. Earlier research code called the encoder
`graph_attention`; the public implementation corrects that name without
changing the underlying attention operation.

If transferring parameters from a separately maintained implementation,
keys containing `graph_attention.` must be translated to `self_attention.`.
This repository does not ship parameter files or a checkpoint-loading script.
