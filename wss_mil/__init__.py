"""WSS-MIL research implementation."""
from .architecture import ComponentAblationTreeMIL

def build_model():
    """Instantiate the architecture described in the associated paper."""
    return ComponentAblationTreeMIL(
        use_pos_embedding=True,
        use_tree=True,
        use_attention=True,
        use_vq=True,
        use_ap=False,
        use_entropy_loss=True,
        use_commit_loss=True,
        pooling_mode="max",
    )

__all__ = ["ComponentAblationTreeMIL", "build_model"]
