import torch.nn.functional as F


def compute_weighted_bce_loss(logits, targets, pos_weight):
    """Weighted Binary Cross Entropy loss via functional API.

    Uses F.binary_cross_entropy_with_logits directly. Eliminates Python object
    allocation and GC pressure in the training hot loop.
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight
    )
