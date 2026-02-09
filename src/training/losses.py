"""Loss functions for soil property prediction training."""

import torch


def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE loss ignoring NaN targets.

    Allows training on samples with partially-observed soil properties
    by masking out NaN values before computing the loss.

    Args:
        predictions: Model predictions tensor
        targets: Ground truth tensor (may contain NaN)

    Returns:
        Scalar MSE loss over valid (non-NaN) entries
    """
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    return torch.nn.functional.mse_loss(predictions[mask], targets[mask])
