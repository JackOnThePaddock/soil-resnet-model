"""Loss functions for soil property prediction training."""

from typing import Optional

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


def masked_weighted_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
    target_weights: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weighted Huber loss with NaN-masking for multi-target regression.

    Args:
        predictions: [batch, targets]
        targets: [batch, targets] with optional NaNs
        delta: Huber transition threshold
        target_weights: Optional [targets] tensor
        sample_weights: Optional [batch] tensor
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )

    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)

    # Compute loss only on observed targets to avoid NaN gradients from masked entries.
    pred_valid = predictions[mask]
    target_valid = targets[mask]

    error = pred_valid - target_valid
    abs_error = torch.abs(error)
    huber = torch.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )

    weight_matrix = torch.ones_like(predictions)
    if target_weights is not None:
        target_weights = target_weights.to(predictions.device, dtype=predictions.dtype)
        weight_matrix = weight_matrix * target_weights.view(1, -1)
    if sample_weights is not None:
        sample_weights = sample_weights.to(predictions.device, dtype=predictions.dtype)
        weight_matrix = weight_matrix * sample_weights.view(-1, 1)

    valid_weights = weight_matrix[mask]
    if valid_weights.sum() <= 0:
        return torch.tensor(0.0, device=predictions.device)
    weighted_loss = huber * valid_weights
    return weighted_loss.sum() / valid_weights.sum()


def esp_consistency_penalty(
    predictions: torch.Tensor,
    target_names: list[str],
    targets: Optional[torch.Tensor] = None,
    transform_types: Optional[dict[str, str]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize inconsistency between direct ESP and derived ESP = 100 * Na / CEC.

    Supported differentiable transforms:
    - identity: y_raw = y_pred
    - log1p: y_raw = exp(y_pred) - 1
    """
    transform_types = transform_types or {}
    try:
        esp_idx = target_names.index("esp")
        na_idx = target_names.index("na")
        cec_idx = target_names.index("cec")
    except ValueError:
        return torch.tensor(0.0, device=predictions.device)

    esp_pred = predictions[:, esp_idx]
    na_pred = predictions[:, na_idx]
    cec_pred = predictions[:, cec_idx]

    def _to_raw(pred: torch.Tensor, transform_name: str) -> torch.Tensor:
        if transform_name == "identity":
            return pred
        if transform_name == "log1p":
            return torch.expm1(torch.clamp(pred, min=-20.0, max=20.0))
        # Unknown / non-differentiable transform -> no consistency penalty.
        return torch.full_like(pred, float("nan"))

    esp_raw = _to_raw(esp_pred, transform_types.get("esp", "identity"))
    na_raw = _to_raw(na_pred, transform_types.get("na", "identity"))
    cec_raw = _to_raw(cec_pred, transform_types.get("cec", "identity"))

    valid = torch.isfinite(esp_raw) & torch.isfinite(na_raw) & torch.isfinite(cec_raw)
    if targets is not None:
        # Apply consistency only where all three targets are actually observed.
        observed = (
            ~torch.isnan(targets[:, esp_idx])
            & ~torch.isnan(targets[:, na_idx])
            & ~torch.isnan(targets[:, cec_idx])
        )
        valid = valid & observed
    if valid.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)

    derived_esp = 100.0 * na_raw / torch.clamp(cec_raw, min=eps)
    return torch.nn.functional.smooth_l1_loss(esp_raw[valid], derived_esp[valid])
