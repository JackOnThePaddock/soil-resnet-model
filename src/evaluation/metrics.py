"""Evaluation metrics for soil property prediction models."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute R², RMSE, and MAE for each target property.

    Args:
        y_true: Ground truth array of shape (n_samples, n_targets)
        y_pred: Predictions array of shape (n_samples, n_targets)
        target_names: Names for each target column

    Returns:
        Dict mapping target name to {r2, rmse, mae}
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]
    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    metrics = {}
    for i, name in enumerate(target_names):
        true = y_true[:, i]
        pred = y_pred[:, i]

        # Skip columns that are all NaN
        mask = ~np.isnan(true)
        if mask.sum() < 2:
            continue

        true = true[mask]
        pred = pred[mask]

        metrics[name] = {
            "r2": float(r2_score(true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(true, pred))),
            "mae": float(mean_absolute_error(true, pred)),
        }

    return metrics


def format_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Model Metrics",
) -> str:
    """Format metrics dict as a printable table."""
    lines = [title, "=" * len(title)]
    lines.append(f"{'Target':<10} {'R²':>8} {'RMSE':>10} {'MAE':>10}")
    lines.append("-" * 42)

    for target, m in metrics.items():
        lines.append(
            f"{target:<10} {m['r2']:8.3f} {m['rmse']:10.3f} {m['mae']:10.3f}"
        )

    return "\n".join(lines)
