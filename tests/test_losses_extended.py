"""Tests for robust training losses and ESP consistency penalty."""

import numpy as np
import torch

from src.training.losses import esp_consistency_penalty, masked_weighted_huber_loss


def test_masked_weighted_huber_handles_nan_and_weights():
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([[1.5, float("nan")], [2.0, 5.0]], dtype=torch.float32)
    target_weights = torch.tensor([2.0, 0.5], dtype=torch.float32)
    sample_weights = torch.tensor([1.0, 2.0], dtype=torch.float32)

    loss = masked_weighted_huber_loss(
        predictions=preds,
        targets=targets,
        delta=1.0,
        target_weights=target_weights,
        sample_weights=sample_weights,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_esp_consistency_penalty_zero_when_consistent_identity():
    # targets: [esp, na, cec]
    na = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    cec = torch.tensor([5.0, 10.0, 20.0], dtype=torch.float32)
    esp = 100.0 * na / cec
    preds = torch.stack([esp, na, cec], dim=1)

    penalty = esp_consistency_penalty(
        predictions=preds,
        target_names=["esp", "na", "cec"],
        transform_types={"esp": "identity", "na": "identity", "cec": "identity"},
    )
    assert float(penalty.item()) < 1e-6


def test_esp_consistency_penalty_zero_when_consistent_log1p():
    na_raw = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cec_raw = np.array([5.0, 10.0, 20.0], dtype=np.float32)
    esp_raw = 100.0 * na_raw / cec_raw
    preds = torch.tensor(
        np.stack(
            [
                np.log1p(esp_raw),
                np.log1p(na_raw),
                np.log1p(cec_raw),
            ],
            axis=1,
        ),
        dtype=torch.float32,
    )

    penalty = esp_consistency_penalty(
        predictions=preds,
        target_names=["esp", "na", "cec"],
        transform_types={"esp": "log1p", "na": "log1p", "cec": "log1p"},
    )
    assert float(penalty.item()) < 1e-5
