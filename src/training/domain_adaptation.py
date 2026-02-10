"""Covariate-shift weighting utilities for domain adaptation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_covariate_shift_weights(
    train_X: np.ndarray,
    reference_X: np.ndarray,
    max_weight: float = 10.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Estimate importance weights w(x) = p_ref(x) / p_train(x).

    Uses a logistic domain classifier to separate train vs reference samples:
      p(domain=1|x) -> reference probability
      weight = p_ref / (1 - p_ref)
    """
    if train_X.ndim != 2 or reference_X.ndim != 2:
        raise ValueError("train_X and reference_X must be 2D")
    if train_X.shape[1] != reference_X.shape[1]:
        raise ValueError(
            f"Feature count mismatch: {train_X.shape[1]} vs {reference_X.shape[1]}"
        )

    n_train = len(train_X)
    n_ref = len(reference_X)
    X = np.vstack([train_X, reference_X]).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(n_train, dtype=np.int32),  # source/train
            np.ones(n_ref, dtype=np.int32),     # target/reference
        ]
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(Xs, y)

    p_ref = clf.predict_proba(Xs[:n_train])[:, 1]
    p_ref = np.clip(p_ref, 1e-4, 1 - 1e-4)
    weights = p_ref / (1.0 - p_ref)
    weights = np.clip(weights, 1.0 / max_weight, max_weight).astype(np.float32)
    if np.mean(weights) > 0:
        weights = weights / np.mean(weights)
    return weights
