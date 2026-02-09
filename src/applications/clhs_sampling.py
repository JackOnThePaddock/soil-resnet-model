"""Conditioned Latin Hypercube Sampling for optimal soil sampling locations."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def clhs_sample(
    X: np.ndarray,
    coords: np.ndarray,
    n_samples: int = 50,
    pca_components: int = 6,
    max_iter: int = 20000,
    n_restarts: int = 3,
    corr_weight: float = 0.5,
    seed: int = 42,
    paddock_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Run conditioned Latin Hypercube Sampling on spectral features.

    Args:
        X: Feature array (n_pixels, n_bands)
        coords: Coordinate array (n_pixels, 2) as (lon, lat)
        n_samples: Number of samples to select
        pca_components: Number of PCA components for stratification
        max_iter: Simulated annealing iterations
        n_restarts: Number of random restarts
        corr_weight: Weight for correlation preservation objective
        seed: Random seed
        paddock_labels: Optional paddock name per pixel

    Returns:
        DataFrame with selected sample locations
    """
    rng = np.random.default_rng(seed)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=min(pca_components, X.shape[1]), random_state=seed)
    X_pca = pca.fit_transform(X_std)

    n, P = X_pca.shape
    rank_norm = np.zeros_like(X_pca, dtype=np.float32)
    for j in range(P):
        rank_norm[:, j] = (rankdata(X_pca[:, j], method="average") - 0.5) / n

    z = (X_pca - X_pca.mean(axis=0)) / X_pca.std(axis=0, ddof=0)
    corr_pop = np.corrcoef(z, rowvar=False) if P > 1 else np.array([[1.0]])
    expected = (np.arange(n_samples) + 0.5) / n_samples

    def cost(idx):
        total = 0.0
        for j in range(P):
            r = np.sort(rank_norm[idx, j])
            total += np.sum((r - expected) ** 2)
        if corr_weight > 0 and P > 1:
            corr_s = np.corrcoef(z[idx], rowvar=False)
            total += corr_weight * np.sum((corr_s - corr_pop) ** 2)
        return float(total)

    best_idx, best_cost = None, None
    for restart in range(n_restarts):
        local_rng = np.random.default_rng(seed + 1000 * restart)
        sample_idx = local_rng.choice(n, size=n_samples, replace=False).tolist()
        selected = set(sample_idx)
        unselected = [i for i in range(n) if i not in selected]
        curr_cost = cost(np.array(sample_idx))
        run_best_idx, run_best_cost = sample_idx.copy(), curr_cost

        for it in range(max_iter):
            T = 1.0 * (1e-3) ** (it / max_iter)
            out_pos = local_rng.integers(0, n_samples)
            in_pos = local_rng.integers(0, len(unselected))
            new_sample = sample_idx.copy()
            new_sample[out_pos] = unselected[in_pos]
            new_cost = cost(np.array(new_sample))
            delta = new_cost - curr_cost
            if delta < 0 or local_rng.random() < np.exp(-delta / max(T, 1e-10)):
                old_idx = sample_idx[out_pos]
                sample_idx[out_pos] = unselected[in_pos]
                unselected[in_pos] = old_idx
                curr_cost = new_cost
                if curr_cost < run_best_cost:
                    run_best_cost = curr_cost
                    run_best_idx = sample_idx.copy()

        if best_cost is None or run_best_cost < best_cost:
            best_cost = run_best_cost
            best_idx = run_best_idx

    best_idx = np.array(best_idx, dtype=int)
    result = pd.DataFrame({
        "id": np.arange(1, len(best_idx) + 1),
        "lon": coords[best_idx, 0],
        "lat": coords[best_idx, 1],
    })
    if paddock_labels is not None:
        result["paddock"] = paddock_labels[best_idx]

    return result
