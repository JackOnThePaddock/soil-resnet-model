"""Feature fusion utilities for AlphaEarth + Bare Earth + management covariates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.bare_earth import sample_bare_earth_at_points
from src.features.management_adjustment import add_management_adjustment_features
from src.features.spectral_gpt import SpectralGPTConfig, pca_spectral_embeddings, train_spectral_gpt_embeddings


def _feature_sort_key(name: str) -> Tuple[int, int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, -1, name)


def detect_alphaearth_feature_columns(df: pd.DataFrame) -> List[str]:
    """Detect AlphaEarth feature columns in deterministic numeric order."""
    cols = df.columns.tolist()
    for pattern in (r"^band_\d+$", r"^a\d+$", r"^ae_\d+$"):
        rx = re.compile(pattern)
        matched = [c for c in cols if rx.match(c.lower())]
        if matched:
            return sorted(matched, key=_feature_sort_key)
    return []


def _compute_spectral_embeddings(
    be_bands: np.ndarray,
    method: str = "spectral_gpt",
    embedding_dim: int = 16,
    random_seed: int = 42,
) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        return pca_spectral_embeddings(
            X_bands=be_bands,
            n_components=embedding_dim,
            random_state=random_seed,
        )
    if method == "spectral_gpt":
        cfg = SpectralGPTConfig(
            embedding_dim=embedding_dim,
            seed=random_seed,
        )
        return train_spectral_gpt_embeddings(be_bands, config=cfg)
    raise ValueError(f"Unsupported spectral embedding method: {method}")


def build_fused_feature_table(
    soil_df: pd.DataFrame,
    bare_earth_raster: Path,
    lon_col: str = "lon",
    lat_col: str = "lat",
    spectral_method: str = "spectral_gpt",
    spectral_dim: int = 16,
    applications_df: Optional[pd.DataFrame] = None,
    obs_date_col: str = "date",
    image_date: str = "2020-09-30",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Build fused feature table:
    - AlphaEarth embeddings (existing)
    - Bare Earth Sentinel-2 bands sampled from raster
    - Spectral embedding features derived from bare-earth bands
    - Optional management adjustment covariates
    """
    df = soil_df.copy()
    df.columns = df.columns.str.lower()
    lon_col = lon_col.lower()
    lat_col = lat_col.lower()
    obs_date_col = obs_date_col.lower()

    alpha_cols = detect_alphaearth_feature_columns(df)
    if not alpha_cols:
        raise ValueError("No AlphaEarth features found (expected band_/Axx/AE_ columns)")

    be_df = sample_bare_earth_at_points(
        raster_path=bare_earth_raster,
        points_df=df,
        lon_col=lon_col,
        lat_col=lat_col,
        output_prefix="be_",
    )
    be_cols = be_df.columns.tolist()
    be_values = be_df.values.astype(np.float32)
    spectral_embed = _compute_spectral_embeddings(
        be_bands=be_values,
        method=spectral_method,
        embedding_dim=spectral_dim,
        random_seed=random_seed,
    )
    spectral_cols = [f"sgpt_{i:02d}" for i in range(spectral_embed.shape[1])]
    spectral_df = pd.DataFrame(spectral_embed, columns=spectral_cols, index=df.index)

    out = pd.concat([df, be_df, spectral_df], axis=1)

    management_cols: List[str] = []
    if applications_df is not None and not applications_df.empty:
        out = add_management_adjustment_features(
            soil_df=out,
            applications_df=applications_df,
            obs_date_col=obs_date_col,
            image_date=image_date,
            lon_col=lon_col,
            lat_col=lat_col,
        )
        management_cols = [
            c for c in out.columns if c.startswith("mgmt_")
        ]

    fused_feature_cols = alpha_cols + be_cols + spectral_cols + management_cols
    # Canonical sequential feature namespace for training scripts.
    for i, src_col in enumerate(fused_feature_cols):
        out[f"feat_{i:03d}"] = out[src_col]

    metadata = {
        "alphaearth_cols": alpha_cols,
        "bareearth_cols": be_cols,
        "spectral_cols": spectral_cols,
        "management_cols": management_cols,
        "fused_feature_cols": [f"feat_{i:03d}" for i in range(len(fused_feature_cols))],
    }
    return out, metadata
