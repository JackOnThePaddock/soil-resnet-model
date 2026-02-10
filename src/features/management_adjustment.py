"""Management covariates to account for post-imagery lime/gypsum interventions."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _haversine_m(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    r = 6371000.0
    lon1r, lat1r, lon2r, lat2r = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _infer_application_type(value: str) -> str:
    """Map free-text material/type labels to canonical classes."""
    v = str(value).strip().lower()
    if "gypsum" in v:
        return "gypsum"
    if "lime" in v or "liming" in v:
        return "lime"
    return "other"


def add_management_adjustment_features(
    soil_df: pd.DataFrame,
    applications_df: pd.DataFrame,
    obs_date_col: str = "date",
    image_date: str = "2020-09-30",
    lon_col: str = "lon",
    lat_col: str = "lat",
    max_distance_m: float = 150.0,
    half_life_days: float = 365.0,
) -> pd.DataFrame:
    """
    Add management covariates capturing post-imagery lime/gypsum impacts.

    Expected application columns:
        - `date` (required)
        - `rate_t_ha` (required)
        - one of `type` or `material` (optional but recommended)
        - optional location keys: `site_id` or (`lat`,`lon`)
    """
    if applications_df.empty:
        out = soil_df.copy()
        for col in (
            "mgmt_lime_rate_post_be",
            "mgmt_gypsum_rate_post_be",
            "mgmt_lime_rate_decay",
            "mgmt_gypsum_rate_decay",
            "mgmt_days_since_lime",
            "mgmt_days_since_gypsum",
            "mgmt_any_post_be",
        ):
            out[col] = 0.0
        return out

    app = applications_df.copy()
    app.columns = app.columns.str.lower()
    obs = soil_df.copy()
    obs.columns = obs.columns.str.lower()

    obs_date_col = obs_date_col.lower()
    lon_col = lon_col.lower()
    lat_col = lat_col.lower()

    if obs_date_col not in obs.columns:
        raise ValueError(f"Observation date column '{obs_date_col}' not found in soil dataframe")
    if "date" not in app.columns:
        raise ValueError("applications_df must include a 'date' column")
    if "rate_t_ha" not in app.columns:
        raise ValueError("applications_df must include a 'rate_t_ha' column")

    obs_dates = _to_datetime_safe(obs[obs_date_col])
    app_dates = _to_datetime_safe(app["date"])
    image_dt = pd.Timestamp(image_date, tz="UTC")

    app["date"] = app_dates
    app["rate_t_ha"] = pd.to_numeric(app["rate_t_ha"], errors="coerce").fillna(0.0)
    type_col = "type" if "type" in app.columns else ("material" if "material" in app.columns else None)
    app["mgmt_type"] = (
        app[type_col].apply(_infer_application_type) if type_col else "other"
    )

    # Initialize output columns.
    out = obs.copy()
    out["mgmt_lime_rate_post_be"] = 0.0
    out["mgmt_gypsum_rate_post_be"] = 0.0
    out["mgmt_lime_rate_decay"] = 0.0
    out["mgmt_gypsum_rate_decay"] = 0.0
    out["mgmt_days_since_lime"] = np.nan
    out["mgmt_days_since_gypsum"] = np.nan
    out["mgmt_any_post_be"] = 0.0

    use_site_key = "site_id" in out.columns and "site_id" in app.columns
    use_spatial = lon_col in out.columns and lat_col in out.columns and {"lon", "lat"}.issubset(app.columns)

    decay_lambda = np.log(2.0) / max(half_life_days, 1.0)

    for idx, row in out.iterrows():
        obs_dt = obs_dates.iloc[idx]
        if pd.isna(obs_dt):
            continue

        candidates = app[(app["date"] >= image_dt) & (app["date"] <= obs_dt)]
        if candidates.empty:
            continue

        if use_site_key:
            candidates = candidates[candidates["site_id"].astype(str) == str(row["site_id"])]
        elif use_spatial:
            dists = _haversine_m(
                np.full(len(candidates), float(row[lon_col])),
                np.full(len(candidates), float(row[lat_col])),
                candidates["lon"].astype(float).values,
                candidates["lat"].astype(float).values,
            )
            candidates = candidates.loc[dists <= max_distance_m].copy()
            if not candidates.empty:
                candidates["distance_m"] = dists[dists <= max_distance_m]
        else:
            # No reliable key available: skip to avoid noisy attribution.
            continue

        if candidates.empty:
            continue

        out.at[idx, "mgmt_any_post_be"] = 1.0
        delta_days = (obs_dt - candidates["date"]).dt.total_seconds() / 86400.0
        decay = np.exp(-decay_lambda * np.clip(delta_days.values, a_min=0.0, a_max=None))
        rate = candidates["rate_t_ha"].values.astype(float)

        for mgmt_type, rate_col, decay_col, since_col in (
            ("lime", "mgmt_lime_rate_post_be", "mgmt_lime_rate_decay", "mgmt_days_since_lime"),
            ("gypsum", "mgmt_gypsum_rate_post_be", "mgmt_gypsum_rate_decay", "mgmt_days_since_gypsum"),
        ):
            mask = candidates["mgmt_type"].values == mgmt_type
            if not np.any(mask):
                continue
            rate_sum = float(np.sum(rate[mask]))
            rate_decay = float(np.sum(rate[mask] * decay[mask]))
            out.at[idx, rate_col] = rate_sum
            out.at[idx, decay_col] = rate_decay
            out.at[idx, since_col] = float(np.min(delta_days.values[mask]))

    out["mgmt_days_since_lime"] = out["mgmt_days_since_lime"].fillna(-1.0)
    out["mgmt_days_since_gypsum"] = out["mgmt_days_since_gypsum"].fillna(-1.0)
    return out
