"""Tests for management adjustment feature engineering."""

import pandas as pd

from src.features.management_adjustment import add_management_adjustment_features


def test_management_features_added_from_lime_and_gypsum():
    soil = pd.DataFrame(
        {
            "site_id": ["A"],
            "lat": [-35.25],
            "lon": [149.10],
            "date": ["2021-06-01"],
        }
    )
    apps = pd.DataFrame(
        {
            "site_id": ["A", "A", "A"],
            "date": ["2019-10-01", "2021-01-15", "2021-03-15"],
            "type": ["lime", "gypsum", "lime"],
            "rate_t_ha": [2.0, 1.5, 1.0],
        }
    )

    out = add_management_adjustment_features(
        soil_df=soil,
        applications_df=apps,
        obs_date_col="date",
        image_date="2020-09-30",
    )

    # 2019 lime is pre-image and ignored.
    assert out["mgmt_lime_rate_post_be"].iloc[0] == 1.0
    assert out["mgmt_gypsum_rate_post_be"].iloc[0] == 1.5
    assert out["mgmt_any_post_be"].iloc[0] == 1.0
    assert out["mgmt_lime_rate_decay"].iloc[0] > 0.0
    assert out["mgmt_gypsum_rate_decay"].iloc[0] > 0.0
    assert out["mgmt_days_since_lime"].iloc[0] >= 0.0
    assert out["mgmt_days_since_gypsum"].iloc[0] >= 0.0


def test_management_features_default_when_no_applications():
    soil = pd.DataFrame(
        {
            "lat": [-35.25],
            "lon": [149.10],
            "date": ["2021-06-01"],
        }
    )
    out = add_management_adjustment_features(soil_df=soil, applications_df=pd.DataFrame())
    assert out["mgmt_any_post_be"].iloc[0] == 0.0
    assert out["mgmt_lime_rate_post_be"].iloc[0] == 0.0
    assert out["mgmt_gypsum_rate_post_be"].iloc[0] == 0.0
