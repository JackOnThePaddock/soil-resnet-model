"""Tests for dataframe feature resolution in SoilEnsemble inference."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.models.ensemble import SoilEnsemble


class _DummyModel:
    """Minimal model stub returning first two feature columns."""

    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[:, 0], x[:, 1]], dim=1)


def _build_stub_ensemble() -> SoilEnsemble:
    ensemble = SoilEnsemble.__new__(SoilEnsemble)
    ensemble.device = torch.device("cpu")
    ensemble.models = [_DummyModel()]
    ensemble.target_names = ["t0", "t1"]
    ensemble.config = {"feature_cols": ["band_0", "band_1", "band_2"]}
    ensemble.target_transform_types = {"t0": "identity", "t1": "identity"}
    ensemble.specialists = {}
    ensemble.specialist_blend_weight = 0.0

    scaler = StandardScaler()
    scaler.fit(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32))
    ensemble.scaler = scaler
    ensemble.n_features = int(scaler.n_features_in_)
    return ensemble


def test_predict_df_uses_config_feature_order():
    ensemble = _build_stub_ensemble()
    df = pd.DataFrame(
        {
            "band_2": [30.0],
            "band_0": [10.0],
            "band_1": [20.0],
        }
    )

    out = ensemble.predict_df(df)

    expected_scaled = ensemble.scaler.transform(np.array([[10.0, 20.0, 30.0]], dtype=np.float32))
    assert out["t0_pred"].iloc[0] == pytest.approx(expected_scaled[0, 0])
    assert out["t1_pred"].iloc[0] == pytest.approx(expected_scaled[0, 1])
    assert out["t0_std"].iloc[0] == pytest.approx(0.0)
    assert out["t1_std"].iloc[0] == pytest.approx(0.0)


def test_predict_df_raises_when_required_columns_missing():
    ensemble = _build_stub_ensemble()
    df = pd.DataFrame({"band_0": [10.0], "band_1": [20.0]})

    with pytest.raises(ValueError, match="Could not infer feature columns"):
        ensemble.predict_df(df)


def test_predict_raises_on_wrong_feature_dimension():
    ensemble = _build_stub_ensemble()

    with pytest.raises(ValueError, match="Expected 3 features"):
        ensemble.predict(np.array([[1.0, 2.0]], dtype=np.float32))
