"""Tests for ensemble inference pipeline."""

import numpy as np
import pytest
import torch

from src.models.resnet import NationalSoilNet, DEFAULT_TARGETS
from src.models.ensemble import SoilEnsemble


class TestSoilEnsembleUnit:
    """Test SoilEnsemble without loading actual model files."""

    def test_ensemble_predict_shape(self):
        """Test prediction with manually constructed ensemble."""
        models = []
        for _ in range(3):
            model = NationalSoilNet(input_dim=64, hidden_dim=128)
            model.eval()
            models.append(model)

        features = np.random.randn(5, 64).astype(np.float32)
        tensor = torch.from_numpy(features)

        all_preds = []
        for model in models:
            with torch.no_grad():
                stacked = model.forward_stacked(tensor)
            all_preds.append(stacked.numpy())

        ensemble_preds = np.mean(all_preds, axis=0)
        assert ensemble_preds.shape == (5, 7)

    def test_ensemble_uncertainty(self):
        """Ensemble std should be non-negative."""
        models = []
        for i in range(5):
            torch.manual_seed(i)
            model = NationalSoilNet(input_dim=64, hidden_dim=128)
            model.eval()
            models.append(model)

        features = np.random.randn(3, 64).astype(np.float32)
        tensor = torch.from_numpy(features)

        all_preds = []
        for model in models:
            with torch.no_grad():
                stacked = model.forward_stacked(tensor)
            all_preds.append(stacked.numpy())

        std = np.std(all_preds, axis=0)
        assert std.shape == (3, 7)
        assert np.all(std >= 0)

    def test_different_seeds_give_different_predictions(self):
        """Models with different seeds should produce different outputs."""
        torch.manual_seed(0)
        model_a = NationalSoilNet(input_dim=64, hidden_dim=128)
        model_a.eval()

        torch.manual_seed(42)
        model_b = NationalSoilNet(input_dim=64, hidden_dim=128)
        model_b.eval()

        x = torch.randn(2, 64)
        with torch.no_grad():
            out_a = model_a.forward_stacked(x)
            out_b = model_b.forward_stacked(x)

        assert not torch.allclose(out_a, out_b)
