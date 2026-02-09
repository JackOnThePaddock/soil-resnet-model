"""Shared test fixtures for soil-resnet-model tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_features():
    """Generate random 64-band feature array (10 samples)."""
    np.random.seed(42)
    return np.random.randn(10, 64).astype(np.float32)


@pytest.fixture
def sample_targets():
    """Generate random 7-target array (10 samples)."""
    np.random.seed(42)
    return np.random.randn(10, 7).astype(np.float32)


@pytest.fixture
def device():
    """Get available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
