"""Tests for SpectralGPT embedding utilities."""

import numpy as np

from src.features.spectral_gpt import SpectralGPTConfig, pca_spectral_embeddings, train_spectral_gpt_embeddings


def test_pca_spectral_embeddings_shape():
    X = np.random.RandomState(42).randn(50, 10).astype(np.float32)
    emb = pca_spectral_embeddings(X, n_components=6)
    assert emb.shape == (50, 6)


def test_train_spectral_gpt_embeddings_shape_small():
    X = np.random.RandomState(0).randn(32, 10).astype(np.float32)
    cfg = SpectralGPTConfig(
        embedding_dim=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        seed=123,
        device="cpu",
    )
    emb = train_spectral_gpt_embeddings(X, config=cfg)
    assert emb.shape == (32, 8)
    assert np.isfinite(emb).all()
