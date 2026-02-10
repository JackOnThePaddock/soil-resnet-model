"""Lightweight Spectral-GPT style embedding extraction for Sentinel-2 spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


@dataclass
class SpectralGPTConfig:
    """Configuration for spectral embedding training."""

    embedding_dim: int = 16
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mask_ratio: float = 0.3
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: Optional[str] = None


class _SpectralGPTModel(nn.Module):
    """Transformer encoder over band tokens with masked-band reconstruction."""

    def __init__(
        self,
        n_bands: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.value_proj = nn.Linear(1, hidden_dim)
        self.band_embedding = nn.Parameter(torch.randn(n_bands, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.recon_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, n_bands] normalized spectra
            mask: [B, n_bands] bool mask for reconstruction loss

        Returns:
            embedding: [B, embedding_dim]
            reconstruction: [B, n_bands]
        """
        tokens = self.value_proj(x.unsqueeze(-1)) + self.band_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        embedding = self.embed_head(pooled)
        reconstruction = self.recon_head(encoded).squeeze(-1)
        return embedding, reconstruction


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_bands(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize spectral bands with safe std fallback."""
    valid = np.isfinite(X)
    count = np.sum(valid, axis=0).astype(np.float32)
    x_safe = np.where(valid, X, 0.0)

    mu = np.divide(
        np.sum(x_safe, axis=0),
        count,
        out=np.zeros(X.shape[1], dtype=np.float32),
        where=count > 0,
    )
    centered = np.where(valid, X - mu, 0.0)
    var = np.divide(
        np.sum(centered**2, axis=0),
        count,
        out=np.ones(X.shape[1], dtype=np.float32),
        where=count > 0,
    )
    sigma = np.sqrt(var).astype(np.float32)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xn = (X - mu) / sigma
    Xn = np.where(np.isfinite(Xn), Xn, 0.0)
    return Xn.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def train_spectral_gpt_embeddings(
    X_bands: np.ndarray,
    config: SpectralGPTConfig = SpectralGPTConfig(),
) -> np.ndarray:
    """
    Train a lightweight Spectral-GPT style model and return embeddings.

    X_bands shape: [n_samples, n_bands]
    Returns: [n_samples, embedding_dim]
    """
    if X_bands.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X_bands.shape}")
    if X_bands.shape[1] < 3:
        raise ValueError("Need at least 3 spectral bands")

    _set_seed(config.seed)
    X_norm, _, _ = _normalize_bands(X_bands.astype(np.float32))

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _SpectralGPTModel(
        n_bands=X_norm.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    X_tensor = torch.from_numpy(X_norm)
    n = len(X_tensor)
    for _ in range(config.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch = X_tensor[idx].to(device)

            # Random masked-band reconstruction objective.
            mask = torch.rand_like(batch) < config.mask_ratio
            masked_input = batch.clone()
            masked_input[mask] = 0.0

            _, recon = model(masked_input, mask=mask)
            if mask.sum() == 0:
                loss = torch.mean((recon - batch) ** 2)
            else:
                loss = torch.mean((recon[mask] - batch[mask]) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Final embeddings for all samples.
    model.eval()
    embeds = np.zeros((n, config.embedding_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, config.batch_size):
            end = min(start + config.batch_size, n)
            batch = X_tensor[start:end].to(device)
            emb, _ = model(batch)
            embeds[start:end] = emb.cpu().numpy()
    return embeds


def pca_spectral_embeddings(
    X_bands: np.ndarray,
    n_components: int = 16,
    random_state: int = 42,
) -> np.ndarray:
    """Fallback spectral embedding extraction via PCA."""
    if X_bands.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X_bands.shape}")
    n_components = int(min(n_components, X_bands.shape[1], max(1, X_bands.shape[0] - 1)))
    X_norm, _, _ = _normalize_bands(X_bands.astype(np.float32))
    if float(np.nanvar(X_norm)) < 1e-10:
        return np.zeros((X_norm.shape[0], n_components), dtype=np.float32)
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X_norm).astype(np.float32)
