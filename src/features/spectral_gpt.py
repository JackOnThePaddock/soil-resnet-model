"""Spectral embedding extraction backends for Sentinel-2 data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import subprocess
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import requests

OFFICIAL_SPECTRALGPT_REPO_URL = "https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT.git"
OFFICIAL_SPECTRALGPT_PLUS_URL = (
    "https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1"
)


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


@dataclass
class OfficialSpectralGPTConfig:
    """Configuration for official pretrained SpectralGPT embedding extraction."""

    checkpoint_path: Optional[str] = None
    checkpoint_url: str = OFFICIAL_SPECTRALGPT_PLUS_URL
    repo_dir: Optional[str] = None
    repo_url: str = OFFICIAL_SPECTRALGPT_REPO_URL
    model_name: str = "mae_vit_base_patch8_128"
    batch_size: int = 16
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


def _ensure_official_repo(repo_dir: Path, repo_url: str) -> Path:
    """Ensure official SpectralGPT code is available locally."""
    repo_dir = Path(repo_dir)
    if (repo_dir / "models_mae_spectral.py").exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", repo_url, str(repo_dir)],
        check=True,
    )
    if not (repo_dir / "models_mae_spectral.py").exists():
        raise FileNotFoundError(f"Official SpectralGPT repo clone incomplete: {repo_dir}")
    return repo_dir


def _download_checkpoint(checkpoint_url: str, checkpoint_path: Path, timeout: int = 180) -> Path:
    """Download official checkpoint if not already present."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 100_000_000:
        return checkpoint_path

    with requests.get(checkpoint_url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(checkpoint_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 8):
                if chunk:
                    f.write(chunk)
    return checkpoint_path


def _safe_nan_minmax_scale_per_band(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalize each (sample, band) plane to [0, 1] with NaN safety.

    Input shape: [N, C, H, W]
    """
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    x_safe = np.where(valid, x, np.nan)

    mn = np.nanmin(x_safe, axis=(2, 3), keepdims=True)
    mx = np.nanmax(x_safe, axis=(2, 3), keepdims=True)
    mn = np.where(np.isfinite(mn), mn, 0.0)
    mx = np.where(np.isfinite(mx), mx, 1.0)
    denom = np.maximum(mx - mn, 1e-6)

    out = (np.where(valid, x, mn) - mn) / denom
    out = np.clip(out, 0.0, 1.0)
    out = np.where(np.isfinite(out), out, 0.0)
    return out.astype(np.float32)


def _to_spectralgpt_s2_12(
    chips: np.ndarray,
    band_names: list[str],
) -> np.ndarray:
    """
    Map 10-band bare-earth chips to SpectralGPT's 12-band Sentinel-2 ordering.

    Expected target order:
    [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12]
    """
    if chips.ndim != 4:
        raise ValueError(f"Expected chips shape [N,H,W,B], got {chips.shape}")
    if len(band_names) != chips.shape[-1]:
        raise ValueError(
            f"band_names length {len(band_names)} != chips bands {chips.shape[-1]}"
        )

    idx = {str(name).strip().lower(): i for i, name in enumerate(band_names)}

    def _pick(*aliases: str) -> np.ndarray:
        for alias in aliases:
            if alias in idx:
                return chips[..., idx[alias]]
        return np.full(chips.shape[:3], np.nan, dtype=np.float32)

    # Bare-earth currently excludes B1 and B9. We proxy them with nearest
    # available bands to keep the official model's expected 12-channel interface.
    b2 = _pick("blue", "b2")
    b3 = _pick("green", "b3")
    b4 = _pick("red", "b4")
    b5 = _pick("red_edge_1", "rededge_1", "b5")
    b6 = _pick("red_edge_2", "rededge_2", "b6")
    b7 = _pick("red_edge_3", "rededge_3", "b7")
    b8 = _pick("nir", "b8")
    b8a = _pick("nir_2", "narrow_nir", "b8a")
    b11 = _pick("swir1", "b11")
    b12 = _pick("swir2", "b12")
    b1 = b2  # proxy coastal aerosol band
    b9 = b8a  # proxy water-vapor band

    stacked = np.stack([b1, b2, b3, b4, b5, b6, b7, b8, b8a, b9, b11, b12], axis=1)
    return _safe_nan_minmax_scale_per_band(stacked)


def reduce_embeddings_with_pca(
    embeddings: np.ndarray,
    n_components: int = 16,
    random_state: int = 42,
) -> np.ndarray:
    """Dimension reduction helper for high-dimensional embedding vectors."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got {embeddings.shape}")
    n_components = int(min(n_components, embeddings.shape[1], max(1, embeddings.shape[0] - 1)))
    if float(np.nanvar(embeddings)) < 1e-10:
        return np.zeros((embeddings.shape[0], n_components), dtype=np.float32)
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(embeddings.astype(np.float32)).astype(np.float32)


class OfficialSpectralGPTEncoder:
    """
    Wrapper for official pretrained SpectralGPT (IEEE TPAMI) encoder inference.

    This uses the official repository code and Zenodo checkpoints.
    """

    def __init__(self, config: OfficialSpectralGPTConfig = OfficialSpectralGPTConfig()):
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        repo_dir = (
            Path(self.config.repo_dir)
            if self.config.repo_dir
            else Path.home() / ".cache" / "soil-resnet-model" / "IEEE_TPAMI_SpectralGPT"
        )
        repo_dir = _ensure_official_repo(repo_dir, self.config.repo_url)
        checkpoint_path = (
            Path(self.config.checkpoint_path)
            if self.config.checkpoint_path
            else repo_dir / "checkpoints" / "SpectralGPT+.pth"
        )
        checkpoint_path = _download_checkpoint(
            checkpoint_url=self.config.checkpoint_url,
            checkpoint_path=checkpoint_path,
        )

        repo_dir_str = str(repo_dir.resolve())
        if repo_dir_str not in sys.path:
            sys.path.insert(0, repo_dir_str)

        models_mae_spectral = importlib.import_module("models_mae_spectral")
        if not hasattr(models_mae_spectral, self.config.model_name):
            raise ValueError(
                f"Model '{self.config.model_name}' not found in official repo."
            )
        model_ctor = getattr(models_mae_spectral, self.config.model_name)
        model = model_ctor()

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        msg = model.load_state_dict(state_dict, strict=False)
        if len(msg.missing_keys) > 0:
            # MAE checkpoints are expected to match; unexpected missing keys may indicate wrong model variant.
            missing = ", ".join(msg.missing_keys[:8])
            raise RuntimeError(
                f"Official checkpoint/model mismatch. Missing keys include: {missing}"
            )

        model.eval().to(self.device)
        return model

    def encode_chips(
        self,
        chips: np.ndarray,
        band_names: list[str],
    ) -> np.ndarray:
        """
        Encode bare-earth chips to official SpectralGPT latent embeddings.

        Args:
            chips: [N, H, W, B] float array.
            band_names: Names for B channels.

        Returns:
            [N, D] float array in latent space (D=768 for base model).
        """
        _set_seed(self.config.seed)
        x = _to_spectralgpt_s2_12(chips, band_names=band_names)  # [N, 12, H, W]

        n = x.shape[0]
        embeds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, self.config.batch_size):
                end = min(start + self.config.batch_size, n)
                batch = torch.from_numpy(x[start:end]).to(self.device)
                latent, _, _ = self.model.forward_encoder(batch, mask_ratio=0.0)
                pooled = torch.mean(latent, dim=1)
                embeds.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(embeds) if embeds else np.zeros((0, 0), dtype=np.float32)


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
