"""
Ensemble loader and prediction interface for trained NationalSoilNet models.

Supports:
- Point predictions with uncertainty (ensemble std)
- Optional inverse-transform from model target space to raw units
- Optional target specialist blending for sparse targets
- Batch predictions from DataFrames
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.models.resnet import DEFAULT_TARGETS, NationalSoilNet


class SoilEnsemble:
    """
    Loads and manages an ensemble of trained NationalSoilNet models.

    Args:
        models_dir: Directory containing model_1.pth ... model_N.pth and scaler.pkl
        device: Torch device (auto-detected if None)
    """

    def __init__(
        self,
        models_dir: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        self.models_dir = Path(models_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        model_files = sorted(self.models_dir.glob("model_*.pth"))
        model_files = [f for f in model_files if re.fullmatch(r"model_\d+\.pth", f.name)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")

        self.models: List[NationalSoilNet] = []
        self.target_names: Optional[List[str]] = None
        self.config: Dict[str, object] = {}
        self.target_transform_types: Dict[str, str] = {}
        self.specialists: Dict[str, List[NationalSoilNet]] = {}
        self.specialist_blend_weight: float = 0.4

        for model_path in model_files:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if self.target_names is None:
                self.config = checkpoint.get("config", {}) or {}
                self.target_names = checkpoint.get("target_names", DEFAULT_TARGETS)
                self.target_transform_types = {
                    k.lower(): str(v).lower()
                    for k, v in (
                        checkpoint.get("target_transform_types")
                        or self.config.get("target_transform_types")
                        or {}
                    ).items()
                }
                if not self.target_transform_types:
                    transforms_path = self.models_dir / "target_transforms.json"
                    if transforms_path.exists():
                        with open(transforms_path, "r", encoding="utf-8") as f:
                            self.target_transform_types = {
                                k.lower(): str(v).lower() for k, v in json.load(f).items()
                            }
                self.specialist_blend_weight = float(
                    self.config.get("specialist_blend_weight", 0.4)
                )

            model = NationalSoilNet(
                input_dim=self.config.get("input_dim", 64),
                hidden_dim=self.config.get("hidden_dim", 128),
                num_res_blocks=self.config.get("num_res_blocks", 2),
                dropout=self.config.get("dropout", 0.2),
                target_names=self.target_names,
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models.append(model)

        if self.target_names is None:
            self.target_names = DEFAULT_TARGETS
        self.n_features: int = int(getattr(self.scaler, "n_features_in_", 64))
        self._load_specialists()

    def _load_specialists(self) -> None:
        """Load optional per-target specialist models if present."""
        specialists_dir = self.models_dir / "specialists"
        if not specialists_dir.exists():
            return

        specialist_files = sorted(specialists_dir.glob("specialist_*.pth"))
        for path in specialist_files:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            target_names = checkpoint.get("target_names", [])
            if len(target_names) != 1:
                continue
            target = str(target_names[0]).lower()
            if target not in self.target_names:
                continue

            model = NationalSoilNet(
                input_dim=self.config.get("input_dim", 64),
                hidden_dim=self.config.get("hidden_dim", 128),
                num_res_blocks=self.config.get("num_res_blocks", 2),
                dropout=self.config.get("dropout", 0.2),
                target_names=[target],
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.specialists.setdefault(target, []).append(model)

    def __len__(self) -> int:
        return len(self.models)

    def _validate_input_array(self, X: np.ndarray) -> np.ndarray:
        """Validate and normalize input feature array shape."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_arr.shape}")
        if X_arr.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X_arr.shape[1]}")
        return X_arr

    def _resolve_feature_columns(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """Resolve feature columns in deterministic order with validation."""
        lower_to_original = {c.lower(): c for c in df.columns}

        def _resolve_explicit(columns: List[str]) -> List[str]:
            missing = [c for c in columns if c.lower() not in lower_to_original]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            resolved = [lower_to_original[c.lower()] for c in columns]
            if len(resolved) != self.n_features:
                raise ValueError(
                    f"Expected {self.n_features} features, got {len(resolved)} from explicit columns"
                )
            return resolved

        if feature_cols is not None:
            return _resolve_explicit(feature_cols)

        trained_feature_cols = self.config.get("feature_cols") if self.config else None
        if isinstance(trained_feature_cols, list) and trained_feature_cols:
            try:
                return _resolve_explicit(trained_feature_cols)
            except ValueError:
                pass

        for pattern in (r"^feat_(\d+)$", r"^band_(\d+)$", r"^ae_(\d+)$", r"^a(\d+)$"):
            matched_cols = []
            regex = re.compile(pattern)
            for col in df.columns:
                match = regex.match(col.lower())
                if match:
                    matched_cols.append((int(match.group(1)), col))

            if matched_cols:
                matched_cols = sorted(matched_cols, key=lambda item: item[0])
                resolved = [col for _, col in matched_cols]
                if len(resolved) == self.n_features:
                    return resolved

        raise ValueError(
            "Could not infer feature columns. Pass `feature_cols` explicitly "
            f"with {self.n_features} columns."
        )

    @staticmethod
    def _inverse_transform_scalar(z: np.ndarray, transform_name: str) -> np.ndarray:
        """Inverse-transform 1D numpy array from model space to raw space."""
        t = str(transform_name).lower()
        out = z.astype(np.float32, copy=True)
        valid = np.isfinite(out)
        if valid.sum() == 0:
            return out
        if t == "identity":
            return out
        if t == "log1p":
            out[valid] = np.expm1(np.clip(out[valid], a_min=-20.0, a_max=20.0))
            return out
        if t == "signed_log1p":
            out[valid] = np.sign(out[valid]) * np.expm1(
                np.clip(np.abs(out[valid]), a_min=0.0, a_max=20.0)
            )
            return out
        return out

    @staticmethod
    def _inverse_std_scalar(
        mean_z: np.ndarray,
        std_z: np.ndarray,
        transform_name: str,
    ) -> np.ndarray:
        """Approximate std conversion via delta method."""
        t = str(transform_name).lower()
        mean = mean_z.astype(np.float32, copy=False)
        std = std_z.astype(np.float32, copy=False)
        if t == "identity":
            return std
        if t == "log1p":
            deriv = np.exp(mean)
            return np.abs(deriv) * std
        if t == "signed_log1p":
            deriv = np.exp(np.abs(mean))
            return np.abs(deriv) * std
        return std

    def _inverse_transform_predictions(
        self,
        mean_preds_z: np.ndarray,
        std_preds_z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse-transform full prediction matrices to raw target units."""
        mean_raw = mean_preds_z.astype(np.float32, copy=True)
        std_raw = std_preds_z.astype(np.float32, copy=True)
        for i, target in enumerate(self.target_names):
            tname = self.target_transform_types.get(target, "identity")
            mean_raw[:, i] = self._inverse_transform_scalar(mean_raw[:, i], tname)
            std_raw[:, i] = self._inverse_std_scalar(mean_preds_z[:, i], std_raw[:, i], tname)
        return mean_raw, std_raw

    def _apply_specialist_blend(
        self,
        X_scaled: np.ndarray,
        mean_raw: np.ndarray,
        std_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend ensemble predictions with optional specialist models per target."""
        if not self.specialists:
            return mean_raw, std_raw

        blend = float(np.clip(self.specialist_blend_weight, 0.0, 1.0))
        if blend <= 0:
            return mean_raw, std_raw

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            for target, models in self.specialists.items():
                if target not in self.target_names or not models:
                    continue
                target_idx = self.target_names.index(target)
                specialist_preds_z = []
                for model in models:
                    pred = model.forward_stacked(X_tensor).cpu().numpy().reshape(-1)
                    specialist_preds_z.append(pred)
                specialist_preds_z = np.vstack(specialist_preds_z)
                specialist_mean_z = np.mean(specialist_preds_z, axis=0)
                specialist_std_z = np.std(specialist_preds_z, axis=0)

                tname = self.target_transform_types.get(target, "identity")
                specialist_mean_raw = self._inverse_transform_scalar(specialist_mean_z, tname)
                specialist_std_raw = self._inverse_std_scalar(
                    specialist_mean_z, specialist_std_z, tname
                )

                mean_raw[:, target_idx] = (
                    (1.0 - blend) * mean_raw[:, target_idx]
                    + blend * specialist_mean_raw
                )
                std_raw[:, target_idx] = np.sqrt(
                    (1.0 - blend) ** 2 * std_raw[:, target_idx] ** 2
                    + blend**2 * specialist_std_raw**2
                )

        return mean_raw, std_raw

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate ensemble predictions from raw (unscaled) features.

        Returns predictions in raw target units.
        """
        X_arr = self._validate_input_array(X)
        X_scaled = self.scaler.transform(X_arr)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        all_preds_z = []
        with torch.no_grad():
            for model in self.models:
                preds = model.forward_stacked(X_tensor)
                all_preds_z.append(preds.cpu().numpy())

        all_preds_z = np.stack(all_preds_z, axis=0)  # (n_models, n_samples, n_targets)
        mean_preds_z = np.mean(all_preds_z, axis=0)
        std_preds_z = np.std(all_preds_z, axis=0)

        mean_raw, std_raw = self._inverse_transform_predictions(mean_preds_z, std_preds_z)
        mean_raw, std_raw = self._apply_specialist_blend(X_scaled, mean_raw, std_raw)

        if return_std:
            return mean_raw, std_raw
        return mean_raw

    def predict_batch(
        self,
        X: np.ndarray,
        batch_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-efficient batch prediction for large arrays.

        Returns predictions in raw target units.
        """
        X_arr = self._validate_input_array(X)
        n_samples = X_arr.shape[0]
        n_targets = len(self.target_names)
        all_model_preds_z = np.zeros((len(self.models), n_samples, n_targets), dtype=np.float32)

        X_scaled = self.scaler.transform(X_arr)

        with torch.no_grad():
            for m_idx, model in enumerate(self.models):
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    X_tensor = torch.FloatTensor(X_scaled[start:end]).to(self.device)
                    preds = model.forward_stacked(X_tensor).cpu().numpy()
                    all_model_preds_z[m_idx, start:end] = preds

        mean_preds_z = np.mean(all_model_preds_z, axis=0)
        std_preds_z = np.std(all_model_preds_z, axis=0)
        mean_raw, std_raw = self._inverse_transform_predictions(mean_preds_z, std_preds_z)
        mean_raw, std_raw = self._apply_specialist_blend(X_scaled, mean_raw, std_raw)
        return mean_raw, std_raw

    def predict_df(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions from a DataFrame, appending prediction columns.

        Returns:
            DataFrame with {target}_pred and {target}_std columns appended
        """
        resolved_feature_cols = self._resolve_feature_columns(df, feature_cols=feature_cols)
        X = df[resolved_feature_cols].values

        mean_preds, std_preds = self.predict(X, return_std=True)

        result = df.copy()
        for i, target in enumerate(self.target_names):
            result[f"{target}_pred"] = mean_preds[:, i]
            result[f"{target}_std"] = std_preds[:, i]

        return result
