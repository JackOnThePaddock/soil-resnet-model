"""
Ensemble loader and prediction interface for trained NationalSoilNet models.

Loads an ensemble of models from a directory and provides methods for:
- Point predictions with uncertainty (ensemble std)
- Batch predictions from DataFrames
- Memory-efficient batched inference for large arrays
"""

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.models.resnet import NationalSoilNet, DEFAULT_TARGETS


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

        # Load scaler
        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Find model files, excluding duplicates like "model_1 (1).pth"
        model_files = sorted(self.models_dir.glob("model_*.pth"))
        model_files = [f for f in model_files if re.fullmatch(r"model_\d+\.pth", f.name)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")

        self.models: List[NationalSoilNet] = []
        self.target_names: Optional[List[str]] = None
        self.config: Optional[dict] = None

        for model_path in model_files:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if self.config is None:
                self.config = checkpoint.get("config", {})
                self.target_names = checkpoint.get("target_names", DEFAULT_TARGETS)

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

    def __len__(self) -> int:
        return len(self.models)

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate ensemble predictions from raw (unscaled) features.

        Args:
            X: Feature array of shape (n_samples, 64)
            return_std: Whether to return uncertainty (ensemble std)

        Returns:
            If return_std=False: mean predictions (n_samples, n_targets)
            If return_std=True: (mean_predictions, std_predictions) tuple
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        all_preds = []
        with torch.no_grad():
            for model in self.models:
                preds = model.forward_stacked(X_tensor)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.stack(all_preds, axis=0)  # (n_models, n_samples, n_targets)
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)

        if return_std:
            return mean_preds, std_preds
        return mean_preds

    def predict_batch(
        self,
        X: np.ndarray,
        batch_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-efficient batch prediction for large arrays.

        Args:
            X: Raw feature array (n_samples, 64)
            batch_size: Samples per batch

        Returns:
            (mean_predictions, std_predictions) arrays
        """
        n_samples = X.shape[0]
        n_targets = len(self.target_names)
        all_model_preds = np.zeros(
            (len(self.models), n_samples, n_targets), dtype=np.float32
        )

        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            for m_idx, model in enumerate(self.models):
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    X_tensor = torch.FloatTensor(X_scaled[start:end]).to(self.device)
                    preds = model.forward_stacked(X_tensor).cpu().numpy()
                    all_model_preds[m_idx, start:end] = preds

        mean_preds = np.mean(all_model_preds, axis=0)
        std_preds = np.std(all_model_preds, axis=0)
        return mean_preds, std_preds

    def predict_df(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions from a DataFrame, appending prediction columns.

        Args:
            df: DataFrame with feature columns
            feature_cols: Feature column names (default: A00-A63)

        Returns:
            DataFrame with {target}_pred and {target}_std columns appended
        """
        if feature_cols is None:
            feature_cols = [f"A{i:02d}" for i in range(64)]

        feature_cols_lower = [c.lower() for c in feature_cols]
        X = df[[c for c in df.columns if c.lower() in feature_cols_lower]].values

        mean_preds, std_preds = self.predict(X, return_std=True)

        result = df.copy()
        for i, target in enumerate(self.target_names):
            result[f"{target}_pred"] = mean_preds[:, i]
            result[f"{target}_std"] = std_preds[:, i]

        return result
