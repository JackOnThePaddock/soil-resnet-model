"""
ResNet ensemble training with K-Fold cross-validation and early stopping.

Usage:
    python -m src.training.train_resnet --data path/to/data.csv --output models/resnet_ensemble
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.models.resnet import NationalSoilNet
from src.models.dataset import SoilDataset
from src.training.losses import masked_mse_loss
from src.evaluation.metrics import compute_metrics


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_names: List[str],
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch, returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model.forward_stacked(features)
        loss = masked_mse_loss(predictions, targets)

        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_names: List[str],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Validate model, returns loss and per-target metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model.forward_stacked(features)
            loss = masked_mse_loss(predictions, targets)

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    metrics = compute_metrics(all_targets, all_preds, target_names)

    return total_loss / total_samples if total_samples > 0 else 0.0, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    target_names: List[str],
    model_save_path: Optional[Path] = None,
) -> Tuple[nn.Module, Dict]:
    """Train a single model with early stopping."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_state = None

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, target_names,
            grad_clip=config.get("grad_clip", 1.0),
        )
        val_loss, val_metrics = validate_epoch(model, val_loader, device, target_names)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
            if model_save_path:
                torch.save({
                    "model_state_dict": best_state,
                    "config": config,
                    "target_names": target_names,
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                }, model_save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            avg_r2 = np.mean([m["r2"] for m in val_metrics.values()])
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, avg_R2={avg_r2:.3f}")

        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    target_names: List[str],
    config: dict,
    output_dir: Path,
) -> Dict:
    """Train ensemble of models with K-Fold CV. Saves model_*.pth + scaler.pkl."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    ensemble_metrics = []
    kfold = KFold(n_splits=config["n_folds"], shuffle=True, random_state=42)

    for model_idx in range(config["ensemble_size"]):
        print(f"\n{'='*60}")
        print(f"Training Model {model_idx + 1}/{config['ensemble_size']}")
        print(f"{'='*60}")

        torch.manual_seed(42 + model_idx * 1000)
        np.random.seed(42 + model_idx * 1000)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            print(f"\n  Fold {fold + 1}/{config['n_folds']}")

            train_loader = DataLoader(
                SoilDataset(X_scaled[train_idx], y[train_idx]),
                batch_size=config["batch_size"], shuffle=True,
            )
            val_loader = DataLoader(
                SoilDataset(X_scaled[val_idx], y[val_idx]),
                batch_size=config["batch_size"], shuffle=False,
            )

            model = NationalSoilNet(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                num_res_blocks=config["num_res_blocks"],
                dropout=config["dropout"],
                target_names=target_names,
            ).to(device)

            model, history = train_model(
                model, train_loader, val_loader, config, device, target_names,
            )

            final_metrics = history["val_metrics"][-1]
            fold_metrics.append(final_metrics)
            for target, m in final_metrics.items():
                print(f"    {target}: R2={m['r2']:.3f}, RMSE={m['rmse']:.3f}")

        # Train final model on all data
        print(f"\n  Training final model {model_idx + 1} on all data...")
        full_loader = DataLoader(
            SoilDataset(X_scaled, y), batch_size=config["batch_size"], shuffle=True,
        )

        final_model = NationalSoilNet(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_res_blocks=config["num_res_blocks"],
            dropout=config["dropout"],
            target_names=target_names,
        ).to(device)

        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        final_model.train()
        for epoch in range(min(200, config["epochs"])):
            for features, targets in full_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                predictions = final_model.forward_stacked(features)
                loss = masked_mse_loss(predictions, targets)
                if loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        final_model.parameters(), config.get("grad_clip", 1.0)
                    )
                    optimizer.step()

        model_path = output_dir / f"model_{model_idx + 1}.pth"
        torch.save({
            "model_state_dict": final_model.state_dict(),
            "config": config,
            "target_names": target_names,
        }, model_path)
        print(f"  Saved: {model_path}")

        avg_metrics = {}
        for target in target_names:
            if target in fold_metrics[0]:
                avg_metrics[target] = {
                    "r2": np.mean([fm[target]["r2"] for fm in fold_metrics]),
                    "rmse": np.mean([fm[target]["rmse"] for fm in fold_metrics]),
                }
        ensemble_metrics.append({
            "model": model_idx + 1,
            **{f"{t}_r2": avg_metrics[t]["r2"] for t in avg_metrics},
            **{f"{t}_rmse": avg_metrics[t]["rmse"] for t in avg_metrics},
        })

    pd.DataFrame(ensemble_metrics).to_csv(output_dir / "ensemble_metrics.csv", index=False)
    print(f"\nEnsemble Training Complete")
    return {"models_dir": str(output_dir), "scaler_path": str(output_dir / "scaler.pkl")}


def load_training_data(
    csv_path: str,
    target_cols: List[str],
    feature_prefix: str = "A",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and validate training data from CSV."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    target_cols_lower = [c.lower() for c in target_cols]

    feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix.lower())])
    if not feature_cols:
        feature_cols = sorted([c for c in df.columns if c.startswith("band_")])
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}'")

    valid_targets = [c for c in target_cols_lower if c in df.columns]
    if not valid_targets:
        raise ValueError(f"No target columns found. Expected any of: {target_cols}")

    print(f"  Found {len(feature_cols)} features, targets: {valid_targets}")
    X = df[feature_cols].values
    y = df[valid_targets].values

    valid_mask = ~np.isnan(X).any(axis=1)
    X, y = X[valid_mask], y[valid_mask]
    print(f"  Samples with valid features: {len(X)}")
    return X, y, valid_targets
