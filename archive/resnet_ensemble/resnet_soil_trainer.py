"""
ResNet Soil Prediction Model with Ensemble Training
====================================================
Trains an ensemble of ResNet models for soil property prediction
from AlphaEarth satellite embeddings (64 bands).

Usage:
    python resnet_soil_trainer.py --data "path/to/training_data.csv"
    python resnet_soil_trainer.py --data "path/to/training_data.csv" --ensemble_size 5
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'input_dim': 64,           # AlphaEarth bands
    'hidden_dim': 128,         # Backbone width
    'num_res_blocks': 2,       # Depth of ResNet
    'dropout': 0.2,            # Regularization
    'learning_rate': 1e-4,     # Lower LR to prevent exploding gradients
    'weight_decay': 1e-4,      # L2 regularization
    'batch_size': 32,
    'epochs': 500,
    'patience': 50,            # Early stopping patience
    'ensemble_size': 5,        # Number of models
    'n_folds': 5,              # K-Fold cross-validation
    'grad_clip': 1.0,          # Gradient clipping
    'targets': ['ph', 'cec', 'esp', 'soc', 'ca', 'mg', 'na'],
    'feature_cols': [f'A{i:02d}' for i in range(64)],
    'feature_cols_alt': [f'band_{i}' for i in range(64)],  # Alternative naming
    'output_dir': Path(r'C:\Users\jackc\Downloads\EW WH & MG SPEIRS\SOIL Tests\exports\resnet_ensemble'),
}


# ============================================================================
# Dataset
# ============================================================================

class SoilDataset(Dataset):
    """PyTorch Dataset for soil property prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


# ============================================================================
# Model Architecture
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection for tabular data."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.activation(out)
        out = self.dropout(out)
        return out


class NationalSoilNet(nn.Module):
    """
    ResNet-style model for multi-target soil property prediction.

    Architecture:
        Input (64) -> Linear(128) -> BN -> SiLU -> [ResBlocks] -> Multi-head outputs
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.2,
        target_names: Optional[List[str]] = None,
    ):
        super().__init__()

        self.target_names = target_names or CONFIG['targets']
        self.num_targets = len(self.target_names)

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        # Multi-head output layers (one per target)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for name in self.target_names
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared backbone
        x = self.input_layer(x)
        x = self.res_blocks(x)

        # Multi-head predictions
        outputs = {name: head(x).squeeze(-1) for name, head in self.heads.items()}
        return outputs

    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions as stacked tensor [batch, num_targets]."""
        outputs = self.forward(x)
        return torch.stack([outputs[name] for name in self.target_names], dim=1)


# ============================================================================
# Training Functions
# ============================================================================

def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss ignoring NaN targets."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    return torch.nn.functional.mse_loss(predictions[mask], targets[mask])


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
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

        if loss.item() > 0:  # Only backprop if we have valid targets
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
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

    # Calculate per-target metrics
    metrics = {}
    for i, name in enumerate(target_names):
        pred = all_preds[:, i]
        true = all_targets[:, i]

        # Skip if all NaN
        mask = ~np.isnan(true)
        if mask.sum() == 0:
            continue

        pred = pred[mask]
        true = true[mask]

        metrics[name] = {
            'r2': r2_score(true, pred),
            'rmse': np.sqrt(mean_squared_error(true, pred)),
            'mae': mean_absolute_error(true, pred),
        }

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
    """
    Train a single model with early stopping.

    Returns trained model and training history.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_state = None

    for epoch in range(config['epochs']):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, target_names,
            grad_clip=config.get('grad_clip', 1.0)
        )
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, target_names
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()

            # Save best model
            if model_save_path:
                torch.save({
                    'model_state_dict': best_state,
                    'config': config,
                    'target_names': target_names,
                    'val_loss': best_val_loss,
                    'epoch': epoch,
                }, model_save_path)
        else:
            patience_counter += 1

        # Log progress
        if (epoch + 1) % 25 == 0 or epoch == 0:
            avg_r2 = np.mean([m['r2'] for m in val_metrics.values()])
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, avg_R2={avg_r2:.3f}")

        if patience_counter >= config['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best state
    if best_state:
        model.load_state_dict(best_state)

    return model, history


# ============================================================================
# Ensemble Training
# ============================================================================

def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    target_names: List[str],
    config: dict,
    output_dir: Path,
) -> Dict:
    """
    Train an ensemble of models with K-Fold cross-validation.

    Saves:
        - model_1.pth through model_N.pth
        - scaler.pkl
        - training_metrics.csv
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    models_dir = output_dir / 'models'
    metrics_dir = output_dir / 'metrics'
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Track metrics across ensemble
    all_fold_metrics = []
    ensemble_metrics = []

    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=42)

    for model_idx in range(config['ensemble_size']):
        print(f"\n{'='*60}")
        print(f"Training Model {model_idx + 1}/{config['ensemble_size']}")
        print(f"{'='*60}")

        # Different seed for each ensemble member
        torch.manual_seed(42 + model_idx * 1000)
        np.random.seed(42 + model_idx * 1000)

        fold_metrics = []

        # K-Fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            print(f"\n  Fold {fold + 1}/{config['n_folds']}")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = SoilDataset(X_train, y_train)
            val_dataset = SoilDataset(X_val, y_val)

            train_loader = DataLoader(
                train_dataset, batch_size=config['batch_size'], shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config['batch_size'], shuffle=False
            )

            # Create model
            model = NationalSoilNet(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                num_res_blocks=config['num_res_blocks'],
                dropout=config['dropout'],
                target_names=target_names,
            ).to(device)

            # Train
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                target_names=target_names,
                model_save_path=None,  # Save final model after CV
            )

            # Record fold metrics
            final_metrics = history['val_metrics'][-1]
            fold_metrics.append(final_metrics)

            for target, m in final_metrics.items():
                print(f"    {target}: R2={m['r2']:.3f}, RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}")

        # Train final model on all data for this ensemble member
        print(f"\n  Training final model {model_idx + 1} on all data...")

        full_dataset = SoilDataset(X_scaled, y)
        full_loader = DataLoader(
            full_dataset, batch_size=config['batch_size'], shuffle=True
        )

        final_model = NationalSoilNet(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_res_blocks=config['num_res_blocks'],
            dropout=config['dropout'],
            target_names=target_names,
        ).to(device)

        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
        grad_clip = config.get('grad_clip', 1.0)

        # Train for fixed epochs (no validation split)
        final_model.train()
        for epoch in range(min(200, config['epochs'])):
            for features, targets in full_loader:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predictions = final_model.forward_stacked(features)
                loss = masked_mse_loss(predictions, targets)
                if loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), grad_clip)
                    optimizer.step()

        # Save model
        model_path = models_dir / f'model_{model_idx + 1}.pth'
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'config': config,
            'target_names': target_names,
        }, model_path)
        print(f"  Saved: {model_path}")

        # Average fold metrics for this ensemble member
        avg_metrics = {}
        for target in target_names:
            if target in fold_metrics[0]:
                avg_metrics[target] = {
                    'r2': np.mean([fm[target]['r2'] for fm in fold_metrics]),
                    'rmse': np.mean([fm[target]['rmse'] for fm in fold_metrics]),
                    'mae': np.mean([fm[target]['mae'] for fm in fold_metrics]),
                }

        ensemble_metrics.append({
            'model': model_idx + 1,
            **{f'{t}_r2': avg_metrics[t]['r2'] for t in avg_metrics},
            **{f'{t}_rmse': avg_metrics[t]['rmse'] for t in avg_metrics},
            **{f'{t}_mae': avg_metrics[t]['mae'] for t in avg_metrics},
        })
        all_fold_metrics.extend(fold_metrics)

    # Save metrics
    metrics_df = pd.DataFrame(ensemble_metrics)
    metrics_df.to_csv(metrics_dir / 'ensemble_metrics.csv', index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("Ensemble Training Complete")
    print(f"{'='*60}")
    print(f"\nAverage metrics across {config['ensemble_size']} models:")

    for target in target_names:
        r2_col = f'{target}_r2'
        if r2_col in metrics_df.columns:
            avg_r2 = metrics_df[r2_col].mean()
            avg_rmse = metrics_df[f'{target}_rmse'].mean()
            avg_mae = metrics_df[f'{target}_mae'].mean()
            print(f"  {target}: R2={avg_r2:.3f}, RMSE={avg_rmse:.3f}, MAE={avg_mae:.3f}")

    return {
        'ensemble_metrics': ensemble_metrics,
        'scaler_path': str(models_dir / 'scaler.pkl'),
        'models_dir': str(models_dir),
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(
    csv_path: str,
    feature_cols: List[str],
    target_cols: List[str],
    feature_cols_alt: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and validate training data from CSV.

    Returns:
        X: Feature array
        y: Target array
        valid_targets: List of target columns found in data
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} samples")

    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    feature_cols_lower = [c.lower() for c in feature_cols]
    target_cols_lower = [c.lower() for c in target_cols]

    # Check for feature columns - try primary naming first, then alternative
    missing_features = [c for c in feature_cols_lower if c not in df.columns]
    if missing_features and feature_cols_alt:
        # Try alternative naming (e.g., band_0 instead of A00)
        feature_cols_alt_lower = [c.lower() for c in feature_cols_alt]
        missing_alt = [c for c in feature_cols_alt_lower if c not in df.columns]
        if not missing_alt:
            print(f"  Using alternative feature naming: band_0 to band_63")
            feature_cols_lower = feature_cols_alt_lower
            missing_features = []

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features[:5]}...")

    # Find available target columns
    valid_targets = [c for c in target_cols_lower if c in df.columns]
    if not valid_targets:
        raise ValueError(f"No target columns found. Expected any of: {target_cols}")

    print(f"  Found targets: {valid_targets}")

    # Extract features and targets
    X = df[feature_cols_lower].values
    y = df[valid_targets].values

    # Handle missing feature values (drop rows with NaN features)
    valid_feature_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_feature_mask]
    y = y[valid_feature_mask]

    # Report per-target sample counts (targets can have NaN, handled during training)
    print(f"  Samples with valid features: {len(X)}")
    for i, target in enumerate(valid_targets):
        n_valid = np.sum(~np.isnan(y[:, i]))
        print(f"    {target}: {n_valid} samples")

    return X, y, valid_targets


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train ResNet ensemble for soil property prediction'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--ensemble_size', type=int, default=CONFIG['ensemble_size'],
        help=f'Number of ensemble members (default: {CONFIG["ensemble_size"]})'
    )
    parser.add_argument(
        '--output', type=str, default=str(CONFIG['output_dir']),
        help='Output directory for models and metrics'
    )
    parser.add_argument(
        '--epochs', type=int, default=CONFIG['epochs'],
        help=f'Maximum training epochs (default: {CONFIG["epochs"]})'
    )
    parser.add_argument(
        '--batch_size', type=int, default=CONFIG['batch_size'],
        help=f'Batch size (default: {CONFIG["batch_size"]})'
    )

    args = parser.parse_args()

    # Update config
    config = CONFIG.copy()
    config['ensemble_size'] = args.ensemble_size
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    output_dir = Path(args.output)

    print("\n" + "="*60)
    print("ResNet Soil Prediction - Ensemble Training")
    print("="*60)
    print(f"Ensemble size: {config['ensemble_size']}")
    print(f"Output directory: {output_dir}")

    # Load data
    X, y, valid_targets = load_training_data(
        csv_path=args.data,
        feature_cols=config['feature_cols'],
        target_cols=config['targets'],
        feature_cols_alt=config.get('feature_cols_alt'),
    )

    # Train ensemble
    results = train_ensemble(
        X=X,
        y=y,
        target_names=valid_targets,
        config=config,
        output_dir=output_dir,
    )

    print(f"\nModels saved to: {results['models_dir']}")
    print(f"Scaler saved to: {results['scaler_path']}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
