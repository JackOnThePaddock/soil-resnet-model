"""
ResNet Soil Prediction Model - Google Colab Version
====================================================
Upload this file and your CSV to Colab, then run.

Usage in Colab:
    1. Upload this file and your CSV
    2. Run: !python resnet_colab.py --data "soil_data_alphaearth_normalized.csv"
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'input_dim': 64,
    'hidden_dim': 128,
    'num_res_blocks': 2,
    'dropout': 0.2,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 300,
    'patience': 30,
    'ensemble_size': 5,
    'n_folds': 5,
    'grad_clip': 1.0,
    'targets': ['ph', 'cec', 'esp', 'soc', 'ca', 'mg', 'na'],
    'feature_cols': [f'band_{i}' for i in range(64)],
}


# ============================================================================
# Dataset
# ============================================================================

class SoilDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ============================================================================
# Model
# ============================================================================

class ResidualBlock(nn.Module):
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

    def forward(self, x):
        return self.dropout(self.activation(self.block(x) + x))


class NationalSoilNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_res_blocks=2,
                 dropout=0.2, target_names=None):
        super().__init__()
        self.target_names = target_names or CONFIG['targets']

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            ) for name in self.target_names
        })

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return {name: head(x).squeeze(-1) for name, head in self.heads.items()}

    def forward_stacked(self, x):
        outputs = self.forward(x)
        return torch.stack([outputs[name] for name in self.target_names], dim=1)


# ============================================================================
# Training
# ============================================================================

def masked_mse_loss(predictions, targets):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    return nn.functional.mse_loss(predictions[mask], targets[mask])


def train_ensemble(X, y, target_names, config, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(models_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    all_metrics = []
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=42)

    for model_idx in range(config['ensemble_size']):
        print(f"\n{'='*50}")
        print(f"Training Model {model_idx + 1}/{config['ensemble_size']}")
        print(f"{'='*50}")

        torch.manual_seed(42 + model_idx * 1000)
        np.random.seed(42 + model_idx * 1000)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            print(f"\n  Fold {fold + 1}/{config['n_folds']}")

            train_loader = DataLoader(
                SoilDataset(X_scaled[train_idx], y[train_idx]),
                batch_size=config['batch_size'], shuffle=True
            )
            val_loader = DataLoader(
                SoilDataset(X_scaled[val_idx], y[val_idx]),
                batch_size=config['batch_size'], shuffle=False
            )

            model = NationalSoilNet(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                num_res_blocks=config['num_res_blocks'],
                dropout=config['dropout'],
                target_names=target_names,
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=15
            )

            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(config['epochs']):
                # Train
                model.train()
                for features, targets in train_loader:
                    features, targets = features.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = masked_mse_loss(model.forward_stacked(features), targets)
                    if loss.item() > 0:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                        optimizer.step()

                # Validate
                model.eval()
                val_loss = 0
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for features, targets in val_loader:
                        features, targets = features.to(device), targets.to(device)
                        preds = model.forward_stacked(features)
                        val_loss += masked_mse_loss(preds, targets).item()
                        val_preds.append(preds.cpu().numpy())
                        val_targets.append(targets.cpu().numpy())

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}: val_loss={val_loss:.4f}")

                if patience_counter >= config['patience']:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            model.load_state_dict(best_state)

            # Calculate fold metrics
            val_preds = np.vstack(val_preds)
            val_targets = np.vstack(val_targets)

            metrics = {}
            for i, name in enumerate(target_names):
                mask = ~np.isnan(val_targets[:, i])
                if mask.sum() > 0:
                    metrics[name] = {
                        'r2': r2_score(val_targets[mask, i], val_preds[mask, i]),
                        'rmse': np.sqrt(mean_squared_error(val_targets[mask, i], val_preds[mask, i])),
                    }
            fold_metrics.append(metrics)

        # Train final model on all data
        print(f"\n  Training final model {model_idx + 1} on all data...")

        full_loader = DataLoader(
            SoilDataset(X_scaled, y), batch_size=config['batch_size'], shuffle=True
        )

        final_model = NationalSoilNet(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_res_blocks=config['num_res_blocks'],
            dropout=config['dropout'],
            target_names=target_names,
        ).to(device)

        optimizer = torch.optim.AdamW(
            final_model.parameters(), lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        final_model.train()
        for epoch in range(min(150, config['epochs'])):
            for features, targets in full_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = masked_mse_loss(final_model.forward_stacked(features), targets)
                if loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), config['grad_clip'])
                    optimizer.step()

        # Save
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'config': config,
            'target_names': target_names,
        }, models_dir / f'model_{model_idx + 1}.pth')

        # Average metrics
        avg_metrics = {}
        for target in target_names:
            r2s = [fm[target]['r2'] for fm in fold_metrics if target in fm]
            if r2s:
                avg_metrics[target] = np.mean(r2s)

        all_metrics.append(avg_metrics)
        print(f"\n  Model {model_idx + 1} CV Results:")
        for t, r2 in avg_metrics.items():
            print(f"    {t}: R2={r2:.3f}")

    # Final summary
    print(f"\n{'='*50}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*50}")
    print("\nAverage R2 across all models:")
    for target in target_names:
        r2s = [m.get(target, 0) for m in all_metrics]
        print(f"  {target}: {np.mean(r2s):.3f} (+/- {np.std(r2s):.3f})")

    return models_dir


def load_data(csv_path, feature_cols, target_cols):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    # Try to find feature columns
    feature_cols_lower = [c.lower() for c in feature_cols]
    if not all(c in df.columns for c in feature_cols_lower):
        # Try alternative naming
        alt_cols = [f'a{i:02d}' for i in range(64)]
        if all(c in df.columns for c in alt_cols):
            feature_cols_lower = alt_cols

    target_cols_lower = [c.lower() for c in target_cols]
    valid_targets = [c for c in target_cols_lower if c in df.columns]

    X = df[feature_cols_lower].values
    y = df[valid_targets].values

    # Remove rows with NaN features
    valid_mask = ~np.isnan(X).any(axis=1)
    X, y = X[valid_mask], y[valid_mask]

    print(f"Loaded {len(X)} samples with {len(valid_targets)} targets")
    for i, t in enumerate(valid_targets):
        print(f"  {t}: {np.sum(~np.isnan(y[:, i]))} valid values")

    return X, y, valid_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV')
    parser.add_argument('--output', default='./resnet_output', help='Output directory')
    parser.add_argument('--ensemble_size', type=int, default=5)
    args = parser.parse_args()

    config = CONFIG.copy()
    config['ensemble_size'] = args.ensemble_size

    X, y, targets = load_data(args.data, config['feature_cols'], config['targets'])
    models_dir = train_ensemble(X, y, targets, config, args.output)

    print(f"\nModels saved to: {models_dir}")


if __name__ == '__main__':
    main()
