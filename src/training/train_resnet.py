"""
ResNet ensemble training with robust loss, target transforms, and grouped CV support.

Usage:
    python -m src.training.train_resnet --data path/to/data.csv --output models/resnet_ensemble
"""

import copy
import json
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.models.dataset import SoilDataset
from src.models.resnet import NationalSoilNet
from src.training.losses import (
    esp_consistency_penalty,
    masked_mse_loss,
    masked_weighted_huber_loss,
)


def set_global_seed(seed: int) -> None:
    """Set seeds across libraries for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_sort_key(column_name: str) -> Tuple[int, int, str]:
    """Sort feature columns by trailing numeric suffix when available."""
    match = re.search(r"(\d+)$", column_name)
    if match:
        return (0, int(match.group(1)), column_name)
    return (1, -1, column_name)


def _find_feature_columns(columns: List[str], feature_prefix: str) -> List[str]:
    """Resolve feature columns from explicit prefix with robust fallback."""
    candidates: List[str] = []
    prefix = feature_prefix.lower().strip()
    if prefix and prefix != "auto":
        numeric_pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
        candidates = [c for c in columns if numeric_pattern.match(c)]
        if not candidates and prefix not in {"a", "ae_", "band_", "feat_"}:
            candidates = [c for c in columns if c.startswith(prefix)]

    if not candidates:
        for fallback_prefix in ("feat_", "band_", "ae_", "a"):
            numeric_pattern = re.compile(rf"^{re.escape(fallback_prefix)}\d+$")
            candidates = [c for c in columns if numeric_pattern.match(c)]
            if candidates:
                break

    return sorted(candidates, key=_feature_sort_key)


def _should_drop_last_train_batch(n_samples: int, batch_size: int) -> bool:
    """
    Avoid batch-norm instability from a final train batch of size 1.

    Returns True only when dropping the last mini-batch would remove exactly one sample.
    """
    return n_samples > batch_size and (n_samples % batch_size == 1)


def _resolve_transform_types(target_names: List[str], config: dict) -> Dict[str, str]:
    """Resolve per-target transform types from config."""
    user_cfg = {
        k.lower(): str(v).lower()
        for k, v in config.get("target_transforms", {}).items()
    }
    transform_types = {t: "identity" for t in target_names}

    if config.get("auto_target_transforms", False):
        for target in ("cec", "esp", "soc", "ca", "mg", "na"):
            if target in transform_types:
                transform_types[target] = "log1p"

    for t in target_names:
        if t in user_cfg:
            transform_types[t] = user_cfg[t]
    return transform_types


def _apply_target_transform(
    values: np.ndarray,
    transform_name: str,
    inverse: bool = False,
) -> np.ndarray:
    """Apply (or inverse) a scalar transform to non-NaN values."""
    out = values.copy()
    valid = np.isfinite(out)
    if valid.sum() == 0:
        return out

    if transform_name == "identity":
        return out
    if transform_name == "log1p":
        if inverse:
            out[valid] = np.expm1(np.clip(out[valid], a_min=-20.0, a_max=20.0))
        else:
            out[valid] = np.log1p(np.clip(out[valid], a_min=0.0, a_max=None))
        return out
    if transform_name == "signed_log1p":
        if inverse:
            out[valid] = np.sign(out[valid]) * np.expm1(
                np.clip(np.abs(out[valid]), a_min=0.0, a_max=20.0)
            )
        else:
            out[valid] = np.sign(out[valid]) * np.log1p(np.abs(out[valid]))
        return out
    raise ValueError(f"Unsupported transform: {transform_name}")


def transform_targets(
    y: np.ndarray,
    target_names: List[str],
    transform_types: Dict[str, str],
) -> np.ndarray:
    """Transform targets column-wise according to transform type."""
    y_t = y.astype(np.float32, copy=True)
    for i, target in enumerate(target_names):
        y_t[:, i] = _apply_target_transform(
            y_t[:, i],
            transform_name=transform_types.get(target, "identity"),
            inverse=False,
        )
    return y_t


def inverse_transform_targets(
    y: np.ndarray,
    target_names: List[str],
    transform_types: Dict[str, str],
) -> np.ndarray:
    """Inverse-transform targets column-wise according to transform type."""
    y_raw = y.astype(np.float32, copy=True)
    for i, target in enumerate(target_names):
        y_raw[:, i] = _apply_target_transform(
            y_raw[:, i],
            transform_name=transform_types.get(target, "identity"),
            inverse=True,
        )
    return y_raw


def _build_target_weights(
    y: np.ndarray,
    target_names: List[str],
    config: dict,
) -> np.ndarray:
    """Build per-target loss weights to reduce sparse-target underfitting."""
    mode = str(config.get("target_weight_mode", "inverse_frequency")).lower()
    manual = {
        k.lower(): float(v)
        for k, v in config.get("target_weights", {}).items()
    }

    if manual:
        weights = np.array([manual.get(t, 1.0) for t in target_names], dtype=np.float32)
    elif mode == "inverse_frequency":
        counts = np.sum(~np.isnan(y), axis=0).astype(np.float32)
        counts = np.clip(counts, a_min=1.0, a_max=None)
        weights = np.max(counts) / counts
    else:
        weights = np.ones(len(target_names), dtype=np.float32)

    if np.mean(weights) > 0:
        weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def _build_sample_weights(
    y: np.ndarray,
    target_weights: np.ndarray,
    mode: str = "rare_target_average",
) -> np.ndarray:
    """
    Build per-sample weights to upweight rows containing sparse targets.

    Each sample's weight is the average target weight across observed (non-NaN) targets.
    """
    mode = mode.lower()
    if mode in {"none", "uniform"}:
        return np.ones(len(y), dtype=np.float32)

    mask = ~np.isnan(y)
    sample_weights = np.ones(len(y), dtype=np.float32)
    for i in range(len(y)):
        valid_targets = mask[i]
        if valid_targets.any():
            sample_weights[i] = float(np.mean(target_weights[valid_targets]))
    if np.mean(sample_weights) > 0:
        sample_weights = sample_weights / np.mean(sample_weights)
    return sample_weights.astype(np.float32)


def _unpack_batch(
    batch: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpack dataloader batch tuple with optional sample weights."""
    if len(batch) == 3:
        features, targets, sample_weights = batch
        return features, targets, sample_weights
    features, targets = batch
    return features, targets, None


def _compute_training_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_names: List[str],
    transform_types: Dict[str, str],
    config: dict,
    target_weights_tensor: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute configured training loss with optional consistency regularization."""
    loss_name = str(config.get("loss_name", "weighted_huber")).lower()

    if loss_name in {"mse", "masked_mse"}:
        base_loss = masked_mse_loss(predictions, targets)
    else:
        base_loss = masked_weighted_huber_loss(
            predictions,
            targets,
            delta=float(config.get("huber_delta", 1.0)),
            target_weights=target_weights_tensor,
            sample_weights=sample_weights,
        )

    esp_weight = float(config.get("esp_consistency_weight", 0.0))
    if esp_weight > 0:
        penalty = esp_consistency_penalty(
            predictions=predictions,
            target_names=target_names,
            targets=targets,
            transform_types=transform_types,
        )
        base_loss = base_loss + esp_weight * penalty

    return base_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_names: List[str],
    transform_types: Dict[str, str],
    config: dict,
    target_weights_tensor: Optional[torch.Tensor] = None,
) -> float:
    """Train for one epoch, returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        features, targets, sample_weights = _unpack_batch(batch)
        features = features.to(device)
        targets = targets.to(device)
        sample_weights_t = sample_weights.to(device) if sample_weights is not None else None

        optimizer.zero_grad()
        predictions = model.forward_stacked(features)
        loss = _compute_training_loss(
            predictions=predictions,
            targets=targets,
            target_names=target_names,
            transform_types=transform_types,
            config=config,
            target_weights_tensor=target_weights_tensor,
            sample_weights=sample_weights_t,
        )

        if torch.isfinite(loss) and loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
            optimizer.step()

        loss_value = float(loss.item()) if torch.isfinite(loss) else 0.0
        total_loss += loss_value * features.size(0)
        total_samples += features.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_names: List[str],
    transform_types: Dict[str, str],
    config: dict,
    target_weights_tensor: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Validate model, returns loss and per-target metrics in raw target space."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            features, targets, sample_weights = _unpack_batch(batch)
            features = features.to(device)
            targets = targets.to(device)
            sample_weights_t = sample_weights.to(device) if sample_weights is not None else None

            predictions = model.forward_stacked(features)
            loss = _compute_training_loss(
                predictions=predictions,
                targets=targets,
                target_names=target_names,
                transform_types=transform_types,
                config=config,
                target_weights_tensor=target_weights_tensor,
                sample_weights=sample_weights_t,
            )

            loss_value = float(loss.item()) if torch.isfinite(loss) else 0.0
            total_loss += loss_value * features.size(0)
            total_samples += features.size(0)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if all_preds:
        all_preds_arr = np.vstack(all_preds)
        all_targets_arr = np.vstack(all_targets)
        preds_raw = inverse_transform_targets(all_preds_arr, target_names, transform_types)
        targets_raw = inverse_transform_targets(all_targets_arr, target_names, transform_types)
        metrics = compute_metrics(targets_raw, preds_raw, target_names)
    else:
        metrics = {}

    return total_loss / total_samples if total_samples > 0 else 0.0, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    target_names: List[str],
    transform_types: Dict[str, str],
    target_weights_tensor: Optional[torch.Tensor] = None,
    model_save_path: Optional[Path] = None,
) -> Tuple[nn.Module, Dict]:
    """Train a single model with early stopping."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("lr_factor", 0.5),
        patience=config.get("lr_patience", 20),
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
        "best_epoch": None,
        "best_val_loss": float("inf"),
        "best_val_metrics": {},
    }
    best_state = None

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            target_names=target_names,
            transform_types=transform_types,
            config=config,
            target_weights_tensor=target_weights_tensor,
        )
        val_loss, val_metrics = validate_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            target_names=target_names,
            transform_types=transform_types,
            config=config,
            target_weights_tensor=target_weights_tensor,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_loss
            history["best_val_metrics"] = val_metrics
            if model_save_path:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": config,
                        "target_names": target_names,
                        "target_transform_types": transform_types,
                        "val_loss": best_val_loss,
                        "epoch": epoch,
                    },
                    model_save_path,
                )
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            avg_r2 = (
                float(np.mean([m["r2"] for m in val_metrics.values()]))
                if val_metrics
                else float("nan")
            )
            print(
                f"  Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, avg_R2={avg_r2:.3f}"
            )

        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def _build_cv_splits(
    X: np.ndarray,
    config: dict,
    groups: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build CV splits based on configured strategy."""
    strategy = str(config.get("cv_strategy", "kfold")).lower()
    n_folds = int(config["n_folds"])
    random_seed = int(config.get("random_seed", 42))

    if strategy == "group_kfold" and groups is not None:
        splitter = GroupKFold(n_splits=n_folds)
        return list(splitter.split(X, groups=groups))

    if strategy == "group_kfold" and groups is None:
        print("Warning: cv_strategy=group_kfold but no groups provided. Falling back to KFold.")

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    return list(splitter.split(X))


def _train_specialist_models(
    X_scaled: np.ndarray,
    y_transformed: np.ndarray,
    target_names: List[str],
    transform_types: Dict[str, str],
    config: dict,
    output_dir: Path,
    device: torch.device,
    sample_weights: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Train optional target-specific specialist models for sparse targets."""
    specialist_targets = [t.lower() for t in config.get("specialist_targets", [])]
    if not specialist_targets:
        return {}

    specialists_dir = output_dir / "specialists"
    specialists_dir.mkdir(parents=True, exist_ok=True)
    result_paths: Dict[str, str] = {}

    specialist_epochs = int(config.get("specialist_epochs", min(150, config["epochs"])))
    specialist_patience = int(config.get("specialist_patience", 20))
    val_fraction = float(config.get("specialist_val_fraction", 0.2))
    random_seed = int(config.get("random_seed", 42))

    for target in specialist_targets:
        if target not in target_names:
            print(f"  Specialist skip: {target} not in targets")
            continue

        target_idx = target_names.index(target)
        valid_mask = ~np.isnan(y_transformed[:, target_idx])
        if valid_mask.sum() < 40:
            print(f"  Specialist skip: {target} has too few samples ({int(valid_mask.sum())})")
            continue

        X_t = X_scaled[valid_mask]
        y_t = y_transformed[valid_mask, target_idx : target_idx + 1]
        if sample_weights is None:
            sw_t = None
        else:
            sw_t = sample_weights[valid_mask]

        idx = np.arange(len(X_t))
        idx_train, idx_val = train_test_split(
            idx,
            test_size=val_fraction,
            random_state=random_seed,
            shuffle=True,
        )

        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(random_seed + target_idx * 1337)

        train_loader = DataLoader(
            SoilDataset(X_t[idx_train], y_t[idx_train], None if sw_t is None else sw_t[idx_train]),
            batch_size=config["batch_size"],
            shuffle=True,
            generator=train_loader_generator,
            drop_last=_should_drop_last_train_batch(
                n_samples=len(idx_train),
                batch_size=config["batch_size"],
            ),
        )
        val_loader = DataLoader(
            SoilDataset(X_t[idx_val], y_t[idx_val], None if sw_t is None else sw_t[idx_val]),
            batch_size=config["batch_size"],
            shuffle=False,
        )

        specialist_model = NationalSoilNet(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_res_blocks=config["num_res_blocks"],
            dropout=config["dropout"],
            target_names=[target],
        ).to(device)

        specialist_config = dict(config)
        specialist_config["epochs"] = specialist_epochs
        specialist_config["patience"] = specialist_patience
        specialist_config["esp_consistency_weight"] = 0.0  # single-head specialist

        specialist_model, _ = train_model(
            model=specialist_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=specialist_config,
            device=device,
            target_names=[target],
            transform_types={target: transform_types.get(target, "identity")},
            target_weights_tensor=torch.tensor([1.0], device=device),
        )

        out_path = specialists_dir / f"specialist_{target}.pth"
        torch.save(
            {
                "model_state_dict": specialist_model.state_dict(),
                "config": config,
                "target_names": [target],
                "target_transform_types": {target: transform_types.get(target, "identity")},
            },
            out_path,
        )
        result_paths[target] = str(out_path)
        print(f"  Specialist saved: {out_path}")

    return result_paths


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    target_names: List[str],
    config: dict,
    output_dir: Path,
    groups: Optional[np.ndarray] = None,
    domain_weights: Optional[np.ndarray] = None,
) -> Dict:
    """Train ensemble of models with CV. Saves model_*.pth + scaler.pkl."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_seed = int(config.get("random_seed", 42))
    set_global_seed(random_seed)

    transform_types = _resolve_transform_types(target_names, config)
    y_transformed = transform_targets(y, target_names, transform_types)
    target_weights = _build_target_weights(y_transformed, target_names, config)
    sample_weights = _build_sample_weights(
        y=y_transformed,
        target_weights=target_weights,
        mode=str(config.get("sample_weight_mode", "rare_target_average")),
    )
    if domain_weights is not None:
        domain_weights = np.asarray(domain_weights, dtype=np.float32)
        if len(domain_weights) != len(sample_weights):
            raise ValueError(
                f"domain_weights length {len(domain_weights)} != n_samples {len(sample_weights)}"
            )
        if np.mean(domain_weights) > 0:
            domain_weights = domain_weights / np.mean(domain_weights)
        sample_weights = sample_weights * domain_weights
        if np.mean(sample_weights) > 0:
            sample_weights = sample_weights / np.mean(sample_weights)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "target_transforms.json", "w", encoding="utf-8") as f:
        json.dump(transform_types, f, indent=2, sort_keys=True)

    ensemble_metrics = []
    cv_splits = _build_cv_splits(X_scaled, config, groups=groups)

    target_weights_tensor = torch.tensor(target_weights, dtype=torch.float32, device=device)

    for model_idx in range(config["ensemble_size"]):
        print(f"\n{'=' * 60}")
        print(f"Training Model {model_idx + 1}/{config['ensemble_size']}")
        print(f"{'=' * 60}")

        model_seed = random_seed + model_idx * 1000
        set_global_seed(model_seed)

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"\n  Fold {fold + 1}/{config['n_folds']}")

            fold_seed = model_seed + fold
            train_loader_generator = torch.Generator()
            train_loader_generator.manual_seed(fold_seed)

            train_loader = DataLoader(
                SoilDataset(X_scaled[train_idx], y_transformed[train_idx], sample_weights[train_idx]),
                batch_size=config["batch_size"],
                shuffle=True,
                generator=train_loader_generator,
                drop_last=_should_drop_last_train_batch(
                    n_samples=len(train_idx),
                    batch_size=config["batch_size"],
                ),
            )
            val_loader = DataLoader(
                SoilDataset(X_scaled[val_idx], y_transformed[val_idx], sample_weights[val_idx]),
                batch_size=config["batch_size"],
                shuffle=False,
            )

            model = NationalSoilNet(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                num_res_blocks=config["num_res_blocks"],
                dropout=config["dropout"],
                target_names=target_names,
            ).to(device)

            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                target_names=target_names,
                transform_types=transform_types,
                target_weights_tensor=target_weights_tensor,
            )

            best_metrics = history.get("best_val_metrics") or {}
            if not best_metrics and history["val_metrics"]:
                best_metrics = history["val_metrics"][-1]

            fold_metrics.append(best_metrics)
            for target, m in best_metrics.items():
                print(f"    {target}: R2={m['r2']:.3f}, RMSE={m['rmse']:.3f}")

        # Train final model on all data
        print(f"\n  Training final model {model_idx + 1} on all data...")
        full_loader_generator = torch.Generator()
        full_loader_generator.manual_seed(model_seed + 99999)
        full_loader = DataLoader(
            SoilDataset(X_scaled, y_transformed, sample_weights),
            batch_size=config["batch_size"],
            shuffle=True,
            generator=full_loader_generator,
            drop_last=_should_drop_last_train_batch(
                n_samples=len(X_scaled),
                batch_size=config["batch_size"],
            ),
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
        final_epochs = int(min(config.get("final_epochs", 200), config["epochs"]))
        for _ in range(final_epochs):
            for batch in full_loader:
                features, targets, batch_sample_weights = _unpack_batch(batch)
                features = features.to(device)
                targets = targets.to(device)
                batch_sample_weights_t = (
                    batch_sample_weights.to(device) if batch_sample_weights is not None else None
                )

                optimizer.zero_grad()
                predictions = final_model.forward_stacked(features)
                loss = _compute_training_loss(
                    predictions=predictions,
                    targets=targets,
                    target_names=target_names,
                    transform_types=transform_types,
                    config=config,
                    target_weights_tensor=target_weights_tensor,
                    sample_weights=batch_sample_weights_t,
                )
                if torch.isfinite(loss) and loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        final_model.parameters(), config.get("grad_clip", 1.0)
                    )
                    optimizer.step()

        model_path = output_dir / f"model_{model_idx + 1}.pth"
        save_config = dict(config)
        save_config["targets"] = target_names
        save_config["target_weights"] = {
            target_names[i]: float(target_weights[i]) for i in range(len(target_names))
        }
        save_config["target_transform_types"] = transform_types

        torch.save(
            {
                "model_state_dict": final_model.state_dict(),
                "config": save_config,
                "target_names": target_names,
                "target_transform_types": transform_types,
            },
            model_path,
        )
        print(f"  Saved: {model_path}")

        avg_metrics = {}
        for target in target_names:
            target_metrics = [fm[target] for fm in fold_metrics if target in fm]
            if target_metrics:
                avg_metrics[target] = {
                    "r2": float(np.mean([tm["r2"] for tm in target_metrics])),
                    "rmse": float(np.mean([tm["rmse"] for tm in target_metrics])),
                }
        ensemble_metrics.append(
            {
                "model": model_idx + 1,
                **{f"{t}_r2": avg_metrics[t]["r2"] for t in avg_metrics},
                **{f"{t}_rmse": avg_metrics[t]["rmse"] for t in avg_metrics},
            }
        )

    specialist_paths = _train_specialist_models(
        X_scaled=X_scaled,
        y_transformed=y_transformed,
        target_names=target_names,
        transform_types=transform_types,
        config=config,
        output_dir=output_dir,
        device=device,
        sample_weights=sample_weights,
    )

    pd.DataFrame(ensemble_metrics).to_csv(output_dir / "ensemble_metrics.csv", index=False)
    print("\nEnsemble Training Complete")
    return {
        "models_dir": str(output_dir),
        "scaler_path": str(output_dir / "scaler.pkl"),
        "target_transforms_path": str(output_dir / "target_transforms.json"),
        "specialists": specialist_paths,
    }


def load_training_data(
    csv_path: str,
    target_cols: List[str],
    feature_prefix: str = "auto",
    n_features: Optional[int] = None,
    group_by: str = "latlon",
    group_round: int = 4,
    return_metadata: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, List[str]],
    Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]],
]:
    """Load and validate training data from CSV."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    target_cols_lower = [c.lower() for c in target_cols]

    feature_cols = _find_feature_columns(df.columns.tolist(), feature_prefix)
    if not feature_cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}'")
    if n_features is not None and len(feature_cols) != int(n_features):
        raise ValueError(f"Expected {n_features} feature columns, found {len(feature_cols)}")

    valid_targets = [c for c in target_cols_lower if c in df.columns]
    if not valid_targets:
        raise ValueError(f"No target columns found. Expected any of: {target_cols}")

    print(f"  Found {len(feature_cols)} features, targets: {valid_targets}")
    X = df[feature_cols].values.astype(np.float32)
    y = df[valid_targets].values.astype(np.float32)

    valid_mask = ~np.isnan(X).any(axis=1)

    groups: Optional[np.ndarray] = None
    group_by = group_by.lower()
    if group_by in {"latlon", "lat_lon"}:
        lat_col = next((c for c in ("lat", "latitude") if c in df.columns), None)
        lon_col = next((c for c in ("lon", "longitude") if c in df.columns), None)
        if lat_col and lon_col:
            lat_group = df[lat_col].round(group_round).astype(str).values
            lon_group = df[lon_col].round(group_round).astype(str).values
            groups = np.array([f"{a}_{b}" for a, b in zip(lat_group, lon_group)], dtype=object)
    elif group_by in {"site", "site_id"} and "site_id" in df.columns:
        groups = df["site_id"].astype(str).values

    X, y = X[valid_mask], y[valid_mask]
    if groups is not None:
        groups = groups[valid_mask]

    print(f"  Samples with valid features: {len(X)}")

    if not return_metadata:
        return X, y, valid_targets

    metadata: Dict[str, object] = {
        "feature_cols": feature_cols,
        "groups": groups,
        "group_by": group_by,
        "group_round": group_round,
    }
    return X, y, valid_targets, metadata
