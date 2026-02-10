#!/usr/bin/env python
"""Train a ResNet ensemble for soil property prediction."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.domain_adaptation import compute_covariate_shift_weights
from src.training.train_resnet import train_ensemble, load_training_data


def main():
    parser = argparse.ArgumentParser(description="Train ResNet ensemble")
    parser.add_argument("--data", type=str, required=True, help="Training CSV file")
    parser.add_argument("--config", type=str, default="configs/resnet.yaml", help="Config YAML")
    parser.add_argument("--output", type=str, default="models/resnet_ensemble", help="Output dir")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Override ensemble size")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument(
        "--cv-strategy",
        type=str,
        default=None,
        choices=["kfold", "group_kfold"],
        help="Override CV strategy",
    )
    parser.add_argument(
        "--reference-data",
        type=str,
        default=None,
        help="Optional reference-domain CSV for covariate-shift weighting",
    )

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    config = {
        "input_dim": cfg["model"]["input_dim"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "num_res_blocks": cfg["model"]["num_blocks"],
        "dropout": cfg["model"]["dropout"],
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
        "batch_size": training_cfg["batch_size"],
        "epochs": args.epochs or training_cfg["epochs"],
        "patience": training_cfg["patience"],
        "grad_clip": training_cfg["grad_clip"],
        "ensemble_size": args.ensemble_size or training_cfg["ensemble_size"],
        "n_folds": training_cfg["n_splits"],
        "lr_patience": training_cfg.get("lr_patience", 20),
        "lr_factor": training_cfg.get("lr_factor", 0.5),
        "random_seed": args.seed if args.seed is not None else training_cfg.get("random_seed", 42),
        "cv_strategy": args.cv_strategy or training_cfg.get("cv_strategy", "kfold"),
        "final_epochs": training_cfg.get("final_epochs", 200),
        "loss_name": training_cfg.get("loss_name", "weighted_huber"),
        "huber_delta": training_cfg.get("huber_delta", 1.0),
        "esp_consistency_weight": training_cfg.get("esp_consistency_weight", 0.0),
        "target_weight_mode": training_cfg.get("target_weight_mode", "inverse_frequency"),
        "sample_weight_mode": training_cfg.get("sample_weight_mode", "rare_target_average"),
        "auto_target_transforms": training_cfg.get("auto_target_transforms", True),
        "target_transforms": training_cfg.get("target_transforms", {}),
        "specialist_targets": training_cfg.get("specialist_targets", []),
        "specialist_epochs": training_cfg.get("specialist_epochs", 150),
        "specialist_patience": training_cfg.get("specialist_patience", 20),
        "specialist_val_fraction": training_cfg.get("specialist_val_fraction", 0.2),
        "specialist_blend_weight": training_cfg.get("specialist_blend_weight", 0.4),
    }

    targets = [t.lower() for t in cfg["targets"]]
    X, y, valid_targets, metadata = load_training_data(
        args.data,
        targets,
        feature_prefix=data_cfg.get("feature_prefix", "auto"),
        n_features=data_cfg.get("n_features"),
        group_by=data_cfg.get("group_by", "latlon"),
        group_round=int(data_cfg.get("group_round", 4)),
        return_metadata=True,
    )
    config["targets"] = valid_targets
    config["feature_cols"] = metadata["feature_cols"]
    if int(config["input_dim"]) != int(X.shape[1]):
        print(
            f"Input dim override: config input_dim={config['input_dim']} -> detected {X.shape[1]}"
        )
        config["input_dim"] = int(X.shape[1])
    reference_data = args.reference_data or data_cfg.get("reference_data")

    domain_weights = None
    if reference_data:
        ref_path = Path(reference_data)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference data not found: {ref_path}")
        ref_df = pd.read_csv(ref_path)
        ref_df.columns = ref_df.columns.str.lower()
        missing = [c for c in metadata["feature_cols"] if c not in ref_df.columns]
        if missing:
            raise ValueError(
                f"Reference data missing required feature columns: {missing[:10]}"
            )
        X_ref = ref_df[metadata["feature_cols"]].values.astype(np.float32)
        valid_ref = ~np.isnan(X_ref).any(axis=1)
        X_ref = X_ref[valid_ref]
        if len(X_ref) < 20:
            raise ValueError(f"Reference dataset too small after filtering: {len(X_ref)} rows")
        domain_weights = compute_covariate_shift_weights(
            train_X=X,
            reference_X=X_ref,
            random_state=config["random_seed"],
        )
        print(
            "Computed covariate-shift weights: "
            f"min={domain_weights.min():.3f}, max={domain_weights.max():.3f}, "
            f"mean={domain_weights.mean():.3f}"
        )

    print(f"\nTraining ensemble of {config['ensemble_size']} models")
    train_ensemble(
        X=X,
        y=y,
        target_names=valid_targets,
        config=config,
        output_dir=Path(args.output),
        groups=metadata.get("groups"),
        domain_weights=domain_weights,
    )


if __name__ == "__main__":
    main()
