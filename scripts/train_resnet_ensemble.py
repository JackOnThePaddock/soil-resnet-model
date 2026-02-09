#!/usr/bin/env python
"""Train a ResNet ensemble for soil property prediction."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_resnet import train_ensemble, load_training_data


def main():
    parser = argparse.ArgumentParser(description="Train ResNet ensemble")
    parser.add_argument("--data", type=str, required=True, help="Training CSV file")
    parser.add_argument("--config", type=str, default="configs/resnet.yaml", help="Config YAML")
    parser.add_argument("--output", type=str, default="models/resnet_ensemble", help="Output dir")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Override ensemble size")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    config = {
        "input_dim": cfg["model"]["input_dim"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "num_res_blocks": cfg["model"]["num_blocks"],
        "dropout": cfg["model"]["dropout"],
        "learning_rate": cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "batch_size": cfg["training"]["batch_size"],
        "epochs": args.epochs or cfg["training"]["epochs"],
        "patience": cfg["training"]["patience"],
        "grad_clip": cfg["training"]["grad_clip"],
        "ensemble_size": args.ensemble_size or cfg["training"]["ensemble_size"],
        "n_folds": cfg["training"]["n_splits"],
    }

    targets = [t.lower() for t in cfg["targets"]]
    X, y, valid_targets = load_training_data(args.data, targets)

    print(f"\nTraining ensemble of {config['ensemble_size']} models")
    train_ensemble(X, y, valid_targets, config, Path(args.output))


if __name__ == "__main__":
    main()
