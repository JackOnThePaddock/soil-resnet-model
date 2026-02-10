#!/usr/bin/env python
"""Train baseline models (SVR, RF, Extra Trees, Cubist, GPR) for comparison."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_baselines import train_all_baselines


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--data", type=str, required=True, help="Training CSV file")
    parser.add_argument("--config", type=str, default="configs/baselines.yaml", help="Config YAML")
    parser.add_argument("--output", type=str, default="results/metrics", help="Output dir for metrics")

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    targets = cfg.get("targets", ["pH", "CEC", "ESP", "SOC", "Ca", "Mg", "Na"])
    data_cfg = cfg.get("data", {})
    train_all_baselines(
        args.data,
        targets,
        args.output,
        feature_prefix=data_cfg.get("feature_prefix", "auto"),
        n_features=data_cfg.get("n_features", 64),
    )


if __name__ == "__main__":
    main()
