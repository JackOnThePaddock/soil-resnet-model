#!/usr/bin/env python
"""Evaluate trained models and generate comparison report."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.model_comparison import generate_comparison_table, format_results_markdown


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics", help="Metrics directory")
    parser.add_argument("--output", type=str, default="results/metrics/model_comparison.csv", help="Output CSV")

    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir)

    # Find all metrics CSVs
    metrics_files = {}
    for csv_file in sorted(metrics_dir.glob("*.csv")):
        if csv_file.name != Path(args.output).name:
            metrics_files[csv_file.stem] = str(csv_file)

    if not metrics_files:
        print(f"No metrics files found in {metrics_dir}")
        sys.exit(1)

    print(f"Found {len(metrics_files)} metrics files")
    combined = generate_comparison_table(metrics_files, args.output)

    if not combined.empty:
        print(f"\n{format_results_markdown(args.output)}")


if __name__ == "__main__":
    main()
