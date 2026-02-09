#!/usr/bin/env python
"""Generate RESULTS.md from metrics CSVs."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    metrics_dir = Path("results/metrics")
    output = Path("RESULTS.md")

    lines = [
        "# Results",
        "",
        "## ResNet Ensemble (NationalSoilNet)",
        "",
    ]

    # Ensemble metrics
    ensemble_csv = metrics_dir / "holdout_metrics_ensemble.csv"
    if ensemble_csv.exists():
        df = pd.read_csv(ensemble_csv)
        lines.append("### Holdout Metrics (Ensemble Mean)")
        lines.append("")
        lines.append("| Target | N Test | R² | RMSE (raw) | MAE (raw) |")
        lines.append("|--------|--------|-----|------------|-----------|")
        for _, row in df.iterrows():
            lines.append(
                f"| {row['target'].upper()} | {row['n_test']} | "
                f"{row['r2_raw']:.3f} | {row['rmse_raw']:.3f} | {row['mae_raw']:.3f} |"
            )
        lines.append("")

    # Per-model metrics
    per_model_csv = metrics_dir / "holdout_metrics_per_model.csv"
    if per_model_csv.exists():
        df = pd.read_csv(per_model_csv)
        lines.append("### Per-Model Variance")
        lines.append("")
        lines.append("| Model | Target | R² | RMSE |")
        lines.append("|-------|--------|-----|------|")
        for _, row in df.iterrows():
            lines.append(f"| {row['model']} | {row['target']} | {row['r2_norm']:.3f} | {row['rmse_norm']:.3f} |")
        lines.append("")

    # Baseline comparison
    baseline_csv = metrics_dir / "baseline_comparison.csv"
    if baseline_csv.exists():
        df = pd.read_csv(baseline_csv)
        lines.append("## Baseline Model Comparison")
        lines.append("")
        lines.append("| Model | Target | R² | RMSE | MAE |")
        lines.append("|-------|--------|-----|------|-----|")
        for _, row in df.iterrows():
            r2 = f"{row['r2']:.3f}" if pd.notna(row.get("r2")) else "N/A"
            rmse = f"{row['rmse']:.3f}" if pd.notna(row.get("rmse")) else "N/A"
            mae = f"{row['mae']:.3f}" if pd.notna(row.get("mae")) else "N/A"
            lines.append(f"| {row.get('model', '')} | {row['target']} | {r2} | {rmse} | {mae} |")
        lines.append("")

    output.write_text("\n".join(lines))
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
