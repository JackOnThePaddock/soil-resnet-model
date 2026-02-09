"""Side-by-side model comparison report generation."""

from pathlib import Path
from typing import Dict, List

import pandas as pd


def generate_comparison_table(
    metrics_files: Dict[str, str],
    output_path: str,
) -> pd.DataFrame:
    """Combine metrics from multiple model runs into a comparison table."""
    all_rows = []
    for model_name, path in metrics_files.items():
        path = Path(path)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = model_name
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Comparison table saved: {output_path}")
    return combined


def format_results_markdown(metrics_csv: str) -> str:
    """Format a metrics CSV into a markdown table."""
    df = pd.read_csv(metrics_csv)
    lines = ["| Model | Target | R² | RMSE | MAE | N |", "|-------|--------|-----|------|-----|---|"]
    for _, row in df.iterrows():
        model = row.get("model", "")
        target = row.get("target", "")
        r2 = f"{row.get('r2', 0):.3f}" if pd.notna(row.get("r2")) else "N/A"
        rmse = f"{row.get('rmse', 0):.3f}" if pd.notna(row.get("rmse")) else "N/A"
        mae = f"{row.get('mae', 0):.3f}" if pd.notna(row.get("mae")) else "N/A"
        n = str(int(row.get("n", 0))) if pd.notna(row.get("n")) else ""
        lines.append(f"| {model} | {target} | {r2} | {rmse} | {mae} | {n} |")
    return "\n".join(lines)
