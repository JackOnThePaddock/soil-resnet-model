#!/usr/bin/env python
"""Build fused training features from AlphaEarth + Bare Earth + Spectral-GPT + management."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.fusion import build_fused_feature_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fused feature table for soil model training")
    parser.add_argument("--soil-csv", type=str, required=True, help="Input soil points CSV with lat/lon + AlphaEarth features")
    parser.add_argument("--bare-earth-raster", type=str, required=True, help="Bare Earth GeoTIFF")
    parser.add_argument("--output-csv", type=str, required=True, help="Output fused CSV")
    parser.add_argument("--output-meta", type=str, default=None, help="Optional metadata JSON output")
    parser.add_argument("--lon-col", type=str, default="lon", help="Longitude column in soil CSV")
    parser.add_argument("--lat-col", type=str, default="lat", help="Latitude column in soil CSV")
    parser.add_argument("--obs-date-col", type=str, default="date", help="Observation date column")
    parser.add_argument("--image-date", type=str, default="2020-09-30", help="Bare-earth image reference date")
    parser.add_argument(
        "--spectral-method",
        type=str,
        default="spectral_gpt",
        choices=["spectral_gpt", "pca"],
        help="Spectral embedding method for bare-earth bands",
    )
    parser.add_argument("--spectral-dim", type=int, default=16, help="Spectral embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--applications-csv",
        type=str,
        default=None,
        help="Optional management applications CSV (lime/gypsum)",
    )
    args = parser.parse_args()

    soil_df = pd.read_csv(args.soil_csv)
    applications_df = (
        pd.read_csv(args.applications_csv)
        if args.applications_csv and Path(args.applications_csv).exists()
        else None
    )

    fused_df, metadata = build_fused_feature_table(
        soil_df=soil_df,
        bare_earth_raster=Path(args.bare_earth_raster),
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        spectral_method=args.spectral_method,
        spectral_dim=args.spectral_dim,
        applications_df=applications_df,
        obs_date_col=args.obs_date_col,
        image_date=args.image_date,
        random_seed=args.seed,
    )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(out_csv, index=False)
    print(f"Saved fused features: {out_csv} ({len(fused_df)} rows, {len(fused_df.columns)} cols)")
    print(f"Feature counts: {json.dumps({k: len(v) for k, v in metadata.items()}, indent=2)}")

    if args.output_meta:
        out_meta = Path(args.output_meta)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        print(f"Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
