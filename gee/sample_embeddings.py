#!/usr/bin/env python
"""Sample AlphaEarth satellite embeddings at soil sample locations using Google Earth Engine."""

import argparse
import sys
from pathlib import Path

import ee
import pandas as pd


def initialize_gee(project: str):
    """Initialize Google Earth Engine."""
    ee.Authenticate()
    ee.Initialize(project=project)


def create_feature_collection(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"):
    """Create ee.FeatureCollection from DataFrame with lat/lon columns."""
    features = []
    for idx, row in df.iterrows():
        point = ee.Geometry.Point([row[lon_col], row[lat_col]])
        feat = ee.Feature(point, {"index": idx})
        features.append(feat)
    return ee.FeatureCollection(features)


def sample_embeddings(fc, start_year: int = 2019, end_year: int = 2024):
    """Sample AlphaEarth embeddings at feature collection points."""
    collection = ee.ImageCollection("projects/sat-io/open-datasets/STAC/alphaearth-embeddings-sentinel2")
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    filtered = collection.filterDate(start_date, end_date)
    median = filtered.median()

    sampled = median.sampleRegions(
        collection=fc,
        scale=10,
        geometries=True,
    )
    return sampled


def batch_sample(df, batch_size: int = 500, start_year: int = 2019, end_year: int = 2024):
    """Sample in batches to avoid GEE limits."""
    all_results = []
    n_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(df))
        batch_df = df.iloc[start:end].copy()
        print(f"  Batch {i+1}/{n_batches}: rows {start}-{end-1}")

        fc = create_feature_collection(batch_df)
        sampled = sample_embeddings(fc, start_year, end_year)

        results = sampled.getInfo()
        for feat in results["features"]:
            props = feat["properties"]
            all_results.append(props)

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description="Sample AlphaEarth embeddings at soil locations")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV with lat/lon columns")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--project", type=str, required=True, help="GEE project ID")
    parser.add_argument("--start-year", type=int, default=2019, help="Start year for composite")
    parser.add_argument("--end-year", type=int, default=2024, help="End year for composite")
    parser.add_argument("--batch-size", type=int, default=500, help="Points per batch")
    parser.add_argument("--lat-col", type=str, default="latitude", help="Latitude column name")
    parser.add_argument("--lon-col", type=str, default="longitude", help="Longitude column name")

    args = parser.parse_args()

    print(f"Initializing GEE with project: {args.project}")
    initialize_gee(args.project)

    print(f"Loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  {len(df)} samples")

    print(f"Sampling embeddings ({args.start_year}-{args.end_year})...")
    results = batch_sample(
        df,
        batch_size=args.batch_size,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved {len(results)} samples to {output_path}")


if __name__ == "__main__":
    main()
