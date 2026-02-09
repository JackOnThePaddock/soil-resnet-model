#!/usr/bin/env python
"""Run end-to-end farm prediction pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.applications.farm_pipeline import run_farm_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run farm soil predictions")
    parser.add_argument("--shp", type=str, required=True, help="Paddock boundaries shapefile")
    parser.add_argument("--models", type=str, default="models/resnet_ensemble", help="Models directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--project", type=str, default=None, help="GEE project ID")
    parser.add_argument("--skip-download", action="store_true", help="Skip GEE download")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction")
    parser.add_argument("--block-size", type=int, default=512, help="Raster block size")

    args = parser.parse_args()
    run_farm_pipeline(
        boundaries_shp=args.shp, models_dir=args.models, output_dir=args.output,
        project=args.project, skip_download=args.skip_download,
        skip_predict=args.skip_predict, block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
