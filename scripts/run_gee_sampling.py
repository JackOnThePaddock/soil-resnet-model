#!/usr/bin/env python
"""Sample AlphaEarth embeddings from Google Earth Engine for paddock boundaries."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.gee_sampler import sample_embeddings_for_boundaries


def main():
    parser = argparse.ArgumentParser(description="Download AlphaEarth embeddings from GEE")
    parser.add_argument("--shp", type=str, required=True, help="Paddock boundaries shapefile")
    parser.add_argument("--output", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--project", type=str, default=None, help="GEE project ID")
    parser.add_argument("--years", type=int, default=5, help="Number of years for median (default: 5)")

    args = parser.parse_args()
    sample_embeddings_for_boundaries(args.shp, args.output, n_years=args.years, project=args.project)


if __name__ == "__main__":
    main()
