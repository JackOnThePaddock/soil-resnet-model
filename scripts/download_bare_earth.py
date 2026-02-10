#!/usr/bin/env python
"""Download GA Barest Earth (Sentinel-2) coverage from DEA WCS."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.bare_earth import (
    DEFAULT_COVERAGE_ID,
    DEA_OWS_BASE_URL,
    discover_barest_earth_products,
    download_barest_earth_geotiff,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GA Barest Earth Sentinel-2 GeoTIFF")
    parser.add_argument("--output", type=str, default=None, help="Output GeoTIFF path")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--bbox-from-csv", type=str, default=None, help="Infer bbox from a CSV containing lon/lat columns")
    parser.add_argument("--lon-col", type=str, default="lon", help="Longitude column for --bbox-from-csv")
    parser.add_argument("--lat-col", type=str, default="lat", help="Latitude column for --bbox-from-csv")
    parser.add_argument("--bbox-padding-deg", type=float, default=0.02, help="Padding around inferred bbox")
    parser.add_argument("--coverage-id", type=str, default=DEFAULT_COVERAGE_ID, help="DEA WCS coverage ID")
    parser.add_argument("--base-url", type=str, default=DEA_OWS_BASE_URL, help="DEA OWS base URL")
    parser.add_argument(
        "--bands",
        type=str,
        default="red,green,blue,red_edge_1,red_edge_2,red_edge_3,nir,nir_2,swir1,swir2",
        help="Comma-separated band names to request",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Print discovered WMS/WCS bare-earth identifiers before download",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Print discovered identifiers and exit without downloading",
    )
    args = parser.parse_args()

    if args.discover or args.discover_only:
        products = discover_barest_earth_products(args.base_url)
        print(json.dumps(products, indent=2))
    if args.discover_only:
        return

    if not args.output:
        parser.error("--output is required unless --discover-only is set")
    if args.bbox is None and not args.bbox_from_csv:
        parser.error("Provide either --bbox or --bbox-from-csv")

    if args.bbox_from_csv:
        df = pd.read_csv(args.bbox_from_csv)
        lon_col = args.lon_col
        lat_col = args.lat_col
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(f"Expected columns '{lon_col}' and '{lat_col}' in {args.bbox_from_csv}")
        west = float(df[lon_col].min()) - args.bbox_padding_deg
        east = float(df[lon_col].max()) + args.bbox_padding_deg
        south = float(df[lat_col].min()) - args.bbox_padding_deg
        north = float(df[lat_col].max()) + args.bbox_padding_deg
        bbox = (west, south, east, north)
    else:
        bbox = tuple(args.bbox)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    out_path = download_barest_earth_geotiff(
        bbox_lonlat=bbox,
        output_path=Path(args.output),
        base_url=args.base_url,
        coverage_id=args.coverage_id,
        bands=bands,
    )
    print(f"Downloaded: {out_path}")


if __name__ == "__main__":
    main()
