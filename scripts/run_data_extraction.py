#!/usr/bin/env python
"""Extract Australian soil data from TERN/ANSIS into SQLite database."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.extract_pipeline import run_extraction_pipeline
from src.data.db_handler import export_to_csv


def main():
    parser = argparse.ArgumentParser(description="Extract Australian soil data for training")
    parser.add_argument("--db", type=str, default="data/raw/soil_data.db", help="Database path")
    parser.add_argument("--export-csv", type=str, help="Export database to CSV")
    parser.add_argument("--depth-min", type=int, default=0, help="Min depth (cm)")
    parser.add_argument("--depth-max", type=int, default=15, help="Max depth (cm)")
    parser.add_argument("--min-date", type=str, default="2017-01-01", help="Min observation date")

    args = parser.parse_args()

    if args.export_csv:
        export_to_csv(args.export_csv, args.db)
        return

    success = run_extraction_pipeline(
        upper_depth=args.depth_min, lower_depth=args.depth_max,
        min_date=args.min_date, db_path=args.db,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
