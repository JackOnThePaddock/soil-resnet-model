"""Orchestration pipeline for soil data extraction."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.tern_client import TERNAPIClient, test_tern_api, fetch_soil_data_for_training
from src.data.data_cleaner import clean_soil_data, merge_property_data
from src.data.db_handler import init_database, insert_soil_data, get_data_summary, export_to_csv


DEFAULT_PROPERTY_GROUPS = ["Soil pH", "Exchangeable Cations", "Organic Carbon", "CEC"]


def run_extraction_pipeline(
    property_groups: Optional[List[str]] = None,
    upper_depth: int = 0, lower_depth: int = 15,
    min_date: str = "2017-01-01", db_path: str = "soil_data.db",
) -> bool:
    """Run the complete soil data extraction pipeline."""
    if property_groups is None:
        property_groups = DEFAULT_PROPERTY_GROUPS

    print("=" * 60)
    print("Australian Soil Data Extraction Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Depth: {upper_depth}-{lower_depth} cm | Min date: {min_date}")
    print("=" * 60)

    init_database(db_path)

    if not test_tern_api():
        print("TERN API unavailable. Use ANSIS portal: https://portal.ansis.net/")
        return False

    raw_dfs = fetch_soil_data_for_training(property_groups, upper_depth, lower_depth)
    if not raw_dfs or all(df.empty for df in raw_dfs):
        print("No data retrieved.")
        return False

    cleaned = [clean_soil_data(df, min_date) for df in raw_dfs if not df.empty]
    if not cleaned:
        return False

    merged = merge_property_data(cleaned)
    print(f"Merged dataset: {len(merged)} records")

    records = insert_soil_data(merged, db_path)
    print(f"Inserted {records} records")

    summary = get_data_summary(db_path)
    print(f"Total in DB: {summary['total_records']}")
    return True
