"""
Main Orchestration Script for Australian Soil Data Extraction

This script coordinates the data fetching, cleaning, and storage pipeline.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATABASE_PATH,
    PROPERTY_GROUPS,
    DEFAULT_UPPER_DEPTH,
    DEFAULT_LOWER_DEPTH,
    MIN_OBSERVATION_DATE
)
from soil_data_fetcher import TERNAPIClient, test_tern_api, fetch_soil_data_for_training
from data_cleaner import clean_soil_data, merge_property_data
from db_handler import init_database, insert_soil_data, get_data_summary, export_to_csv


def run_extraction_pipeline(
    property_groups: list = PROPERTY_GROUPS,
    upper_depth: int = DEFAULT_UPPER_DEPTH,
    lower_depth: int = DEFAULT_LOWER_DEPTH,
    min_date: str = MIN_OBSERVATION_DATE,
    db_path: str = DATABASE_PATH
) -> bool:
    """
    Run the complete soil data extraction pipeline.

    Args:
        property_groups: List of property groups to fetch
        upper_depth: Minimum upper depth (cm)
        lower_depth: Maximum lower depth (cm)
        min_date: Minimum observation date (YYYY-MM-DD)
        db_path: Path to SQLite database

    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("Australian Soil Data Extraction Pipeline")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target depth: {upper_depth}-{lower_depth} cm")
    print(f"Min date: {min_date}")
    print(f"Database: {db_path}")
    print("-" * 60)

    # Step 1: Initialize database
    print("\n[1/5] Initializing database...")
    init_database(db_path)

    # Step 2: Test API connectivity
    print("\n[2/5] Testing TERN API connectivity...")
    if not test_tern_api():
        print("\n" + "=" * 60)
        print("TERN API UNAVAILABLE - Using ANSIS Fallback")
        print("=" * 60)
        print("""
The TERN Soil Data Federator has been deprecated and replaced by ANSIS.

To get your soil data, follow these steps:

1. VISIT: https://portal.ansis.net/

2. QUERY for your data:
   - Select property types: pH, CEC, Organic Carbon, Exchangeable Cations
   - Set depth filter: 0-15 cm
   - Set date filter: 2017 onwards (if available)

3. DOWNLOAD as JSON

4. RUN THIS COMMAND:
   python main.py --json path/to/downloaded_data.json

Alternative: Use the ANSIS API directly (if available with your credentials)
   python ansis_fallback.py your_data.json
""")
        return False

    print("API is available!")

    # Step 3: Fetch data from API
    print(f"\n[3/5] Fetching data for {len(property_groups)} property groups...")
    raw_dataframes = fetch_soil_data_for_training(
        property_groups=property_groups,
        upper_depth=upper_depth,
        lower_depth=lower_depth
    )

    if not raw_dataframes or all(df.empty for df in raw_dataframes):
        print("WARNING: No data retrieved from API.")
        print("This may be due to:")
        print("  - Demo credentials (limited to 5 records)")
        print("  - API deprecation")
        print("  - No data matching filters")
        return False

    total_raw = sum(len(df) for df in raw_dataframes)
    print(f"Total raw records retrieved: {total_raw}")

    # Step 4: Clean and merge data
    print("\n[4/5] Cleaning and processing data...")

    cleaned_dataframes = []
    for i, df in enumerate(raw_dataframes):
        if not df.empty:
            print(f"  Processing property group {i+1}...")
            df_clean = clean_soil_data(df)
            cleaned_dataframes.append(df_clean)
            print(f"    Cleaned: {len(df)} -> {len(df_clean)} records")

    # Merge all property data
    if cleaned_dataframes:
        merged_df = merge_property_data(cleaned_dataframes)
        print(f"\nMerged dataset: {len(merged_df)} records")
    else:
        print("WARNING: No cleaned data to merge.")
        return False

    # Step 5: Insert into database
    print("\n[5/5] Inserting data into database...")
    records_inserted = insert_soil_data(merged_df, db_path)
    print(f"Records inserted: {records_inserted}")

    # Print summary
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)

    summary = get_data_summary(db_path)
    print(f"\nDatabase Summary:")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Date range: {summary['date_range']['min']} to {summary['date_range']['max']}")
    print(f"\nValue ranges:")
    for prop, ranges in summary['value_ranges'].items():
        print(f"  {prop}: {ranges['min']} - {ranges['max']}")
    print(f"\nNull counts:")
    for prop, count in summary['null_counts'].items():
        print(f"  {prop}: {count}")
    print(f"\nData sources:")
    for source, count in summary['sources'].items():
        print(f"  {source}: {count}")

    return True


def process_json_file(json_path: str, db_path: str = DATABASE_PATH) -> bool:
    """
    Process a JSON file downloaded from ANSIS portal.

    Use this function when the TERN API is unavailable.

    Args:
        json_path: Path to JSON file from ANSIS
        db_path: Path to SQLite database

    Returns:
        True if successful
    """
    import json

    print(f"Processing JSON file: {json_path}")

    # Initialize database
    init_database(db_path)

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # Handle various JSON structures
        if 'features' in data:  # GeoJSON
            df = pd.DataFrame([f['properties'] for f in data['features']])
        elif 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame([data])
    else:
        print("ERROR: Unsupported JSON structure")
        return False

    print(f"Loaded {len(df)} records from JSON")

    # Clean data
    df_clean = clean_soil_data(df)
    print(f"After cleaning: {len(df_clean)} records")

    # Insert into database
    records_inserted = insert_soil_data(df_clean, db_path)
    print(f"Records inserted: {records_inserted}")

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Australian soil data for AI training"
    )
    parser.add_argument(
        '--json',
        type=str,
        help='Process a JSON file from ANSIS instead of using API'
    )
    parser.add_argument(
        '--export-csv',
        type=str,
        help='Export database to CSV file'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=DATABASE_PATH,
        help=f'Database path (default: {DATABASE_PATH})'
    )
    parser.add_argument(
        '--depth-min',
        type=int,
        default=DEFAULT_UPPER_DEPTH,
        help=f'Minimum depth in cm (default: {DEFAULT_UPPER_DEPTH})'
    )
    parser.add_argument(
        '--depth-max',
        type=int,
        default=DEFAULT_LOWER_DEPTH,
        help=f'Maximum depth in cm (default: {DEFAULT_LOWER_DEPTH})'
    )
    parser.add_argument(
        '--min-date',
        type=str,
        default=MIN_OBSERVATION_DATE,
        help=f'Minimum observation date YYYY-MM-DD (default: {MIN_OBSERVATION_DATE})'
    )

    args = parser.parse_args()

    if args.export_csv:
        print(f"Exporting database to {args.export_csv}...")
        export_to_csv(args.export_csv, args.db)
        return

    if args.json:
        success = process_json_file(args.json, args.db)
    else:
        success = run_extraction_pipeline(
            upper_depth=args.depth_min,
            lower_depth=args.depth_max,
            min_date=args.min_date,
            db_path=args.db
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
