"""
SQLite Database Handler for Soil Data Storage
"""

import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd

from config import DATABASE_PATH


def get_connection(db_path: str = DATABASE_PATH) -> sqlite3.Connection:
    """Get a database connection."""
    return sqlite3.connect(db_path)


def init_database(db_path: str = DATABASE_PATH) -> None:
    """Initialize the database with the required schema."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Create main observations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS soil_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT,
            dataset_source TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            observation_date DATE,
            upper_depth_cm INTEGER DEFAULT 0,
            lower_depth_cm INTEGER DEFAULT 15,

            -- Soil properties (standardized units)
            ph_cacl2 REAL,           -- pH in CaCl2
            cec_cmol_kg REAL,        -- Cation Exchange Capacity (cmol(+)/kg)
            esp_percent REAL,        -- Exchangeable Sodium Percentage (%)
            soc_percent REAL,        -- Soil Organic Carbon (%)
            ca_cmol_kg REAL,         -- Exchangeable Calcium (cmol(+)/kg)
            mg_cmol_kg REAL,         -- Exchangeable Magnesium (cmol(+)/kg)
            na_cmol_kg REAL,         -- Exchangeable Sodium (cmol(+)/kg)

            -- Metadata
            method_ph TEXT,          -- Method code (e.g., "4A1")
            method_cec TEXT,
            method_cations TEXT,     -- Method for Ca/Mg/Na (e.g., "15A1", "15C1")
            data_quality TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for common queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_location
        ON soil_observations(latitude, longitude)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_date
        ON soil_observations(observation_date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_depth
        ON soil_observations(upper_depth_cm, lower_depth_cm)
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at: {db_path}")


def insert_soil_data(df: pd.DataFrame, db_path: str = DATABASE_PATH) -> int:
    """
    Insert soil observation data into the database.

    Args:
        df: DataFrame with soil observation data
        db_path: Path to SQLite database

    Returns:
        Number of records inserted
    """
    conn = get_connection(db_path)

    # Map DataFrame columns to database columns
    column_mapping = {
        'site_id': 'site_id',
        'dataset_source': 'dataset_source',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'observation_date': 'observation_date',
        'upper_depth_cm': 'upper_depth_cm',
        'lower_depth_cm': 'lower_depth_cm',
        'ph_cacl2': 'ph_cacl2',
        'cec_cmol_kg': 'cec_cmol_kg',
        'esp_percent': 'esp_percent',
        'soc_percent': 'soc_percent',
        'ca_cmol_kg': 'ca_cmol_kg',
        'mg_cmol_kg': 'mg_cmol_kg',
        'na_cmol_kg': 'na_cmol_kg',
        'method_ph': 'method_ph',
        'method_cec': 'method_cec',
        'method_cations': 'method_cations',
        'data_quality': 'data_quality'
    }

    # Rename columns if needed
    df_to_insert = df.rename(columns=column_mapping)

    # Only keep columns that exist in the database
    valid_columns = list(column_mapping.values())
    df_to_insert = df_to_insert[[c for c in df_to_insert.columns if c in valid_columns]]

    # Insert data in chunks to avoid SQLite variable limits
    chunk_size = 100  # Small chunks to avoid "too many SQL variables" error
    records_inserted = 0

    for i in range(0, len(df_to_insert), chunk_size):
        chunk = df_to_insert.iloc[i:i + chunk_size]
        try:
            chunk.to_sql(
                'soil_observations',
                conn,
                if_exists='append',
                index=False
            )
            records_inserted += len(chunk)
        except sqlite3.IntegrityError as e:
            # Handle duplicate records - insert one by one
            for _, row in chunk.iterrows():
                try:
                    row.to_frame().T.to_sql(
                        'soil_observations',
                        conn,
                        if_exists='append',
                        index=False
                    )
                    records_inserted += 1
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates

        if (i + chunk_size) % 1000 == 0:
            print(f"  Inserted {min(i + chunk_size, len(df_to_insert))}/{len(df_to_insert)} records...")

    conn.commit()
    conn.close()

    return records_inserted


def get_observation_count(db_path: str = DATABASE_PATH) -> int:
    """Get total number of observations in database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM soil_observations")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_data_summary(db_path: str = DATABASE_PATH) -> dict:
    """Get summary statistics of the soil data."""
    conn = get_connection(db_path)

    summary = {}

    # Total records
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM soil_observations")
    summary['total_records'] = cursor.fetchone()[0]

    # Date range
    cursor.execute("""
        SELECT MIN(observation_date), MAX(observation_date)
        FROM soil_observations
    """)
    date_range = cursor.fetchone()
    summary['date_range'] = {'min': date_range[0], 'max': date_range[1]}

    # Property value ranges
    cursor.execute("""
        SELECT
            MIN(ph_cacl2), MAX(ph_cacl2),
            MIN(cec_cmol_kg), MAX(cec_cmol_kg),
            MIN(soc_percent), MAX(soc_percent),
            MIN(esp_percent), MAX(esp_percent)
        FROM soil_observations
    """)
    ranges = cursor.fetchone()
    summary['value_ranges'] = {
        'ph_cacl2': {'min': ranges[0], 'max': ranges[1]},
        'cec_cmol_kg': {'min': ranges[2], 'max': ranges[3]},
        'soc_percent': {'min': ranges[4], 'max': ranges[5]},
        'esp_percent': {'min': ranges[6], 'max': ranges[7]}
    }

    # Null counts
    cursor.execute("""
        SELECT
            SUM(CASE WHEN ph_cacl2 IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN cec_cmol_kg IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN soc_percent IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN ca_cmol_kg IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN mg_cmol_kg IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN na_cmol_kg IS NULL THEN 1 ELSE 0 END)
        FROM soil_observations
    """)
    nulls = cursor.fetchone()
    summary['null_counts'] = {
        'ph_cacl2': nulls[0],
        'cec_cmol_kg': nulls[1],
        'soc_percent': nulls[2],
        'ca_cmol_kg': nulls[3],
        'mg_cmol_kg': nulls[4],
        'na_cmol_kg': nulls[5]
    }

    # Data sources
    cursor.execute("""
        SELECT dataset_source, COUNT(*)
        FROM soil_observations
        GROUP BY dataset_source
    """)
    summary['sources'] = dict(cursor.fetchall())

    conn.close()
    return summary


def export_to_csv(output_path: str, db_path: str = DATABASE_PATH) -> None:
    """Export all soil observations to CSV."""
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM soil_observations", conn)
    df.to_csv(output_path, index=False)
    conn.close()
    print(f"Exported {len(df)} records to {output_path}")


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print("Database schema created successfully.")
