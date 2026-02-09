"""SQLite database handler for soil data storage."""

import sqlite3
from typing import Optional

import pandas as pd


def get_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_database(db_path: str) -> None:
    """Initialize database with soil observations schema."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS soil_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT, dataset_source TEXT,
            latitude REAL NOT NULL, longitude REAL NOT NULL,
            observation_date DATE,
            upper_depth_cm INTEGER DEFAULT 0, lower_depth_cm INTEGER DEFAULT 15,
            ph_cacl2 REAL, cec_cmol_kg REAL, esp_percent REAL, soc_percent REAL,
            ca_cmol_kg REAL, mg_cmol_kg REAL, na_cmol_kg REAL,
            method_ph TEXT, method_cec TEXT, method_cations TEXT,
            data_quality TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON soil_observations(latitude, longitude)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON soil_observations(observation_date)")
    conn.commit()
    conn.close()
    print(f"Database initialized: {db_path}")


def insert_soil_data(df: pd.DataFrame, db_path: str) -> int:
    """Insert soil observations into database. Returns count of records inserted."""
    conn = get_connection(db_path)
    valid_cols = ["site_id", "dataset_source", "latitude", "longitude", "observation_date",
                  "upper_depth_cm", "lower_depth_cm", "ph_cacl2", "cec_cmol_kg", "esp_percent",
                  "soc_percent", "ca_cmol_kg", "mg_cmol_kg", "na_cmol_kg",
                  "method_ph", "method_cec", "method_cations", "data_quality"]
    df_ins = df[[c for c in df.columns if c in valid_cols]]
    records = 0
    for i in range(0, len(df_ins), 100):
        chunk = df_ins.iloc[i:i + 100]
        try:
            chunk.to_sql("soil_observations", conn, if_exists="append", index=False)
            records += len(chunk)
        except sqlite3.IntegrityError:
            for _, row in chunk.iterrows():
                try:
                    row.to_frame().T.to_sql("soil_observations", conn, if_exists="append", index=False)
                    records += 1
                except sqlite3.IntegrityError:
                    pass
    conn.commit()
    conn.close()
    return records


def get_data_summary(db_path: str) -> dict:
    """Get summary statistics of the soil data."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM soil_observations")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(observation_date), MAX(observation_date) FROM soil_observations")
    dr = cursor.fetchone()
    conn.close()
    return {"total_records": total, "date_range": {"min": dr[0], "max": dr[1]}}


def export_to_csv(output_path: str, db_path: str) -> None:
    conn = get_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM soil_observations", conn)
    df.to_csv(output_path, index=False)
    conn.close()
    print(f"Exported {len(df)} records to {output_path}")
