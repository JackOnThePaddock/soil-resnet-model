"""
Parallel ESPADE extraction for 2017+ records only.
Outputs a standalone CSV for review.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

BASE_DIR = Path(r"C:\Users\jackc\Downloads\Soil Data")
ESPADE_DIR = BASE_DIR / "espade_soil_data"
PROFILE_DIR = ESPADE_DIR / "profile_data"
SAMPLE_DIR = ESPADE_DIR / "sample_data"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_FILE = OUTPUT_DIR / "espade_2017plus_lab.csv"

TARGET_FILES = [
    "sample_data_200_part_288.xlsx",
    "sample_data_200_part_290.xlsx",
    "sample_data_200_part_304.xlsx",
    "sample_data_200_part_306.xlsx",
    "sample_data_200_part_307.xlsx",
    "sample_data_200_part_308.xlsx",
    "sample_data_200_part_309.xlsx",
    "sample_data_200_part_310.xlsx",
    "sample_data_200_part_311.xlsx",
    "sample_data_200_part_312.xlsx",
    "sample_data_200_part_313.xlsx",
    "sample_data_200_part_315.xlsx",
    "sample_data_200_part_316.xlsx",
    "sample_data_200_part_317.xlsx",
    "sample_data_200_part_318.xlsx",
]

PROFILE_COLS = ["SoilProfileID", "Latitude", "Longitude", "SoilProfileDate"]
SAMPLE_COLS = [
    "SoilProfileID",
    "BoundUpper",
    "BoundLower",
    "SampleDate",
    "N4A1",
    "N4B1",  # Lab pH
    "N15A1_CA",
    "N15A1_MG",
    "N15A1_NA",
    "N15A1_ECEC",
    "N15B1_CA",
    "N15B1_MG",
    "N15B1_NA",
    "N15B1_CEC",
    "N15C1_CA",
    "N15C1_MG",
    "N15C1_NA",
    "N15C1_CEC",
]

MAX_DEPTH_CM = 15
MAX_PH = 8.5
MAX_ESP = 25.0

# Globals for worker processes
PROFILE_COORDS = {}
PROFILE_DATES = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def extract_year(date_val) -> int | None:
    if pd.isna(date_val) or date_val is None:
        return None
    if isinstance(date_val, (pd.Timestamp,)):
        try:
            return int(date_val.year)
        except Exception:
            pass
    s = str(date_val)
    matches = re.findall(r"(\d{4})", s)
    for m in reversed(matches):
        year = int(m)
        if 2000 <= year <= 2030:
            return year
    return None


def safe_float(val) -> float | None:
    try:
        if pd.isna(val) or val in ("", "NA", "na"):
            return None
        return float(val)
    except Exception:
        return None


def first_valid(*values):
    for v in values:
        f = safe_float(v)
        if f is not None:
            return f
    return None


def init_worker(profile_coords, profile_dates):
    global PROFILE_COORDS, PROFILE_DATES
    PROFILE_COORDS = profile_coords
    PROFILE_DATES = profile_dates


def process_file(file_path: str, file_index: int):
    records = []
    counts = {
        "skipped_depth": 0,
        "skipped_ph": 0,
        "skipped_esp": 0,
        "skipped_no_data": 0,
        "skipped_no_profile": 0,
        "kept": 0,
    }

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        return records, counts, f"ERROR reading {Path(file_path).name}: {e}"

    available_cols = [c for c in SAMPLE_COLS if c in df.columns]
    if not available_cols:
        return records, counts, None

    df = df[available_cols]

    for row_idx, row in enumerate(df.itertuples(index=False), start=1):
        row_dict = row._asdict()
        pid = row_dict.get("SoilProfileID")
        try:
            pid = int(float(pid))
        except Exception:
            continue

        if pid not in PROFILE_COORDS:
            counts["skipped_no_profile"] += 1
            continue

        lat, lon = PROFILE_COORDS[pid]
        if not (-45 <= lat <= -10 and 112 <= lon <= 155):
            continue

        upper = safe_float(row_dict.get("BoundUpper"))
        lower = safe_float(row_dict.get("BoundLower"))
        if upper is None or lower is None:
            continue

        lower_cm = lower * 100 if lower < 1 else lower
        if upper != 0 or lower_cm > MAX_DEPTH_CM:
            counts["skipped_depth"] += 1
            continue

        date_val = row_dict.get("SampleDate")
        if pd.isna(date_val) or date_val is None:
            date_val = PROFILE_DATES.get(pid)
        year = extract_year(date_val)
        if not year or year < 2017:
            continue

        ph = first_valid(row_dict.get("N4A1"), row_dict.get("N4B1"))
        ca = first_valid(row_dict.get("N15A1_CA"), row_dict.get("N15B1_CA"), row_dict.get("N15C1_CA"))
        mg = first_valid(row_dict.get("N15A1_MG"), row_dict.get("N15B1_MG"), row_dict.get("N15C1_MG"))
        na = first_valid(row_dict.get("N15A1_NA"), row_dict.get("N15B1_NA"), row_dict.get("N15C1_NA"))
        cec = first_valid(row_dict.get("N15A1_ECEC"), row_dict.get("N15B1_CEC"), row_dict.get("N15C1_CEC"))

        if ph is None and ca is None and cec is None:
            counts["skipped_no_data"] += 1
            continue

        if ph is not None and ph > MAX_PH:
            counts["skipped_ph"] += 1
            continue

        esp = None
        if na is not None and cec is not None and cec > 0:
            esp = (na / cec) * 100

        if esp is not None and esp > MAX_ESP:
            counts["skipped_esp"] += 1
            continue

        rec_id = f"ESPADE_LAB_{year}_{file_index:02d}_{row_idx:06d}"
        records.append(
            {
                "id": rec_id,
                "year": year,
                "lat": lat,
                "lon": lon,
                "ph": ph,
                "cec": cec,
                "esp": esp,
                "soc": None,
                "ca": ca,
                "mg": mg,
                "na": na,
            }
        )
        counts["kept"] += 1

    return records, counts, None


def main():
    log("=" * 60)
    log("ESPADE PARALLEL EXTRACTION (2017+ only)")
    log(f"Output: {OUTPUT_FILE}")
    log("=" * 60)

    # Load profiles once
    log("[1/3] Loading profile coordinates and dates...")
    profile_coords = {}
    profile_dates = {}

    profile_files = sorted(PROFILE_DIR.glob("profile_data_part_*.xlsx"))
    for i, pf in enumerate(profile_files, start=1):
        try:
            df = pd.read_excel(pf, usecols=lambda x: x in PROFILE_COLS, engine="openpyxl")
            for _, row in df.iterrows():
                pid = row.get("SoilProfileID")
                lat = row.get("Latitude")
                lon = row.get("Longitude")
                if pd.notna(pid) and pd.notna(lat) and pd.notna(lon):
                    try:
                        pid_int = int(float(pid))
                        profile_coords[pid_int] = (float(lat), float(lon))
                        date = row.get("SoilProfileDate")
                        if pd.notna(date):
                            profile_dates[pid_int] = date
                    except Exception:
                        pass
        except Exception as e:
            log(f"  Error {pf.name}: {e}")

        if i % 10 == 0:
            log(f"  {i}/{len(profile_files)} profile files")

    log(f"  Loaded profiles: {len(profile_coords)}")

    # Build file list
    sample_files = [SAMPLE_DIR / f for f in TARGET_FILES]
    missing = [f.name for f in sample_files if not f.exists()]
    if missing:
        log(f"Missing files: {missing}")
        return

    # Run parallel extraction
    log("[2/3] Running parallel extraction...")
    max_workers = min(15, os.cpu_count() or 15)
    log(f"  Workers: {max_workers}")

    all_records = []
    agg_counts = {
        "skipped_depth": 0,
        "skipped_ph": 0,
        "skipped_esp": 0,
        "skipped_no_data": 0,
        "skipped_no_profile": 0,
        "kept": 0,
    }

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(profile_coords, profile_dates)) as ex:
        futures = {}
        for idx, sf in enumerate(sample_files, start=1):
            futures[ex.submit(process_file, str(sf), idx)] = sf.name

        completed = 0
        for fut in as_completed(futures):
            fname = futures[fut]
            records, counts, err = fut.result()
            completed += 1
            if err:
                log(f"  {fname}: {err}")
            all_records.extend(records)
            for k in agg_counts:
                agg_counts[k] += counts.get(k, 0)
            log(f"  Done {completed}/{len(sample_files)}: {fname} (kept {counts.get('kept', 0)})")

    # Write output
    log("[3/3] Writing output...")
    if all_records:
        out_df = pd.DataFrame(all_records)
        out_df.to_csv(OUTPUT_FILE, index=False)
        log(f"  Wrote {len(out_df)} records to {OUTPUT_FILE}")
    else:
        log("  No records found.")

    log("Summary:")
    log(f"  Kept: {agg_counts['kept']}")
    log(f"  Skipped depth: {agg_counts['skipped_depth']}")
    log(f"  Skipped pH: {agg_counts['skipped_ph']}")
    log(f"  Skipped ESP: {agg_counts['skipped_esp']}")
    log(f"  Skipped no lab data: {agg_counts['skipped_no_data']}")
    log(f"  Skipped no profile: {agg_counts['skipped_no_profile']}")


if __name__ == "__main__":
    main()
