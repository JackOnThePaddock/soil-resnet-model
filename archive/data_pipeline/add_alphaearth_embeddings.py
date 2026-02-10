"""
Add annual AlphaEarth embeddings (Google Earth Engine) to soil points.

Single-file mode:
  python add_alphaearth_embeddings.py --project PROJECT --input in.csv --output out.csv

Batch mode (soil_tests_YYYY folders):
  python add_alphaearth_embeddings.py --project PROJECT --batch-dir "C:/Users/jackc/Downloads/Soil Data/Soil Tests"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    import ee
except Exception as e:
    print("ERROR: earthengine-api (ee) is not available:", e, file=sys.stderr)
    sys.exit(1)

DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

DEFAULT_INPUT = Path(r"C:\Users\jackc\Downloads\Soil Data\Soil Tests\soil_data_export.csv")
DEFAULT_OUTPUT = Path(r"C:\Users\jackc\Downloads\Soil Data\Soil Tests\soil_data_with_alphaearth.csv")
DEFAULT_BATCH_DIR = Path(r"C:\Users\jackc\Downloads\Soil Data\Soil Tests")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--batch-dir", default=None, help="Folder with soil_tests_YYYY subfolders")
    ap.add_argument("--years", default=None, help="Comma-separated years for batch mode (e.g., 2017,2018,2019)")
    ap.add_argument("--max-workers", type=int, default=0, help="Max concurrent years (0 = auto)")
    ap.add_argument("--project", required=True, help="GEE project ID")
    ap.add_argument("--lat-col", default="latitude")
    ap.add_argument("--lon-col", default="longitude")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--date-col", default="observation_date")
    ap.add_argument("--year-col", default="year")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--tile-scale", type=int, default=4)
    return ap.parse_args()


def ensure_year(df: pd.DataFrame, year_col: str, date_col: str) -> pd.DataFrame:
    if year_col in df.columns:
        return df
    if date_col in df.columns:
        years = pd.to_datetime(df[date_col], errors="coerce").dt.year
        df[year_col] = years
        return df
    raise ValueError(f"No '{year_col}' or '{date_col}' column found to derive year.")


def to_fc(df: pd.DataFrame, lat_col: str, lon_col: str, row_id_col: str):
    feats = []
    for lat, lon, row_id in df[[lat_col, lon_col, row_id_col]].itertuples(index=False, name=None):
        if pd.isna(lat) or pd.isna(lon):
            continue
        feats.append(
            ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"row_id": int(row_id)})
        )
    return ee.FeatureCollection(feats)


def get_image_for_year(year: int) -> ee.Image | None:
    # Use calendarRange to match the correct year based on system:time_start.
    coll = ee.ImageCollection(DATASET_ID).filter(ee.Filter.calendarRange(int(year), int(year), "year"))
    try:
        if int(coll.size().getInfo()) == 0:
            return None
    except Exception:
        pass
    # Mosaic the tiles for full coverage in that year.
    return coll.mosaic()


def get_band_names(img: ee.Image) -> List[str]:
    names = img.bandNames().getInfo()
    if not names:
        raise RuntimeError("No band names returned from AlphaEarth image.")
    return [str(n) for n in names]


def sample_year(
    df_year: pd.DataFrame,
    year: int,
    lat_col: str,
    lon_col: str,
    row_id_col: str,
    scale: int,
    tile_scale: int,
    chunk_size: int,
    band_names: List[str],
) -> pd.DataFrame:
    if df_year.empty:
        return pd.DataFrame(columns=["row_id"] + band_names)

    img = get_image_for_year(year)
    if img is None:
        return pd.DataFrame(columns=["row_id"] + band_names)

    img = img.select(band_names)

    rows = []
    for start in range(0, len(df_year), chunk_size):
        chunk = df_year.iloc[start:start + chunk_size]
        fc = to_fc(chunk, lat_col, lon_col, row_id_col)
        samples = img.sampleRegions(
            collection=fc,
            properties=["row_id"],
            scale=scale,
            tileScale=tile_scale,
            geometries=False,
        )
        info = samples.getInfo()
        for feat in info.get("features", []):
            props = feat.get("properties", {})
            rows.append(props)

    if not rows:
        return pd.DataFrame(columns=["row_id"] + band_names)

    df = pd.DataFrame(rows)
    for b in band_names:
        if b not in df.columns:
            df[b] = pd.NA
    return df[["row_id"] + band_names]


def build_output(df: pd.DataFrame, band_names: List[str], id_col: str, lat_col: str, lon_col: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = df[id_col] if id_col in df.columns else df.get("site_id")
    out["lat"] = df[lat_col]
    out["lon"] = df[lon_col]

    for i, b in enumerate(band_names):
        out[f"band_{i}"] = df.get(b, pd.NA)

    out["ph"] = df.get("ph_cacl2", df.get("ph"))
    out["cec"] = df.get("cec_cmol_kg", df.get("cec"))
    out["esp"] = df.get("esp_percent", df.get("esp"))
    out["soc"] = df.get("soc_percent", df.get("soc"))
    out["ca"] = df.get("ca_cmol_kg", df.get("ca"))
    out["mg"] = df.get("mg_cmol_kg", df.get("mg"))
    out["na"] = df.get("na_cmol_kg", df.get("na"))
    return out


def process_single(input_path: Path, output_path: Path, args, band_names: List[str]) -> Tuple[Path, int]:
    df = pd.read_csv(input_path)
    df = ensure_year(df, args.year_col, args.date_col)

    df = df.reset_index(drop=True).copy()
    df["_row_id"] = df.index.astype(int)

    years = sorted([int(y) for y in df[args.year_col].dropna().unique()])
    if len(years) != 1:
        print(f"WARNING: {input_path.name} has multiple years: {years}")

    all_samples = []
    for y in years:
        df_y = df[df[args.year_col] == y]
        print(f"Sampling {y} ({len(df_y)} points) for {input_path.name}...")
        sample_df = sample_year(
            df_y,
            y,
            args.lat_col,
            args.lon_col,
            "_row_id",
            args.scale,
            args.tile_scale,
            args.chunk_size,
            band_names,
        )
        sample_df["year"] = y
        all_samples.append(sample_df)

    if all_samples:
        emb = pd.concat(all_samples, ignore_index=True)
    else:
        emb = pd.DataFrame(columns=["row_id"] + band_names)

    emb = emb.rename(columns={"row_id": "_row_id"})
    merged = df.merge(emb, on="_row_id", how="left")

    out = build_output(merged, band_names, args.id_col, args.lat_col, args.lon_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote: {output_path} ({len(out)} rows)")
    return output_path, len(out)


def main():
    args = parse_args()

    try:
        ee.Initialize(project=args.project)
    except Exception as e:
        print("ERROR: Failed to initialize Earth Engine.")
        print("", e)
        print("Try: earthengine authenticate  (or provide a valid project id)")
        sys.exit(1)

    band_img = get_image_for_year(2017) or get_image_for_year(2018) or get_image_for_year(2019)
    if band_img is None:
        print("ERROR: Could not find any AlphaEarth annual image for 2017-2019.")
        sys.exit(1)
    band_names = get_band_names(band_img)
    print(f"AlphaEarth bands: {len(band_names)}")

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        years = None
        if args.years:
            years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
        else:
            years = [int(p.name.replace("soil_tests_", "")) for p in batch_dir.glob("soil_tests_20??") if p.is_dir()]
            years = sorted([y for y in years if 2000 <= y <= 2099])

        if not years:
            print("No year folders found for batch mode.")
            sys.exit(1)

        tasks = []
        for y in years:
            in_path = batch_dir / f"soil_tests_{y}" / f"soil_data_{y}.csv"
            out_path = batch_dir / f"soil_tests_{y}" / f"soil_data_{y}_alphaearth.csv"
            if not in_path.exists():
                print(f"Missing input: {in_path}")
                continue
            tasks.append((y, in_path, out_path))

        if not tasks:
            print("No valid inputs found for batch mode.")
            sys.exit(1)

        max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else min(len(tasks), 8)
        print(f"Running {len(tasks)} years with max_workers={max_workers}")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(process_single, in_path, out_path, args, band_names): (y, in_path)
                for y, in_path, out_path in tasks
            }
            for fut in as_completed(futures):
                y, in_path = futures[fut]
                try:
                    out_path, count = fut.result()
                    print(f"Done {y}: {out_path.name} ({count} rows)")
                except Exception as e:
                    print(f"ERROR {y} ({in_path.name}): {e}")
        return

    in_path = Path(args.input)
    out_path = Path(args.output)
    process_single(in_path, out_path, args, band_names)


if __name__ == "__main__":
    main()
