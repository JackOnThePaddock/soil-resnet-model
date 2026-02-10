import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapefile
from PyPDF2 import PdfReader


def extract_float(text: str, patterns: list[str]):
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def map_pdf_paddock(name: str):
    n = name.strip().lower()
    if n.startswith("vida"):
        m = re.search(r"vida\s*l\s*([123])", n)
        if m:
            return f"Lease_{m.group(1)}"
    if n.startswith("md") or n.startswith("mh"):
        return "MERRIMU_DRIVEWAY"
    if n.startswith("40088"):
        return "400"
    if n.startswith("400"):
        return "400"
    if n.startswith("300"):
        return "300"
    return None


def read_pdf_samples(pdf_path: Path) -> pd.DataFrame:
    reader = PdfReader(str(pdf_path))
    rows = []
    for page in reader.pages:
        text = page.extract_text() or ""
        m = re.search(r"PADDOCK NAME\s*:?(.+)", text, re.IGNORECASE)
        if not m:
            continue
        paddock_name = m.group(1).strip().split("\n")[0].strip()
        paddock = map_pdf_paddock(paddock_name)
        if not paddock:
            continue

        ph = extract_float(
            text,
            [
                r"pH\s*\(1:5\s*CaCl2\)\s*([0-9.]+)",
                r"pH\s*\(1:5\s*CaCI2\)\s*([0-9.]+)",
            ],
        )
        cec = extract_float(
            text,
            [
                r"eCEC\s*cmol\+?/kg\s*([0-9.]+)",
                r"eCEC\s*cmol\+?\/kg\s*([0-9.]+)",
            ],
        )

        if ph is None and cec is None:
            continue

        rows.append({"paddock": paddock, "pH": ph, "CEC": cec, "source": "pdf"})

    return pd.DataFrame(rows)


def paddock_from_shp(path: Path):
    name = path.stem.lower()
    if "lightning_tree" in name:
        return "LIGHTNING_TREE"
    if "hill_pdk" in name or "hill_pdk_north" in name or "hill_pdk_south" in name:
        return "HILLPDK"
    return None


def read_shp_samples(shp_path: Path) -> pd.DataFrame:
    paddock = paddock_from_shp(shp_path)
    if not paddock:
        return pd.DataFrame()
    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    field_idx = {name: idx for idx, name in enumerate(fields)}

    rows = []
    for rec in reader.records():
        ph = rec[field_idx["pH"]] if "pH" in field_idx else None
        cec = rec[field_idx["CEC"]] if "CEC" in field_idx else None
        if ph is None and cec is None:
            continue
        rows.append({"paddock": paddock, "pH": ph, "CEC": cec, "source": "shp"})
    return pd.DataFrame(rows)


def read_400_points(points_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(points_csv)
    if "pH (CaCl2)" not in df.columns or "eCEC (cmol+/kg)" not in df.columns:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "paddock": "400",
            "pH": df["pH (CaCl2)"],
            "CEC": df["eCEC (cmol+/kg)"],
            "source": "points_400",
        }
    )


def calibrate_raster(src_path: Path, out_path: Path, obs_min: float, obs_max: float):
    with rasterio.open(src_path) as src:
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        mask = (data == nodata) | np.isnan(data)
        valid = ~mask
        if not np.any(valid):
            return

        pred_min = float(np.nanmin(data[valid]))
        pred_max = float(np.nanmax(data[valid]))
        if pred_max == pred_min:
            return

        scaled = (data - pred_min) / (pred_max - pred_min)
        scaled = scaled * (obs_max - obs_min) + obs_min
        scaled = np.clip(scaled, obs_min, obs_max)
        scaled[mask] = nodata

        profile = src.profile.copy()
        profile.update(dtype="float32", compress="LZW", nodata=nodata)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(scaled.astype(np.float32), 1)

    return pred_min, pred_max


def main():
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
    out_dir = base_dir / "exports" / "rf_minmax_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = base_dir / "EW_WH_and_MG_Speirs" / "1771_001.pdf"
    points_csv = base_dir / "exports" / "400_calibration_points.csv"
    preds_dir = base_dir / "exports" / "speirs_paddock_predictions"

    shp_dir = base_dir / "EW_WH_and_MG_Speirs"
    shp_paths = sorted(shp_dir.glob("*Soil_Sampling*.shp"))

    df_pdf = read_pdf_samples(pdf_path) if pdf_path.exists() else pd.DataFrame()
    df_shp = pd.concat([read_shp_samples(p) for p in shp_paths], ignore_index=True)
    df_pts = read_400_points(points_csv) if points_csv.exists() else pd.DataFrame()

    samples = pd.concat([df_pdf, df_shp, df_pts], ignore_index=True)
    samples = samples.dropna(subset=["paddock"])
    samples.to_csv(out_dir / "soil_samples_combined.csv", index=False)

    minmax = (
        samples.groupby("paddock")[["pH", "CEC"]]
        .agg(["min", "max"])
        .reset_index()
    )
    minmax.columns = ["paddock", "pH_min", "pH_max", "CEC_min", "CEC_max"]
    minmax.to_csv(out_dir / "paddock_minmax.csv", index=False)

    minmax_map = {
        row["paddock"]: row
        for _, row in minmax.iterrows()
    }

    calib_dir = out_dir / "rfe_best_minmax_calibrated"
    calib_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pattern = re.compile(r"^(.*)_(pH|CEC)_5yr_rfe_best\.tif$", re.IGNORECASE)
    for tif in preds_dir.glob("*_5yr_rfe_best.tif"):
        match = pattern.match(tif.name)
        if not match:
            continue
        paddock = match.group(1)
        var = match.group(2).upper()
        if paddock not in minmax_map:
            continue

        row = minmax_map[paddock]
        col_prefix = "pH" if var == "PH" else var
        obs_min = row[f"{col_prefix}_min"]
        obs_max = row[f"{col_prefix}_max"]
        if pd.isna(obs_min) or pd.isna(obs_max) or obs_min == obs_max:
            continue

        out_path = calib_dir / tif.name
        pred_min, pred_max = calibrate_raster(tif, out_path, obs_min, obs_max)
        summary_rows.append(
            {
                "paddock": paddock,
                "variable": var,
                "obs_min": obs_min,
                "obs_max": obs_max,
                "pred_min": pred_min,
                "pred_max": pred_max,
                "output": str(out_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "calibration_summary.csv", index=False)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
