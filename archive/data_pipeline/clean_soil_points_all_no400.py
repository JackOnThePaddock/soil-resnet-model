import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400.csv"
OUT_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top20cm.csv"
OUT_STD_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top20cm_standardized.csv"
OUT_KML = BASE_DIR / "external_sources" / "soil_points_all_no400_top20cm.kml"


def to_kml(df, out_path):
    def esc(val):
        if val is None:
            return ""
        s = str(val)
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n")
        f.write("  <Document>\n")
        for _, row in df.iterrows():
            lat = row.get("lat")
            lon = row.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                continue
            name_parts = []
            if row.get("paddock"):
                name_parts.append(str(row.get("paddock")))
            if row.get("sample_id"):
                name_parts.append(str(row.get("sample_id")))
            if row.get("site_id") and not name_parts:
                name_parts.append(str(row.get("site_id")))
            name = "_".join(name_parts) if name_parts else "soil_point"

            f.write("    <Placemark>\n")
            f.write(f"      <name>{esc(name)}</name>\n")
            f.write("      <ExtendedData>\n")
            for col in df.columns:
                val = row.get(col)
                if pd.isna(val):
                    val = None
                f.write(f"        <Data name=\"{esc(col)}\"><value>{esc(val)}</value></Data>\n")
            f.write("      </ExtendedData>\n")
            f.write("      <Point>\n")
            f.write(f"        <coordinates>{lon},{lat},0</coordinates>\n")
            f.write("      </Point>\n")
            f.write("    </Placemark>\n")
        f.write("  </Document>\n")
        f.write("</kml>\n")

def safe_write_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        df.to_csv(alt, index=False)
        return alt


def safe_write_kml(df, path):
    try:
        to_kml(df, path)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        to_kml(df, alt)
        return alt


def main():
    df = pd.read_csv(IN_CSV)

    # standardize column names for outputs
    rename_map = {
        "pH": "ph",
        "CEC": "cec_cmolkg",
        "ESP": "esp_pct",
        "Na_cmol": "na_cmolkg",
    }
    df = df.rename(columns=rename_map)

    # remove rows with unknown dates
    if "date" in df.columns:
        date_series = df["date"]
        date_ok = date_series.notna() & date_series.astype(str).str.strip().ne("") & date_series.astype(str).str.lower().ne("nan")
        df = df[date_ok].copy()

    # ensure numeric columns
    for col in ["lat", "lon", "depth_upper_m", "depth_lower_m", "ph", "cec_cmolkg", "esp_pct", "na_cmolkg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # filter: remove layers deeper than 20 cm (0.2 m) when depth is known
    depth_upper = df["depth_upper_m"].notna()
    depth_lower = df["depth_lower_m"].notna()
    has_depth = depth_upper | depth_lower

    depth_ok = pd.Series(True, index=df.index)
    # if we have depth info, keep only horizons where upper < 0.2 and lower <= 0.2
    depth_ok = depth_ok & (~has_depth | ((df["depth_upper_m"].fillna(0) < 0.2) & (df["depth_lower_m"].fillna(0.2) <= 0.2)))

    df = df[depth_ok].copy()
    df["depth_ok"] = True

    # Standardize outputs (z-score) for modeling
    for col in ["ph", "cec_cmolkg", "esp_pct", "na_cmolkg"]:
        if col in df.columns:
            mean = df[col].mean(skipna=True)
            std = df[col].std(skipna=True)
            if std and std > 0:
                df[f"{col}_z"] = (df[col] - mean) / std
            else:
                df[f"{col}_z"] = None

    # write cleaned CSV
    out_csv = safe_write_csv(df, OUT_CSV)

    # write standardized CSV (keep id/coords + z-scores)
    std_cols = [
        "source",
        "site_id",
        "sample_id",
        "paddock",
        "lat",
        "lon",
        "date",
        "depth_upper_m",
        "depth_lower_m",
        "ph_z",
        "cec_cmolkg_z",
        "esp_pct_z",
        "na_cmolkg_z",
    ]
    for c in std_cols:
        if c not in df.columns:
            df[c] = None
    out_std_csv = safe_write_csv(df[std_cols], OUT_STD_CSV)

    # kml for cleaned set
    out_kml = safe_write_kml(df, OUT_KML)

    print(f"Wrote {out_csv} ({len(df)} rows)")
    print(f"Wrote {out_std_csv}")
    print(f"Wrote {out_kml}")


if __name__ == "__main__":
    main()
