import csv
from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
ANSIS_CSV = BASE_DIR / "external_sources" / "ansis_soil_cores_last10yrs.csv"
LOCAL_CSV = BASE_DIR / "outputs" / "training" / "training_points_1x1.csv"
OUT_DIR = BASE_DIR / "external_sources"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "soil_points_all_no400.csv"
OUT_KML = OUT_DIR / "soil_points_all_no400.kml"


def load_ansis():
    df = pd.read_csv(ANSIS_CSV)
    df["source"] = "ANSIS"
    df["paddock"] = None
    df["sample_id"] = None
    df.rename(columns={"site_id": "site_id"}, inplace=True)
    cols = [
        "source",
        "site_id",
        "sample_id",
        "paddock",
        "lat",
        "lon",
        "date",
        "depth_upper_m",
        "depth_lower_m",
        "depth_upper_unit",
        "depth_lower_unit",
        "pH",
        "pH_unit",
        "CEC",
        "CEC_unit",
        "ESP",
        "ESP_unit",
        "Na_cmol",
        "Na_cmol_unit",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def load_local():
    df = pd.read_csv(LOCAL_CSV)
    # exclude paddock 400 if present
    if "paddock" in df.columns:
        df = df[df["paddock"].astype(str).str.strip().str.lower() != "400"]
    df["source"] = "LOCAL"
    df["site_id"] = None
    df["date"] = None
    df["depth_upper_m"] = None
    df["depth_lower_m"] = None
    df["depth_upper_unit"] = None
    df["depth_lower_unit"] = None
    # Units unknown in local CSV; keep blank
    df["pH_unit"] = None
    df["CEC_unit"] = None
    df["ESP_unit"] = None
    df["Na_cmol_unit"] = None
    cols = [
        "source",
        "site_id",
        "sample_id",
        "paddock",
        "lat",
        "lon",
        "date",
        "depth_upper_m",
        "depth_lower_m",
        "depth_upper_unit",
        "depth_lower_unit",
        "pH",
        "pH_unit",
        "CEC",
        "CEC_unit",
        "ESP",
        "ESP_unit",
        "Na_cmol",
        "Na_cmol_unit",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


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


def main():
    ansis = load_ansis()
    local = load_local()
    combined = pd.concat([ansis, local], ignore_index=True)
    # drop rows with missing coords
    combined = combined[combined["lat"].notna() & combined["lon"].notna()]

    combined.to_csv(OUT_CSV, index=False)
    to_kml(combined, OUT_KML)
    print(f"Wrote {OUT_CSV} ({len(combined)} rows)")
    print(f"Wrote {OUT_KML}")


if __name__ == "__main__":
    main()
