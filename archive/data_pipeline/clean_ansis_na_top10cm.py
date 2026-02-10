import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_raw.csv"
OUT_CSV = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_top10cm.csv"
OUT_STD = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_top10cm_standardized.csv"


def main():
    df = pd.read_csv(IN_CSV)

    # remove missing date/coords
    date_ok = df["date"].notna() & df["date"].astype(str).str.strip().ne("")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    date_ok = date_ok & (df["date_parsed"] >= pd.Timestamp("2017-01-01"))
    coord_ok = df["lat"].notna() & df["lon"].notna()
    df = df[date_ok & coord_ok].copy()

    # numeric conversions
    for col in ["lat", "lon", "depth_upper_m", "depth_lower_m", "na_cmolkg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # top 10 cm filter where depth known
    depth_upper = df["depth_upper_m"].notna()
    depth_lower = df["depth_lower_m"].notna()
    has_depth = depth_upper | depth_lower
    depth_ok = (~has_depth) | ((df["depth_upper_m"].fillna(0) < 0.1) & (df["depth_lower_m"].fillna(0.1) <= 0.1))
    df = df[depth_ok].copy()

    # normalize na (z-score)
    mean = df["na_cmolkg"].mean(skipna=True)
    std = df["na_cmolkg"].std(skipna=True)
    if std and std > 0:
        df["na_cmolkg_z"] = (df["na_cmolkg"] - mean) / std
    else:
        df["na_cmolkg_z"] = None

    df.to_csv(OUT_CSV, index=False)
    df[["site_id", "lat", "lon", "date", "depth_upper_m", "depth_lower_m", "na_cmolkg_z"]].to_csv(OUT_STD, index=False)

    print(f"Wrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_STD}")


if __name__ == "__main__":
    main()
