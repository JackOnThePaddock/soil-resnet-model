from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path(r"C:\Users\jackc\Documents\National Soil Data Standardised")
IN_DIR = BASE_DIR / "by_year_cleaned_top10cm_metrics_alphaearth"
OUT_DIR = BASE_DIR / "by_year_cleaned_top10cm_metrics_alphaearth" / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ATTRS = {
    "ph": "ph",
    "cec_cmolkg": "cec_cmolkg",
    "esp_pct": "esp_pct",
}


def band_cols(df):
    return [c for c in df.columns if re.fullmatch(r"A\d{2}", c)]


def main():
    for attr, target in ATTRS.items():
        files = sorted(IN_DIR.glob(f"top10cm_{attr}_*_alphaearth_*.csv"))
        if not files:
            print(f"No files for {attr}")
            continue

        frames = []
        for path in files:
            df = pd.read_csv(path)
            if target not in df.columns:
                print(f"Skipping {path.name}: missing {target}")
                continue
            keep = [c for c in ["lat", "lon", "date", "year", target] if c in df.columns]
            keep += band_cols(df)
            df = df[keep].copy()
            frames.append(df)

        if not frames:
            print(f"No valid data for {attr}")
            continue

        merged = pd.concat(frames, ignore_index=True)
        out_path = OUT_DIR / f"top10cm_{attr}_alphaearth_merged.csv"
        merged.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
