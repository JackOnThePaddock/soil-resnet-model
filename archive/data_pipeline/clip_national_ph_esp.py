import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")

NATIONAL_ALL = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median.csv"
NATIONAL_ESP = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_pct_top10cm_alphaearth_5yr.csv"

OUT_PH = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median_ph_le85.csv"
OUT_ESP = BASE_DIR / "external_sources" / "ansis_exchangeable_sodium_pct_top10cm_alphaearth_5yr_esp_le15.csv"


def main():
    # Clip pH
    df_all = pd.read_csv(NATIONAL_ALL)
    df_all = df_all.rename(columns={"pH": "ph"})
    if "ph" in df_all.columns:
        df_ph = df_all[df_all["ph"].notna() & (df_all["ph"] <= 8.5)].copy()
    else:
        df_ph = df_all.copy()
    df_ph.to_csv(OUT_PH, index=False)
    print(f"Wrote {OUT_PH} ({len(df_ph)} rows)")

    # Clip ESP
    df_esp = pd.read_csv(NATIONAL_ESP)
    df_esp = df_esp.rename(columns={"ESP": "esp_pct"})
    if "esp_pct" in df_esp.columns:
        df_esp = df_esp[df_esp["esp_pct"].notna() & (df_esp["esp_pct"] <= 15)].copy()
    df_esp.to_csv(OUT_ESP, index=False)
    print(f"Wrote {OUT_ESP} ({len(df_esp)} rows)")


if __name__ == "__main__":
    main()
