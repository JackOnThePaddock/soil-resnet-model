import pandas as pd
from pathlib import Path

base = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
files = {
    'national_all': base / 'external_sources' / 'soil_points_all_no400_top10cm_alphaearth_5yr_median.csv',
    'local_points': base / 'outputs' / 'training' / 'training_points_alphaearth_5yr.csv'
}

for name, p in files.items():
    if not p.exists():
        print(name, 'missing', p)
        continue
    df = pd.read_csv(p)
    df = df.rename(columns={'ESP': 'esp_pct', 'pH': 'ph'})
    print(f"\n== {name} {p}")
    for col in ['ph', 'esp_pct']:
        if col not in df.columns:
            print(col, 'missing')
            continue
        s = df[col].dropna()
        if s.empty:
            print(col, 'no data')
            continue
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        iqr = q.loc[0.75] - q.loc[0.25]
        lo = q.loc[0.25] - 1.5 * iqr
        hi = q.loc[0.75] + 1.5 * iqr
        out = s[(s < lo) | (s > hi)]
        print(f"{col}: n={len(s)} min={s.min():.3f} max={s.max():.3f} Q1={q.loc[0.25]:.3f} Q3={q.loc[0.75]:.3f} IQR={iqr:.3f} outliers={len(out)}")
        print('  p01..p99:', ', '.join([f"{k}:{v:.3f}" for k, v in q.items()]))
