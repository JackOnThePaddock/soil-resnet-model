import os
import re
import numpy as np
import pandas as pd
import rasterio

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp"
IN_DIR = os.path.join(BASE, "clipped_ndvi35")
OUT_DIR = os.path.join(BASE, "clipped_ndvi35_bias_corrected")
BIAS_CSV = os.path.join(BASE, "cv_paddock_holdout_bias_summary.csv")
NODATA = -9999.0

os.makedirs(OUT_DIR, exist_ok=True)

bias_df = pd.read_csv(BIAS_CSV)
bias_df = bias_df[bias_df["feature_set"] == "best_bands"].copy()

# Map paddock + target -> mean error (prediction - actual)
bias_map = {}
for row in bias_df.itertuples(index=False):
    key = (str(row.paddock).upper(), str(row.target).upper())
    bias_map[key] = float(row.me_bias)

pattern = re.compile(r"^(400|HILLPDK|LIGHTNING_TREE)_(pH|CEC|ESP)_rf_bestbands_clip_ndvi35\.tif$", re.IGNORECASE)

applied = []
for fname in os.listdir(IN_DIR):
    m = pattern.match(fname)
    if not m:
        continue
    paddock = m.group(1).upper()
    target = m.group(2).upper()
    key = (paddock, target)
    if key not in bias_map:
        print(f"No bias for {key}, skipping {fname}")
        continue
    bias = bias_map[key]
    in_path = os.path.join(IN_DIR, fname)
    out_path = os.path.join(OUT_DIR, fname.replace("_clip_ndvi35", "_clip_ndvi35_bias"))

    with rasterio.open(in_path) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        mask = data == NODATA
        # Correct prediction: pred - mean_error
        data[~mask] = data[~mask] - bias
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)

    applied.append({
        "file": fname,
        "paddock": paddock,
        "target": target,
        "bias_applied": -bias,
    })
    print(f"Wrote {out_path}")

pd.DataFrame(applied).to_csv(os.path.join(OUT_DIR, "bias_applied_summary.csv"), index=False)
