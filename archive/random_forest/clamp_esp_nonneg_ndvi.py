import os
import numpy as np
import rasterio

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\all_paddocks_1x1\clipped_ndvi35"

def clamp_file(in_path, out_path):
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        nodata = src.nodata
        if nodata is None:
            nodata = -9999.0
            meta.update(nodata=nodata)
        meta.update(dtype="float32")
        with rasterio.open(out_path, "w", **meta) as dst:
            for _, window in src.block_windows(1):
                data = src.read(1, window=window).astype(np.float32)
                mask = data == nodata
                data[~mask] = np.where(data[~mask] < 0, 0, data[~mask])
                dst.write(data, 1, window=window)

for fname in os.listdir(BASE):
    if not fname.lower().endswith(".tif"):
        continue
    if "_esp_" not in fname.lower():
        continue
    if "_nonneg" in fname.lower():
        continue
    in_path = os.path.join(BASE, fname)
    out_path = os.path.join(BASE, fname[:-4] + "_nonneg.tif")
    clamp_file(in_path, out_path)
    print(f"Wrote {out_path}")
