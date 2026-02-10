import os
import numpy as np
import rasterio

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp"
DIRS = [
    BASE,
    os.path.join(BASE, "clipped_ndvi35"),
    os.path.join(BASE, "clipped_ndvi35_bias_corrected"),
]


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


for d in DIRS:
    if not os.path.isdir(d):
        continue
    for fname in os.listdir(d):
        if not fname.lower().endswith(".tif"):
            continue
        if "esp" not in fname.lower():
            continue
        if "_nonneg" in fname.lower():
            continue
        in_path = os.path.join(d, fname)
        out_name = fname[:-4] + "_nonneg.tif"
        out_path = os.path.join(d, out_name)
        clamp_file(in_path, out_path)
        print(f"Wrote {out_path}")
