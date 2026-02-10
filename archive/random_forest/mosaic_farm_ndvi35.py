import os
import glob
import rasterio
from rasterio.merge import merge

BASE = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\all_paddocks_1x1"
IN_DIR = os.path.join(BASE, "clipped_ndvi35")
OUT_DIR = os.path.join(BASE, "farm_mosaics_ndvi35")

os.makedirs(OUT_DIR, exist_ok=True)


def mosaic(pattern, out_name):
    files = sorted(glob.glob(os.path.join(IN_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    srcs = [rasterio.open(fp) for fp in files]
    try:
        mosaic_arr, out_trans = merge(srcs, nodata=-9999.0)
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "height": mosaic_arr.shape[1],
            "width": mosaic_arr.shape[2],
            "transform": out_trans,
            "nodata": -9999.0,
            "compress": "LZW",
            "count": 1,
            "dtype": "float32",
        })
        out_path = os.path.join(OUT_DIR, out_name)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mosaic_arr[0].astype("float32"), 1)
        print(f"Wrote {out_path}")
    finally:
        for src in srcs:
            src.close()


mosaic("*_pH_rf_bestbands_1x1_clip_ndvi35.tif", "farm_pH_rf_bestbands_1x1_ndvi35.tif")
mosaic("*_CEC_rf_bestbands_1x1_clip_ndvi35.tif", "farm_CEC_rf_bestbands_1x1_ndvi35.tif")
mosaic("*_ESP_rf_bestbands_1x1_clip_ndvi35_nonneg.tif", "farm_ESP_rf_bestbands_1x1_ndvi35_nonneg.tif")
