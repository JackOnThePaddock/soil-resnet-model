import os
import numpy as np
import rasterio

base = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\clipped_ndvi35_bias_corrected"
paths = {
    "HILLPDK": os.path.join(base, "HILLPDK_LIME_rate_tpha_pH55_clip_ndvi35_bias.tif"),
    "LIGHTNING_TREE": os.path.join(base, "LIGHTNING_TREE_LIME_rate_tpha_pH55_clip_ndvi35_bias.tif"),
}

for name, path in paths.items():
    with rasterio.open(path) as src:
        data = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        # pixel area in hectares
        px_w = src.transform.a
        px_h = -src.transform.e
        area_ha = (px_w * px_h) / 10000.0

        total_t = np.nansum(data) * area_ha
        mean_rate = np.nanmean(data)
        valid_px = np.count_nonzero(~np.isnan(data))
        total_area_ha = valid_px * area_ha

        print(f"{name}: Total lime = {total_t:.2f} t; Area = {total_area_ha:.2f} ha; Mean rate = {mean_rate:.2f} t/ha")
