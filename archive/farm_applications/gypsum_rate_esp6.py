import os
import numpy as np
import rasterio

base = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\clipped_ndvi35_bias_corrected"

pairs = {
    "HILLPDK": {
        "cec": os.path.join(base, "HILLPDK_CEC_rf_bestbands_3x3_clip_ndvi35_bias.tif"),
        "esp": os.path.join(base, "HILLPDK_ESP_rf_bestbands_3x3_clip_ndvi35_bias_nonneg.tif"),
    },
    "LIGHTNING_TREE": {
        "cec": os.path.join(base, "LIGHTNING_TREE_CEC_rf_bestbands_3x3_clip_ndvi35_bias.tif"),
        "esp": os.path.join(base, "LIGHTNING_TREE_ESP_rf_bestbands_3x3_clip_ndvi35_bias_nonneg.tif"),
    },
}

target_esp = 6.0
factor = 0.06

for paddock, paths in pairs.items():
    cec_path = paths["cec"]
    esp_path = paths["esp"]

    if not os.path.exists(cec_path):
        raise FileNotFoundError(cec_path)
    if not os.path.exists(esp_path):
        raise FileNotFoundError(esp_path)

    with rasterio.open(cec_path) as cec_src, rasterio.open(esp_path) as esp_src:
        if cec_src.shape != esp_src.shape or cec_src.transform != esp_src.transform:
            raise ValueError(f"Raster mismatch for {paddock}")

        cec = cec_src.read(1).astype("float32")
        esp = esp_src.read(1).astype("float32")

        cec_nodata = cec_src.nodata
        esp_nodata = esp_src.nodata

        nodata = esp_nodata if esp_nodata is not None else cec_nodata
        if nodata is None:
            nodata = -9999.0

        mask = np.zeros(cec.shape, dtype=bool)
        if cec_nodata is not None:
            mask |= cec == cec_nodata
        if esp_nodata is not None:
            mask |= esp == esp_nodata
        mask |= np.isnan(cec) | np.isnan(esp)

        rate = (esp - target_esp) * cec * factor
        rate = np.where(rate < 0, 0, rate)
        rate = rate.astype("float32")
        rate[mask] = nodata

        out_meta = cec_src.meta.copy()
        out_meta.update({
            "dtype": "float32",
            "nodata": nodata,
            "compress": "deflate",
        })

        out_path = os.path.join(base, f"{paddock}_GYPSUM_rate_tpha_ESP6_clip_ndvi35_bias.tif")
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(rate, 1)

        print(f"Wrote {out_path}")
