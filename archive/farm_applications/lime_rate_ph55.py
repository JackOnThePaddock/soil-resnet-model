import os
import numpy as np
import rasterio

base = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\clipped_ndvi35_bias_corrected"

pairs = {
    "HILLPDK": {
        "ph": os.path.join(base, "HILLPDK_pH_rf_bestbands_3x3_clip_ndvi35_bias.tif"),
        "cec": os.path.join(base, "HILLPDK_CEC_rf_bestbands_3x3_clip_ndvi35_bias.tif"),
    },
    "LIGHTNING_TREE": {
        "ph": os.path.join(base, "LIGHTNING_TREE_pH_rf_bestbands_3x3_clip_ndvi35_bias.tif"),
        "cec": os.path.join(base, "LIGHTNING_TREE_CEC_rf_bestbands_3x3_clip_ndvi35_bias.tif"),
    },
}

TARGET_PH = 5.5
NV = 0.95
EF = 0.8
PHBC_MULT = 0.13  # pHBC = 0.13 * CEC
MIN_TRIGGER = 0.5
MAX_CAP = 4.0

for paddock, paths in pairs.items():
    ph_path = paths["ph"]
    cec_path = paths["cec"]

    if not os.path.exists(ph_path):
        raise FileNotFoundError(ph_path)
    if not os.path.exists(cec_path):
        raise FileNotFoundError(cec_path)

    with rasterio.open(ph_path) as ph_src, rasterio.open(cec_path) as cec_src:
        if ph_src.shape != cec_src.shape or ph_src.transform != cec_src.transform:
            raise ValueError(f"Raster mismatch for {paddock}")

        ph = ph_src.read(1).astype("float32")
        cec = cec_src.read(1).astype("float32")

        ph_nodata = ph_src.nodata
        cec_nodata = cec_src.nodata

        nodata = ph_nodata if ph_nodata is not None else cec_nodata
        if nodata is None:
            nodata = -9999.0

        mask = np.zeros(ph.shape, dtype=bool)
        if ph_nodata is not None:
            mask |= ph == ph_nodata
        if cec_nodata is not None:
            mask |= cec == cec_nodata
        mask |= np.isnan(ph) | np.isnan(cec)

        ph_deficit = (TARGET_PH - ph)
        phbc = PHBC_MULT * cec
        rate = (ph_deficit * phbc) / (NV * EF)

        rate = np.where(rate < MIN_TRIGGER, 0, rate)
        rate = np.where(rate > MAX_CAP, MAX_CAP, rate)
        rate = rate.astype("float32")
        rate[mask] = nodata

        out_meta = ph_src.meta.copy()
        out_meta.update({
            "dtype": "float32",
            "nodata": nodata,
            "compress": "deflate",
        })

        out_path = os.path.join(base, f"{paddock}_LIME_rate_tpha_pH55_clip_ndvi35_bias.tif")
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(rate, 1)

        print(f"Wrote {out_path}")
