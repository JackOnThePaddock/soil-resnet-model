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
        arr = src.read(1, masked=True)
        nodata = src.nodata
        print(name)
        print(" nodata:", nodata)
        print(" dtype:", arr.dtype)
        print(" shape:", arr.shape)
        print(" masked count:", np.count_nonzero(arr.mask))
        print(" valid count:", np.count_nonzero(~arr.mask))
        if np.count_nonzero(~arr.mask) > 0:
            vals = arr.compressed()
            print(" min:", float(vals.min()))
            print(" max:", float(vals.max()))
            print(" mean:", float(vals.mean()))
        print("--")
