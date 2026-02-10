import os
import rasterio

base = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\clipped_ndvi35_bias_corrected"
paths = {
    "HILLPDK": os.path.join(base, "HILLPDK_LIME_rate_tpha_pH55_clip_ndvi35_bias.tif"),
    "LIGHTNING_TREE": os.path.join(base, "LIGHTNING_TREE_LIME_rate_tpha_pH55_clip_ndvi35_bias.tif"),
}

for name, path in paths.items():
    with rasterio.open(path) as src:
        print(name, src.transform)
        print("px w", src.transform.a, "px h", src.transform.e)
        print("crs", src.crs)
        print("--")
