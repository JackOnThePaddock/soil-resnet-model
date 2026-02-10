import os
import numpy as np
import rasterio
import math

base = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\clipped_ndvi35_bias_corrected"
paths = {
    "HILLPDK": os.path.join(base, "HILLPDK_LIME_rate_tpha_pH55_clip_ndvi35_bias.tif"),
    "LIGHTNING_TREE": os.path.join(base, "LIGHTNING_TREE_LIME_rate_tpha_pH55_clip_ndvi35_bias.tif"),
}

def meters_per_degree(lat_deg):
    lat = math.radians(lat_deg)
    # WGS84 approximations
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2*lat) + 1.175 * math.cos(4*lat) - 0.0023 * math.cos(6*lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3*lat) + 0.118 * math.cos(5*lat)
    return m_per_deg_lat, m_per_deg_lon

for name, path in paths.items():
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        height, width = arr.shape

        # approximate pixel area in ha using raster center latitude
        center_row = height / 2.0
        center_lat = src.transform.f + src.transform.e * center_row
        m_per_deg_lat, m_per_deg_lon = meters_per_degree(center_lat)

        px_w_deg = src.transform.a
        px_h_deg = abs(src.transform.e)
        px_w_m = px_w_deg * m_per_deg_lon
        px_h_m = px_h_deg * m_per_deg_lat
        area_ha = (px_w_m * px_h_m) / 10000.0

        valid = arr.compressed()
        total_t = float(valid.sum() * area_ha)
        total_area_ha = float(valid.size * area_ha)
        mean_rate = float(valid.mean()) if valid.size else float('nan')

        print(f"{name}: Total lime = {total_t:.2f} t; Area = {total_area_ha:.2f} ha; Mean rate = {mean_rate:.2f} t/ha")
