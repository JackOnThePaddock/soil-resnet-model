import os
import glob
import math
import csv
import numpy as np
import rasterio

base_1x1 = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\all_paddocks_1x1\clipped_ndvi35"
base_3x3 = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\clipped_ndvi35_bias_corrected"

bias_paddocks = {"400", "HILLPDK", "LIGHTNING_TREE"}

TARGET_PH = 5.5
NV = 0.95
EF = 0.8
PHBC_MULT = 0.13
MIN_TRIGGER = 0.5
MAX_CAP = 4.0

TARGET_ESP = 6.0
GYPSUM_FACTOR = 0.06

out_csv = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\lime_gypsum_totals_all_paddocks.csv"

# helpers

def meters_per_degree(lat_deg):
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2*lat) + 1.175 * math.cos(4*lat) - 0.0023 * math.cos(6*lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3*lat) + 0.118 * math.cos(5*lat)
    return m_per_deg_lat, m_per_deg_lon


def calc_totals(arr, transform):
    # arr is masked array
    height, width = arr.shape
    center_lat = transform.f + transform.e * (height / 2.0)
    m_per_deg_lat, m_per_deg_lon = meters_per_degree(center_lat)

    px_w_deg = transform.a
    px_h_deg = abs(transform.e)
    px_w_m = px_w_deg * m_per_deg_lon
    px_h_m = px_h_deg * m_per_deg_lat
    area_ha = (px_w_m * px_h_m) / 10000.0

    valid = arr.compressed()
    total_t = float(valid.sum() * area_ha) if valid.size else 0.0
    area_ha_total = float(valid.size * area_ha) if valid.size else 0.0
    mean_rate = float(valid.mean()) if valid.size else 0.0
    return total_t, area_ha_total, mean_rate

# discover paddocks from 1x1 pH files
ph_files = glob.glob(os.path.join(base_1x1, "*_pH_rf_bestbands_1x1_clip_ndvi35.tif"))
paddocks = sorted({os.path.basename(p).replace("_pH_rf_bestbands_1x1_clip_ndvi35.tif", "") for p in ph_files})

rows = []
for paddock in paddocks:
    if paddock in bias_paddocks:
        source = "3x3_bias"
        ph_path = os.path.join(base_3x3, f"{paddock}_pH_rf_bestbands_3x3_clip_ndvi35_bias.tif")
        cec_path = os.path.join(base_3x3, f"{paddock}_CEC_rf_bestbands_3x3_clip_ndvi35_bias.tif")
        esp_path = os.path.join(base_3x3, f"{paddock}_ESP_rf_bestbands_3x3_clip_ndvi35_bias_nonneg.tif")
    else:
        source = "1x1_ndvi"
        ph_path = os.path.join(base_1x1, f"{paddock}_pH_rf_bestbands_1x1_clip_ndvi35.tif")
        cec_path = os.path.join(base_1x1, f"{paddock}_CEC_rf_bestbands_1x1_clip_ndvi35.tif")
        esp_path = os.path.join(base_1x1, f"{paddock}_ESP_rf_bestbands_1x1_clip_ndvi35_nonneg.tif")

    if not (os.path.exists(ph_path) and os.path.exists(cec_path) and os.path.exists(esp_path)):
        print(f"Skipping {paddock} (missing rasters)")
        continue

    with rasterio.open(ph_path) as ph_src, rasterio.open(cec_path) as cec_src, rasterio.open(esp_path) as esp_src:
        if ph_src.shape != cec_src.shape or ph_src.transform != cec_src.transform:
            raise ValueError(f"Raster mismatch pH/CEC for {paddock}")
        if ph_src.shape != esp_src.shape or ph_src.transform != esp_src.transform:
            raise ValueError(f"Raster mismatch pH/ESP for {paddock}")

        ph = ph_src.read(1, masked=True).astype("float32")
        cec = cec_src.read(1, masked=True).astype("float32")
        esp = esp_src.read(1, masked=True).astype("float32")

        mask = ph.mask | cec.mask | esp.mask
        ph = np.ma.array(ph.data, mask=mask)
        cec = np.ma.array(cec.data, mask=mask)
        esp = np.ma.array(esp.data, mask=mask)

        # Lime rate
        ph_deficit = TARGET_PH - ph
        phbc = PHBC_MULT * cec
        lime = (ph_deficit * phbc) / (NV * EF)
        lime = np.ma.where(lime < MIN_TRIGGER, 0, lime)
        lime = np.ma.where(lime > MAX_CAP, MAX_CAP, lime)
        lime = np.ma.array(lime, mask=mask)

        # Gypsum rate
        gypsum = (esp - TARGET_ESP) * cec * GYPSUM_FACTOR
        gypsum = np.ma.where(gypsum < 0, 0, gypsum)
        gypsum = np.ma.array(gypsum, mask=mask)

        lime_total_t, area_ha, lime_mean = calc_totals(lime, ph_src.transform)
        gypsum_total_t, _, gypsum_mean = calc_totals(gypsum, ph_src.transform)

        rows.append({
            "paddock": paddock,
            "source": source,
            "area_ha": round(area_ha, 2),
            "lime_total_t": round(lime_total_t, 2),
            "lime_mean_tpha": round(lime_mean, 2),
            "gypsum_total_t": round(gypsum_total_t, 2),
            "gypsum_mean_tpha": round(gypsum_mean, 2),
        })

# write CSV
rows = sorted(rows, key=lambda r: r["paddock"])
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "paddock", "source", "area_ha", "lime_total_t", "lime_mean_tpha", "gypsum_total_t", "gypsum_mean_tpha"
    ])
    writer.writeheader()
    writer.writerows(rows)

# print totals
lime_sum = sum(r["lime_total_t"] for r in rows)
gypsum_sum = sum(r["gypsum_total_t"] for r in rows)
area_sum = sum(r["area_ha"] for r in rows)

print(f"Wrote {out_csv}")
print(f"TOTAL: Area {area_sum:.2f} ha | Lime {lime_sum:.2f} t | Gypsum {gypsum_sum:.2f} t")
