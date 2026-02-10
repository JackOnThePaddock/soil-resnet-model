import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject
from rasterio.enums import Resampling
import shapefile
from shapely.geometry import shape, mapping

BASE_DIR = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp"
BOUNDARIES = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SpeirsBoundaries\boundaries\boundaries.shp"
NDVI_SRC = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\Imagery\Bare Imagery Composite\GA Barest Earth (Sentinel-2) clip.tiff"
OUT_DIR = os.path.join(BASE_DIR, "clipped_ndvi35")
THRESH = 0.35
NODATA = -9999.0


def load_boundaries():
    reader = shapefile.Reader(BOUNDARIES)
    fields = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
    name_idx = fields.index("FIELD_NAME")
    geoms = {}
    for sr in reader.shapeRecords():
        name = str(sr.record[name_idx]).strip().upper()
        geoms[name] = shape(sr.shape.__geo_interface__)
    return geoms


def match_geom(fname, geoms):
    if fname.startswith("400_"):
        key = "400"
    elif fname.startswith("HILLPDK_"):
        key = "HILLPDK"
    elif fname.startswith("LIGHTNING_TREE_"):
        key = "LIGHTNING TREE"
    else:
        return None
    return geoms.get(key)


def ndvi_mask(dst_shape, dst_transform, dst_crs, ndvi_src_path):
    dst_red = np.empty(dst_shape, dtype=np.float32)
    dst_nir = np.empty(dst_shape, dtype=np.float32)
    with rasterio.open(ndvi_src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_red,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=rasterio.band(src, 7),
            destination=dst_nir,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    ndvi = (dst_nir - dst_red) / (dst_nir + dst_red)
    mask_keep = np.isfinite(ndvi) & (ndvi <= THRESH)
    return mask_keep


def process_raster(path, geom):
    with rasterio.open(path) as src:
        data, transform = mask(src, [mapping(geom)], crop=True, nodata=NODATA)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
            "nodata": NODATA,
        })
        mask_keep = ndvi_mask((data.shape[1], data.shape[2]), transform, src.crs, NDVI_SRC)
        out = data[0].astype(np.float32)
        out[(~mask_keep) | (data[0] == NODATA)] = NODATA
    return out, out_meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    geoms = load_boundaries()
    inputs = [f for f in os.listdir(BASE_DIR) if f.endswith("_rf_bestbands.tif")]
    if not inputs:
        print("No input rasters found.")
        return
    for fname in inputs:
        geom = match_geom(fname, geoms)
        if geom is None:
            print(f"Skip (no paddock match): {fname}")
            continue
        in_path = os.path.join(BASE_DIR, fname)
        out_name = fname.replace(".tif", "_clip_ndvi35.tif")
        out_path = os.path.join(OUT_DIR, out_name)
        out_data, out_meta = process_raster(in_path, geom)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_data, 1)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
