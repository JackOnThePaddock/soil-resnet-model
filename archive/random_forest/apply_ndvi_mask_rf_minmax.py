from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling


def build_ndvi_mask(src_path: Path, out_path: Path) -> Path:
    if out_path.exists():
        return out_path

    with rasterio.open(src_path) as src:
        red = src.read(1).astype(np.float32)
        nir = src.read(7).astype(np.float32)
        denom = nir + red
        ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
        mask = np.where(ndvi < 0.35, 1, 0).astype(np.uint8)

        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", nodata=0, compress="LZW")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mask, 1)

    return out_path


def reproject_mask(mask_path: Path, dst_profile: dict) -> np.ndarray:
    with rasterio.open(mask_path) as src:
        mask_src = src.read(1)
        dst = np.zeros((dst_profile["height"], dst_profile["width"]), dtype=np.uint8)
        rasterio.warp.reproject(
            source=mask_src,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
        return dst


def apply_mask_to_raster(src_path: Path, mask_path: Path, out_path: Path) -> None:
    with rasterio.open(src_path) as src:
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        profile = src.profile.copy()
        profile.update(nodata=nodata, compress="LZW")

        mask = reproject_mask(
            mask_path,
            {"height": src.height, "width": src.width, "transform": src.transform, "crs": src.crs},
        )
        out = data.copy()
        out[mask == 0] = nodata

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)


def main() -> None:
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_minmax_calibration")
    in_dir = base_dir / "rfe_best_minmax_calibrated"
    out_dir = base_dir / "rfe_best_minmax_calibrated_ndvi35"
    out_dir.mkdir(parents=True, exist_ok=True)

    ndvi_src = Path(
        r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\Imagery\Bare Imagery Composite\GA Barest Earth (Sentinel-2) clip.tiff"
    )
    if not ndvi_src.exists():
        raise FileNotFoundError(ndvi_src)

    mask_path = base_dir / "ndvi_mask_035.tif"
    build_ndvi_mask(ndvi_src, mask_path)

    for tif in in_dir.glob("*.tif"):
        out_path = out_dir / tif.name
        apply_mask_to_raster(tif, mask_path, out_path)

    print(f"Masked outputs: {out_dir}")


if __name__ == "__main__":
    main()
