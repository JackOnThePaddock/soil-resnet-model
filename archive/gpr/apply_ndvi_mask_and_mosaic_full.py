from pathlib import Path

import numpy as np
import rasterio
import rasterio.merge
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


def mosaic_group(paths: list[Path], out_path: Path) -> None:
    sources = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = rasterio.merge.merge(sources)
        meta = sources[0].meta.copy()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "LZW",
            }
        )
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
    finally:
        for src in sources:
            src.close()


def main() -> None:
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\gpr_alphaearth_full")
    pred_dir = base_dir / "predictions_clipped"
    masked_dir = base_dir / "predictions_clipped_ndvi35"
    mosaic_dir = base_dir / "farm_mosaics_ndvi35"
    masked_dir.mkdir(parents=True, exist_ok=True)
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    ndvi_src = Path(
        r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\Imagery\Bare Imagery Composite\GA Barest Earth (Sentinel-2) clip.tiff"
    )
    if not ndvi_src.exists():
        raise FileNotFoundError(ndvi_src)

    mask_path = base_dir / "ndvi_mask_035.tif"
    build_ndvi_mask(ndvi_src, mask_path)

    for var_dir in pred_dir.iterdir():
        if not var_dir.is_dir():
            continue
        out_var = masked_dir / var_dir.name
        out_var.mkdir(parents=True, exist_ok=True)
        for tif in var_dir.glob("*.tif"):
            out_path = out_var / tif.name
            apply_mask_to_raster(tif, mask_path, out_path)

    for var_dir in masked_dir.iterdir():
        if not var_dir.is_dir():
            continue
        mean_files = sorted(var_dir.glob("*_gpr_mean.tif"))
        std_files = sorted(var_dir.glob("*_gpr_std.tif"))
        if mean_files:
            out_mean = mosaic_dir / f"farm_{var_dir.name}_mean_gpr_ndvi35.tif"
            mosaic_group(mean_files, out_mean)
        if std_files:
            out_std = mosaic_dir / f"farm_{var_dir.name}_std_gpr_ndvi35.tif"
            mosaic_group(std_files, out_std)

    print(f"Masked rasters: {masked_dir}")
    print(f"Farm mosaics: {mosaic_dir}")


if __name__ == "__main__":
    main()
