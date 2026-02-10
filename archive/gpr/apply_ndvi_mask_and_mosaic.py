import re
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
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\gpr_alphaearth")
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

    pattern = re.compile(r"^(.*)_(pH|CEC)_gpr_(mean|std)\.tif$", re.IGNORECASE)
    for tif in pred_dir.glob("*.tif"):
        if not pattern.match(tif.name):
            continue
        out_path = masked_dir / tif.name
        apply_mask_to_raster(tif, mask_path, out_path)

    groups = {}
    for tif in masked_dir.glob("*.tif"):
        match = pattern.match(tif.name)
        if not match:
            continue
        suffix = f"{match.group(2).upper()}_{match.group(3).lower()}"
        groups.setdefault(suffix, []).append(tif)

    for suffix, files in groups.items():
        out_path = mosaic_dir / f"farm_{suffix}_gpr_ndvi35.tif"
        mosaic_group(sorted(files), out_path)

    print(f"Masked rasters: {masked_dir}")
    print(f"Farm mosaics: {mosaic_dir}")


if __name__ == "__main__":
    main()
