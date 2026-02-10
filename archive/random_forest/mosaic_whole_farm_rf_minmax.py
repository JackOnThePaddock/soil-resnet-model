import re
from pathlib import Path

import rasterio
import rasterio.merge


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
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_minmax_calibration")
    in_dir = base_dir / "rfe_best_minmax_calibrated_ndvi35"
    out_dir = base_dir / "farm_mosaics_ndvi35"
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^(.*)_(pH|CEC)_5yr_rfe_best\.tif$", re.IGNORECASE)
    groups = {}
    for tif in in_dir.glob("*.tif"):
        match = pattern.match(tif.name)
        if not match:
            continue
        var = match.group(2).upper()
        groups.setdefault(var, []).append(tif)

    for var, files in groups.items():
        out_path = out_dir / f"farm_{var}_rf_rfe_best_minmax_ndvi35.tif"
        mosaic_group(sorted(files), out_path)

    print(f"Farm mosaics saved to: {out_dir}")


if __name__ == "__main__":
    main()
