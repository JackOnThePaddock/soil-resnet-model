"""Farm-wide mosaic creation from per-paddock prediction rasters."""

from pathlib import Path
from typing import Dict, List

import numpy as np

NODATA = -9999.0


def create_mosaics(
    pred_dir: str, mosaic_dir: str, target_names: List[str], paddock_names: List[str],
    file_pattern: str = "{paddock}_5yr_median_{target}.tif", nodata: float = NODATA,
) -> Dict[str, Path]:
    """Merge per-paddock prediction rasters into farm-wide mosaics."""
    import rasterio
    from rasterio.merge import merge

    pred_dir, mosaic_dir = Path(pred_dir), Path(mosaic_dir)
    mosaic_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}

    for target in target_names:
        pred_files = [pred_dir / file_pattern.format(paddock=n, target=target)
                      for n in paddock_names]
        pred_files = [p for p in pred_files if p.exists()]
        if not pred_files:
            continue

        srcs = [rasterio.open(p) for p in pred_files]
        mosaic_data, out_transform = merge(srcs, nodata=nodata)
        out_meta = srcs[0].meta.copy()
        out_meta.update(height=mosaic_data.shape[1], width=mosaic_data.shape[2],
                        transform=out_transform, nodata=nodata, count=mosaic_data.shape[0],
                        dtype="float32", compress="deflate")

        out_path = mosaic_dir / f"farm_mosaic_{target}.tif"
        with rasterio.open(out_path, "w", **out_meta) as dest:
            for b in range(mosaic_data.shape[0]):
                dest.write(mosaic_data[b].astype(np.float32), b + 1)
        for s in srcs:
            s.close()
        output_files[target] = out_path
        print(f"  Mosaic: {out_path.name} ({len(pred_files)} paddocks)")

    return output_files


def print_summary(
    pred_dir: str, target_names: List[str], paddock_names: List[str],
    file_pattern: str = "{paddock}_5yr_median_{target}.tif", nodata: float = NODATA,
) -> None:
    """Print summary statistics per paddock per target."""
    import rasterio

    pred_dir = Path(pred_dir)
    header = f"{'Paddock':<22}"
    for t in target_names:
        header += f"{'  ' + t.upper():>10} {'(unc)':>7}"
    print(header)
    print("-" * len(header))

    for name in paddock_names:
        row = f"{name:<22}"
        for target in target_names:
            p = pred_dir / file_pattern.format(paddock=name, target=target)
            if not p.exists():
                row += f"{'N/A':>10} {'N/A':>7}"
                continue
            with rasterio.open(p) as src:
                mean_band, std_band = src.read(1), src.read(2)
                valid = mean_band != nodata
                if valid.sum() > 0:
                    row += f"{np.mean(mean_band[valid]):10.2f} {np.mean(std_band[valid]):7.3f}"
                else:
                    row += f"{'empty':>10} {'':>7}"
        print(row)
