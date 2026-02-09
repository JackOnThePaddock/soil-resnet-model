"""NDVI masking for soil prediction rasters."""

from pathlib import Path
from typing import Union

import numpy as np


def apply_ndvi_mask(
    input_path: Union[str, Path],
    ndvi_path: Union[str, Path],
    output_path: Union[str, Path],
    threshold: float = 0.35,
    nodata: float = -9999.0,
) -> Path:
    """Mask prediction raster by NDVI threshold (keep pixels where NDVI < threshold)."""
    import rasterio

    input_path, ndvi_path, output_path = Path(input_path), Path(ndvi_path), Path(output_path)

    with rasterio.open(ndvi_path) as ndvi_src:
        ndvi = ndvi_src.read(1)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        data = src.read()

        # Mask where NDVI >= threshold (dense vegetation, unreliable bare-soil predictions)
        veg_mask = ndvi >= threshold
        for band in range(data.shape[0]):
            data[band][veg_mask] = nodata

        profile.update(nodata=nodata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    return output_path
