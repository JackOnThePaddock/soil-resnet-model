"""Gypsum application rate calculation from ESP and CEC rasters."""

from pathlib import Path
from typing import Union

import numpy as np


def calculate_gypsum_rate(
    cec_path: Union[str, Path],
    esp_path: Union[str, Path],
    output_path: Union[str, Path],
    target_esp: float = 6.0,
    factor: float = 0.06,
    nodata: float = -9999.0,
) -> Path:
    """
    Calculate gypsum rate (t/ha) from CEC and ESP rasters.

    Formula: rate = (ESP - target_ESP) * CEC * factor
    Only applied where ESP > target_ESP.
    """
    import rasterio

    cec_path, esp_path, output_path = Path(cec_path), Path(esp_path), Path(output_path)

    with rasterio.open(cec_path) as cec_src, rasterio.open(esp_path) as esp_src:
        cec = cec_src.read(1).astype("float32")
        esp = esp_src.read(1).astype("float32")

        mask = np.isnan(cec) | np.isnan(esp)
        if cec_src.nodata is not None:
            mask |= cec == cec_src.nodata
        if esp_src.nodata is not None:
            mask |= esp == esp_src.nodata

        rate = (esp - target_esp) * cec * factor
        rate = np.where(rate < 0, 0, rate).astype("float32")
        rate[mask] = nodata

        out_meta = cec_src.meta.copy()
        out_meta.update(dtype="float32", nodata=nodata, compress="deflate")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(rate, 1)

    print(f"Gypsum rate: {output_path}")
    return output_path
