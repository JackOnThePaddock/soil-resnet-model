"""Lime application rate calculation from soil pH and CEC rasters."""

from pathlib import Path
from typing import Union

import numpy as np


def calculate_lime_rate(
    ph_path: Union[str, Path],
    cec_path: Union[str, Path],
    output_path: Union[str, Path],
    target_ph: float = 5.5,
    neutralizing_value: float = 0.95,
    efficiency_factor: float = 0.80,
    phbc_multiplier: float = 0.13,
    min_trigger: float = 0.5,
    max_cap: float = 4.0,
    nodata: float = -9999.0,
) -> Path:
    """
    Calculate lime rate (t/ha) from pH and CEC rasters.

    Formula: rate = (pH_deficit * pHBC) / (NV * EF)
    Where pHBC = phbc_multiplier * CEC
    """
    import rasterio

    ph_path, cec_path, output_path = Path(ph_path), Path(cec_path), Path(output_path)

    with rasterio.open(ph_path) as ph_src, rasterio.open(cec_path) as cec_src:
        ph = ph_src.read(1).astype("float32")
        cec = cec_src.read(1).astype("float32")

        mask = np.isnan(ph) | np.isnan(cec)
        if ph_src.nodata is not None:
            mask |= ph == ph_src.nodata
        if cec_src.nodata is not None:
            mask |= cec == cec_src.nodata

        ph_deficit = target_ph - ph
        phbc = phbc_multiplier * cec
        rate = (ph_deficit * phbc) / (neutralizing_value * efficiency_factor)
        rate = np.where(rate < min_trigger, 0, rate)
        rate = np.where(rate > max_cap, max_cap, rate)
        rate = rate.astype("float32")
        rate[mask] = nodata

        out_meta = ph_src.meta.copy()
        out_meta.update(dtype="float32", nodata=nodata, compress="deflate")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(rate, 1)

    print(f"Lime rate: {output_path}")
    return output_path
