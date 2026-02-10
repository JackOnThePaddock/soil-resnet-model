"""Tests for bare-earth raster sampling utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.features.bare_earth import sample_bare_earth_at_points


def test_sample_bare_earth_at_points(tmp_path):
    raster_path = tmp_path / "be.tif"
    transform = from_origin(149.0, -35.0, 0.01, 0.01)  # lon/lat grid
    arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    arr2 = np.array([[10, 20], [30, 40]], dtype=np.float32)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(arr1, 1)
        dst.write(arr2, 2)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "nir")

    points = pd.DataFrame(
        {
            "lon": [149.005, 149.015],
            "lat": [-35.005, -35.015],
        }
    )
    out = sample_bare_earth_at_points(raster_path=raster_path, points_df=points)
    assert list(out.columns) == ["be_red", "be_nir"]
    assert out.shape == (2, 2)
    assert np.isfinite(out.values).all()
