"""Utilities for GA Barest Earth (Sentinel-2) ingestion via DEA OWS services."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time

import numpy as np
import pandas as pd
import requests

DEA_OWS_BASE_URL = "https://ows.dea.ga.gov.au/"
DEFAULT_COVERAGE_ID = "s2_barest_earth"
DEFAULT_CRS = "EPSG:3577"
DEFAULT_BAREST_EARTH_BANDS = (
    "red",
    "green",
    "blue",
    "red_edge_1",
    "red_edge_2",
    "red_edge_3",
    "nir",
    "nir_2",
    "swir1",
    "swir2",
)


def discover_barest_earth_products(base_url: str = DEA_OWS_BASE_URL) -> Dict[str, str]:
    """
    Discover GA Barest Earth identifiers from WMS/WCS capabilities.

    Returns:
        Dict with optional keys: `wms_layer`, `wcs_coverage`.
    """
    result: Dict[str, str] = {}

    # WMS layer discovery.
    wms_url = (
        f"{base_url}?service=WMS&version=1.3.0&request=GetCapabilities"
    )
    try:
        wms_xml = requests.get(wms_url, timeout=60).text
        root = ET.fromstring(wms_xml)
        ns = {"wms": "http://www.opengis.net/wms"}

        def _iter_layers(node: ET.Element) -> Iterable[ET.Element]:
            for lyr in node.findall("wms:Layer", ns):
                yield lyr
                yield from _iter_layers(lyr)

        root_layer = root.find(".//wms:Capability/wms:Layer", ns)
        if root_layer is not None:
            for layer in _iter_layers(root_layer):
                name_el = layer.find("wms:Name", ns)
                title_el = layer.find("wms:Title", ns)
                title = (title_el.text or "").lower() if title_el is not None else ""
                if (
                    "bare" in title
                    and "earth" in title
                    and "sentinel" in title
                    and name_el is not None
                ):
                    result["wms_layer"] = name_el.text or ""
                    break
    except Exception:
        pass

    # WCS coverage discovery.
    wcs_url = (
        f"{base_url}?service=WCS&version=2.0.1&request=GetCapabilities"
    )
    try:
        wcs_xml = requests.get(wcs_url, timeout=60).text
        root = ET.fromstring(wcs_xml)
        ns = {
            "wcs": "http://www.opengis.net/wcs/2.0",
            "ows": "http://www.opengis.net/ows/2.0",
        }
        for cs in root.findall(".//wcs:Contents/wcs:CoverageSummary", ns):
            cov_el = cs.find("wcs:CoverageId", ns)
            title_el = cs.find("ows:Title", ns)
            cov = cov_el.text if cov_el is not None else ""
            title = (title_el.text or "").lower() if title_el is not None else ""
            if "bare" in title and "earth" in title and "sentinel" in title:
                result["wcs_coverage"] = cov
                break
    except Exception:
        pass

    return result


def _transform_bbox_4326_to_3577(
    bbox_lonlat: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Transform lon/lat bbox (EPSG:4326) to GDA94 / Australian Albers (EPSG:3577)."""
    from pyproj import Transformer

    west, south, east, north = bbox_lonlat
    transformer = Transformer.from_crs("EPSG:4326", DEFAULT_CRS, always_xy=True)
    x1, y1 = transformer.transform(west, south)
    x2, y2 = transformer.transform(east, north)
    xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
    ymin, ymax = (y1, y2) if y1 <= y2 else (y2, y1)
    return xmin, ymin, xmax, ymax


def _build_rangesubset(bands: Optional[Sequence[str]]) -> Optional[str]:
    """Build WCS range subset parameter."""
    if not bands:
        return None
    return ",".join(str(b).strip() for b in bands if str(b).strip())


def _wcs_getcoverage_params(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    coverage_id: str,
    bands: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """Build WCS 2.0.1 GetCoverage query parameters for projected bounds."""
    params: List[Tuple[str, str]] = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("request", "GetCoverage"),
        ("coverageId", coverage_id),
        ("format", "image/geotiff"),
        ("subset", f"x({xmin},{xmax})"),
        ("subset", f"y({ymin},{ymax})"),
    ]
    rangesubset = _build_rangesubset(bands)
    if rangesubset:
        params.append(("rangesubset", rangesubset))
    return params


def download_barest_earth_geotiff(
    bbox_lonlat: Tuple[float, float, float, float],
    output_path: Path,
    base_url: str = DEA_OWS_BASE_URL,
    coverage_id: str = DEFAULT_COVERAGE_ID,
    bands: Optional[Sequence[str]] = None,
    timeout: int = 600,
) -> Path:
    """
    Download GA Barest Earth Sentinel-2 coverage via WCS 2.0.1 as GeoTIFF.

    Args:
        bbox_lonlat: (west, south, east, north) in EPSG:4326.
        output_path: Output GeoTIFF path.
        coverage_id: WCS coverage id (default `s2_barest_earth`).
        bands: Optional range subset names (e.g., `["red","green","blue","nir"]`).
    """
    xmin, ymin, xmax, ymax = _transform_bbox_4326_to_3577(bbox_lonlat)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = _wcs_getcoverage_params(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        coverage_id=coverage_id,
        bands=bands,
    )

    with requests.get(base_url, params=params, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return output_path


def sample_bare_earth_points_via_wcs(
    points_df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    base_url: str = DEA_OWS_BASE_URL,
    coverage_id: str = DEFAULT_COVERAGE_ID,
    bands: Sequence[str] = DEFAULT_BAREST_EARTH_BANDS,
    buffer_m: float = 6.0,
    max_workers: int = 12,
    timeout: int = 120,
    retries: int = 3,
    output_prefix: str = "be_",
) -> pd.DataFrame:
    """
    Sample bare-earth values at point locations directly via WCS chips.

    This avoids downloading continent-scale rasters when points are spatially
    sparse (e.g., national soil training points).
    """
    from pyproj import Transformer
    from rasterio.io import MemoryFile

    if lon_col not in points_df.columns or lat_col not in points_df.columns:
        raise ValueError(f"Expected columns '{lon_col}' and '{lat_col}' in points dataframe")
    if not bands:
        raise ValueError("At least one band must be provided")

    lon = pd.to_numeric(points_df[lon_col], errors="coerce")
    lat = pd.to_numeric(points_df[lat_col], errors="coerce")
    if lon.isna().any() or lat.isna().any():
        raise ValueError("Found non-numeric or missing lon/lat values")

    lon_arr = lon.values.astype(np.float64)
    lat_arr = lat.values.astype(np.float64)
    n = len(points_df)
    n_bands = len(bands)

    # Deduplicate exact coordinate pairs to avoid repeated network calls.
    keys = [f"{x:.10f},{y:.10f}" for x, y in zip(lon_arr, lat_arr)]
    key_to_rows: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        key_to_rows.setdefault(key, []).append(i)
    unique_keys = list(key_to_rows.keys())
    unique_lonlat = [tuple(map(float, k.split(","))) for k in unique_keys]

    transformer = Transformer.from_crs("EPSG:4326", DEFAULT_CRS, always_xy=True)
    unique_xy = [transformer.transform(x, y) for x, y in unique_lonlat]
    unique_values = np.full((len(unique_keys), n_bands), np.nan, dtype=np.float32)

    def _fetch_vector(x: float, y: float) -> np.ndarray:
        xmin, xmax = x - float(buffer_m), x + float(buffer_m)
        ymin, ymax = y - float(buffer_m), y + float(buffer_m)
        params = _wcs_getcoverage_params(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            coverage_id=coverage_id,
            bands=bands,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()

                with MemoryFile(response.content) as memfile:
                    with memfile.open() as src:
                        arr = src.read(masked=True).astype(np.float32)
                        if arr.ndim != 3 or arr.shape[0] == 0:
                            raise RuntimeError(f"Unexpected raster shape {arr.shape}")

                        # Robust against small nodata artifacts in returned chips.
                        flat = arr.reshape(arr.shape[0], -1)
                        vals = np.ma.median(flat, axis=1).filled(np.nan).astype(np.float32)
                        if not np.isfinite(vals).any():
                            center = arr[:, arr.shape[1] // 2, arr.shape[2] // 2]
                            vals = np.asarray(center.filled(np.nan), dtype=np.float32)
                        if len(vals) >= n_bands:
                            return vals[:n_bands]

                        out = np.full(n_bands, np.nan, dtype=np.float32)
                        out[: len(vals)] = vals
                        return out
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * attempt)
                else:
                    break
        if last_error is not None:
            return np.full(n_bands, np.nan, dtype=np.float32)
        return np.full(n_bands, np.nan, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_fetch_vector, x, y): i
            for i, (x, y) in enumerate(unique_xy)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            unique_values[idx] = future.result()

    values = np.full((n, n_bands), np.nan, dtype=np.float32)
    for key_idx, key in enumerate(unique_keys):
        for row_idx in key_to_rows[key]:
            values[row_idx] = unique_values[key_idx]

    columns = [f"{output_prefix}{str(b).strip().lower()}" for b in bands]
    return pd.DataFrame(values, columns=columns, index=points_df.index)


def _center_pad_or_crop(
    arr: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Center pad/crop [C,H,W] array to target spatial size."""
    if arr.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {arr.shape}")
    c, h, w = arr.shape
    out = np.full((c, out_h, out_w), np.nan, dtype=np.float32)

    src_h0 = max(0, (h - out_h) // 2)
    src_w0 = max(0, (w - out_w) // 2)
    src_h1 = min(h, src_h0 + out_h)
    src_w1 = min(w, src_w0 + out_w)

    copy_h = src_h1 - src_h0
    copy_w = src_w1 - src_w0

    dst_h0 = max(0, (out_h - copy_h) // 2)
    dst_w0 = max(0, (out_w - copy_w) // 2)
    dst_h1 = dst_h0 + copy_h
    dst_w1 = dst_w0 + copy_w

    out[:, dst_h0:dst_h1, dst_w0:dst_w1] = arr[:, src_h0:src_h1, src_w0:src_w1]
    return out


def sample_bare_earth_chips_via_wcs(
    points_df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    base_url: str = DEA_OWS_BASE_URL,
    coverage_id: str = DEFAULT_COVERAGE_ID,
    bands: Sequence[str] = DEFAULT_BAREST_EARTH_BANDS,
    patch_size: int = 128,
    pixel_size_m: float = 10.0,
    max_workers: int = 8,
    timeout: int = 120,
    retries: int = 3,
) -> np.ndarray:
    """
    Fetch fixed-size bare-earth chips around point locations via DEA WCS.

    Returns:
        np.ndarray of shape [N, patch_size, patch_size, n_bands].
    """
    from pyproj import Transformer
    from rasterio.io import MemoryFile

    if lon_col not in points_df.columns or lat_col not in points_df.columns:
        raise ValueError(f"Expected columns '{lon_col}' and '{lat_col}' in points dataframe")
    if not bands:
        raise ValueError("At least one band must be provided")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")

    lon = pd.to_numeric(points_df[lon_col], errors="coerce")
    lat = pd.to_numeric(points_df[lat_col], errors="coerce")
    if lon.isna().any() or lat.isna().any():
        raise ValueError("Found non-numeric or missing lon/lat values")

    lon_arr = lon.values.astype(np.float64)
    lat_arr = lat.values.astype(np.float64)
    n = len(points_df)
    n_bands = len(bands)

    # Deduplicate exact coordinate pairs to avoid repeated network calls.
    keys = [f"{x:.10f},{y:.10f}" for x, y in zip(lon_arr, lat_arr)]
    key_to_rows: Dict[str, List[int]] = {}
    for i, key in enumerate(keys):
        key_to_rows.setdefault(key, []).append(i)
    unique_keys = list(key_to_rows.keys())
    unique_lonlat = [tuple(map(float, k.split(","))) for k in unique_keys]

    transformer = Transformer.from_crs("EPSG:4326", DEFAULT_CRS, always_xy=True)
    unique_xy = [transformer.transform(x, y) for x, y in unique_lonlat]
    unique_chips = np.full(
        (len(unique_keys), patch_size, patch_size, n_bands),
        np.nan,
        dtype=np.float32,
    )
    half_span = float(patch_size) * float(pixel_size_m) / 2.0

    def _fetch_chip(x: float, y: float) -> np.ndarray:
        xmin, xmax = x - half_span, x + half_span
        ymin, ymax = y - half_span, y + half_span
        params = _wcs_getcoverage_params(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            coverage_id=coverage_id,
            bands=bands,
        )

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()
                with MemoryFile(response.content) as memfile:
                    with memfile.open() as src:
                        arr = src.read(masked=True).astype(np.float32)
                        arr = np.ma.filled(arr, np.nan)
                        arr = _center_pad_or_crop(arr, out_h=patch_size, out_w=patch_size)
                        if arr.shape[0] != n_bands:
                            out = np.full((n_bands, patch_size, patch_size), np.nan, dtype=np.float32)
                            copy = min(n_bands, arr.shape[0])
                            out[:copy] = arr[:copy]
                            arr = out
                        return np.moveaxis(arr, 0, -1).astype(np.float32)
            except Exception:
                if attempt < retries:
                    time.sleep(0.5 * attempt)
                else:
                    break
        return np.full((patch_size, patch_size, n_bands), np.nan, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_fetch_chip, x, y): i
            for i, (x, y) in enumerate(unique_xy)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            unique_chips[idx] = future.result()

    chips = np.full((n, patch_size, patch_size, n_bands), np.nan, dtype=np.float32)
    for key_idx, key in enumerate(unique_keys):
        for row_idx in key_to_rows[key]:
            chips[row_idx] = unique_chips[key_idx]
    return chips


def sample_bare_earth_at_points(
    raster_path: Path,
    points_df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    output_prefix: str = "be_",
) -> pd.DataFrame:
    """
    Sample bare-earth raster values at point locations.

    Returns a dataframe with one row per input point and one column per raster band.
    """
    import rasterio
    from pyproj import Transformer

    if lon_col not in points_df.columns or lat_col not in points_df.columns:
        raise ValueError(f"Expected columns '{lon_col}' and '{lat_col}' in points dataframe")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS; cannot sample in coordinate space")

        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        coords = []
        for lon, lat in zip(points_df[lon_col].values, points_df[lat_col].values):
            x, y = transformer.transform(float(lon), float(lat))
            coords.append((x, y))

        sampled = np.array(list(src.sample(coords)), dtype=np.float32)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            sampled[np.isclose(sampled, nodata)] = np.nan

        descriptions = list(src.descriptions or [])
        if not descriptions or all(not d for d in descriptions):
            descriptions = [f"band_{i + 1}" for i in range(src.count)]
        else:
            descriptions = [
                (d.strip().lower().replace(" ", "_") if d else f"band_{i + 1}")
                for i, d in enumerate(descriptions)
            ]

    columns = [f"{output_prefix}{name}" for name in descriptions]
    return pd.DataFrame(sampled, columns=columns, index=points_df.index)
