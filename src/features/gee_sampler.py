"""Consolidated Google Earth Engine AlphaEarth embedding sampler."""

import io
import re
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests


def safe_name(name: str) -> str:
    """Sanitize name for use as filename."""
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return name[:64] if name else "paddock"


def shape_to_ee_polygon(points, parts):
    """Convert shapefile points/parts to ee.Geometry.Polygon."""
    import ee
    part_indices = list(parts) + [len(points)]
    rings = [points[part_indices[i]:part_indices[i + 1]] for i in range(len(part_indices) - 1)]
    return ee.Geometry.Polygon(rings)


def download_embedding(
    geom, out_path: Path, median_image, scale: int = 10,
) -> Path:
    """Download AlphaEarth embedding for a single geometry."""
    import ee

    if out_path.exists():
        return out_path

    clipped = median_image.clip(geom)
    url = clipped.getDownloadURL({
        "scale": scale, "region": geom, "format": "GEO_TIFF", "crs": "EPSG:4326",
    })

    tmp_path = out_path.with_suffix(".download")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Handle zip vs raw tif response
    with open(tmp_path, "rb") as f:
        sig = f.read(4)

    if sig.startswith(b"PK"):
        with zipfile.ZipFile(tmp_path, "r") as z:
            tif_names = [n for n in z.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise RuntimeError(f"No GeoTIFF in zip")
            z.extract(tif_names[0], path=out_path.parent)
            extracted = out_path.parent / tif_names[0]
            if extracted != out_path:
                if out_path.exists():
                    out_path.unlink()
                extracted.rename(out_path)
        tmp_path.unlink(missing_ok=True)
    else:
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)

    return out_path


def sample_embeddings_for_boundaries(
    shp_path: str,
    output_dir: str,
    n_years: int = 5,
    project: Optional[str] = None,
) -> List[Path]:
    """Download 5-year median AlphaEarth embeddings for all paddocks in a shapefile."""
    import ee
    import shapefile

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build median image from last N years
    alpha_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    alpha_col = alpha_col.map(lambda img: img.set("year", ee.Date(img.get("system:time_start")).get("year")))
    all_years = ee.List(alpha_col.aggregate_array("year")).distinct().sort()
    last_n = all_years.slice(-n_years)
    year_list = last_n.getInfo()
    print(f"Using years: {year_list}")

    median_img = alpha_col.filter(ee.Filter.inList("year", last_n)).median()

    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    output_paths = []

    for rec, shape in zip(reader.records(), reader.shapes()):
        attrs = dict(zip(fields, rec))
        name = safe_name(str(attrs.get("FIELD_NAME", attrs.get("Name", "paddock"))))
        geom = shape_to_ee_polygon(shape.points, shape.parts)
        out_path = output_dir / f"{name}_5yr_median.tif"

        print(f"  Downloading {name}...")
        download_embedding(geom, out_path, median_img)
        output_paths.append(out_path)
        print(f"    Saved: {out_path.name}")

    return output_paths
