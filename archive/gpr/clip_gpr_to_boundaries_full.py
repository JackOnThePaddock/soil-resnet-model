import re
from pathlib import Path

import rasterio
import rasterio.mask
import shapefile


def safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return name[:64] if name else "paddock"


def shape_to_geom(shape) -> dict:
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    rings = []
    for i in range(len(parts) - 1):
        ring = points[parts[i] : parts[i + 1]]
        rings.append(ring)
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}


def build_boundary_map(boundary_shp: Path) -> dict:
    reader = shapefile.Reader(str(boundary_shp))
    fields = [f[0] for f in reader.fields[1:]]
    name_idx = fields.index("FIELD_NAME")
    boundary_map = {}
    for rec, shp in zip(reader.records(), reader.shapes()):
        raw_name = str(rec[name_idx])
        name = safe_name(raw_name)
        boundary_map[name] = shape_to_geom(shp)
    return boundary_map


def clip_raster(src_path: Path, geom: dict, out_path: Path) -> None:
    with rasterio.open(src_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        out_image, out_transform = rasterio.mask.mask(
            src,
            [geom],
            crop=True,
            nodata=nodata,
            all_touched=False,
            filled=True,
        )
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "LZW",
                "nodata": nodata,
            }
        )
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)


def main() -> None:
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\gpr_alphaearth_full")
    pred_dir = base_dir / "predictions"
    out_dir = base_dir / "predictions_clipped"
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_shp = Path(
        r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SpeirsBoundaries\boundaries\boundaries.shp"
    )
    if not boundary_shp.exists():
        raise FileNotFoundError(boundary_shp)

    boundary_map = build_boundary_map(boundary_shp)

    for var_dir in pred_dir.iterdir():
        if not var_dir.is_dir():
            continue
        out_var = out_dir / var_dir.name
        out_var.mkdir(parents=True, exist_ok=True)
        for tif in var_dir.glob("*.tif"):
            stem = tif.stem
            if "_gpr_" not in stem:
                continue
            paddock = stem.split("_gpr_")[0]
            geom = boundary_map.get(paddock)
            if geom is None:
                print(f"Skip (no boundary): {tif.name}")
                continue
            out_path = out_var / tif.name
            clip_raster(tif, geom, out_path)
            print(f"Clipped: {out_path.name}")


if __name__ == "__main__":
    main()
