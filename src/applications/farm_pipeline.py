"""End-to-end farm prediction pipeline."""

from pathlib import Path
from typing import Optional

from src.models.ensemble import SoilEnsemble
from src.inference.predict_raster import predict_raster
from src.inference.mosaic import create_mosaics, print_summary
from src.features.gee_sampler import safe_name


def load_paddocks(shp_path: str) -> list:
    """Load paddock polygons from shapefile."""
    import shapefile
    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    paddocks = []
    for rec, shape in zip(reader.records(), reader.shapes()):
        attrs = dict(zip(fields, rec))
        paddocks.append({
            "name": safe_name(str(attrs.get("FIELD_NAME", "paddock"))),
            "points": shape.points,
            "parts": list(shape.parts),
        })
    return paddocks


def run_farm_pipeline(
    boundaries_shp: str,
    models_dir: str,
    output_dir: str,
    project: Optional[str] = None,
    skip_download: bool = False,
    skip_predict: bool = False,
    block_size: int = 512,
) -> None:
    """Run complete farm prediction pipeline."""
    output_dir = Path(output_dir)
    emb_dir = output_dir / "embeddings"
    pred_dir = output_dir / "predictions"
    mosaic_dir = output_dir / "farm_mosaics"

    paddocks = load_paddocks(boundaries_shp)
    paddock_names = [p["name"] for p in paddocks]
    print(f"Found {len(paddocks)} paddocks: {', '.join(paddock_names)}")

    if not skip_download:
        from src.features.gee_sampler import sample_embeddings_for_boundaries
        sample_embeddings_for_boundaries(boundaries_shp, str(emb_dir), project=project)

    if not skip_predict:
        ensemble = SoilEnsemble(models_dir)
        for pdk in paddocks:
            emb_path = emb_dir / f"{pdk['name']}_5yr_median.tif"
            if not emb_path.exists():
                print(f"  Skipping {pdk['name']} (no embedding)")
                continue
            print(f"  Predicting {pdk['name']}...")
            predict_raster(ensemble, emb_path, pred_dir, block_size=block_size)
    else:
        ensemble = SoilEnsemble(models_dir)

    print("\nCreating farm-wide mosaics")
    create_mosaics(str(pred_dir), str(mosaic_dir), ensemble.target_names, paddock_names)
    print_summary(str(pred_dir), ensemble.target_names, paddock_names)
