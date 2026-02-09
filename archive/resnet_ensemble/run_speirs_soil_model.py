"""
Run ResNet Soil Model on Speirs Farm with 5-Year Median AlphaEarth Embeddings
==============================================================================
End-to-end pipeline:
  1. Load paddock boundaries from shapefile
  2. Download 5-year median AlphaEarth embeddings from GEE for each paddock
  3. Run trained ResNet ensemble (5 models) to predict 7 soil properties
  4. Create farm-wide mosaics for each soil property

Usage:
    python run_speirs_soil_model.py
    python run_speirs_soil_model.py --project "ee-yourproject"
    python run_speirs_soil_model.py --skip-download   # skip GEE, use existing embeddings
"""

import argparse
import io
import pickle
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import shapefile

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = Path(r"C:\Users\jackc\OneDrive\Documents\EW WH & MG SPEIRS")
BOUNDARIES_SHP = BASE_DIR / "SpeirsBoundaries" / "boundaries" / "boundaries.shp"
MODELS_DIR = Path(r"C:\Users\jackc\OneDrive\Documents\SOIL AI TRAINED MODELS")

OUTPUT_BASE = BASE_DIR / "SOIL Tests" / "exports" / "resnet_ensemble" / "predictions_5yr_median"
EMB_DIR = OUTPUT_BASE / "embeddings"
PRED_DIR = OUTPUT_BASE / "predictions"
MOSAIC_DIR = OUTPUT_BASE / "farm_mosaics"

TARGETS = ["ph", "cec", "esp", "soc", "ca", "mg", "na"]
NODATA = -9999.0


# ============================================================================
# Model Architecture (inline to avoid import path issues)
# ============================================================================

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.activation(out)
        out = self.dropout(out)
        return out


class NationalSoilNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.2,
        target_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.target_names = target_names or TARGETS
        self.num_targets = len(self.target_names)

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for name in self.target_names
        })

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return {name: head(x).squeeze(-1) for name, head in self.heads.items()}

    def forward_stacked(self, x):
        outputs = self.forward(x)
        return torch.stack([outputs[name] for name in self.target_names], dim=1)


# ============================================================================
# Ensemble Loader
# ============================================================================

class SoilEnsemble:
    def __init__(self, models_dir: Path, device: Optional[torch.device] = None):
        self.models_dir = Path(models_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading ensemble from: {self.models_dir}")
        print(f"Using device: {self.device}")

        # Load scaler
        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load models
        model_files = sorted(self.models_dir.glob("model_*.pth"))
        # Exclude duplicates like "model_1 (1).pth"
        model_files = [f for f in model_files if re.fullmatch(r"model_\d+\.pth", f.name)]
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")

        self.models = []
        self.target_names = None
        self.config = None

        for model_path in model_files:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if self.config is None:
                self.config = checkpoint.get("config", {})
                self.target_names = checkpoint.get("target_names", TARGETS)

            model = NationalSoilNet(
                input_dim=self.config.get("input_dim", 64),
                hidden_dim=self.config.get("hidden_dim", 128),
                num_res_blocks=self.config.get("num_res_blocks", 2),
                dropout=self.config.get("dropout", 0.2),
                target_names=self.target_names,
            ).to(self.device)

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models.append(model)
            print(f"  Loaded: {model_path.name}")

        print(f"Loaded {len(self.models)} models, targets: {self.target_names}")

    def predict_batch(self, X: np.ndarray, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble in batches. Returns (mean, std) arrays of shape (n, n_targets)."""
        n_samples = X.shape[0]
        n_targets = len(self.target_names)
        all_model_preds = np.zeros((len(self.models), n_samples, n_targets), dtype=np.float32)

        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            for m_idx, model in enumerate(self.models):
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    X_tensor = torch.FloatTensor(X_scaled[start:end]).to(self.device)
                    preds = model.forward_stacked(X_tensor).cpu().numpy()
                    all_model_preds[m_idx, start:end] = preds

        mean_preds = np.mean(all_model_preds, axis=0)
        std_preds = np.std(all_model_preds, axis=0)
        return mean_preds, std_preds


# ============================================================================
# Step 1: Load Boundaries
# ============================================================================

def safe_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return name[:64] if name else "paddock"


def load_paddocks(shp_path: Path) -> List[dict]:
    """Load paddock polygons from shapefile. Returns list of {name, points, parts}."""
    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    paddocks = []
    for rec, shape in zip(reader.records(), reader.shapes()):
        attrs = dict(zip(fields, rec))
        name = safe_name(str(attrs.get("FIELD_NAME", "paddock")))
        paddocks.append({
            "name": name,
            "raw_name": attrs.get("FIELD_NAME", "paddock"),
            "points": shape.points,
            "parts": list(shape.parts),
            "farm": attrs.get("FARM_NAME", ""),
        })
    return paddocks


def shape_to_ee_polygon(points, parts):
    """Convert shapefile points/parts to ee.Geometry.Polygon."""
    import ee
    part_indices = parts + [len(points)]
    rings = []
    for i in range(len(part_indices) - 1):
        ring = points[part_indices[i]:part_indices[i + 1]]
        rings.append(ring)
    return ee.Geometry.Polygon(rings)


# ============================================================================
# Step 2: Download 5-Year Median Embeddings from GEE
# ============================================================================

def download_embeddings(paddocks: List[dict], emb_dir: Path, project: Optional[str] = None):
    """Download 5-year median AlphaEarth embeddings for each paddock."""
    import ee

    # Initialize GEE
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load the full collection and determine available years
    alpha_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    alpha_col = alpha_col.map(
        lambda img: img.set("year", ee.Date(img.get("system:time_start")).get("year"))
    )
    all_years = ee.List(alpha_col.aggregate_array("year")).distinct().sort()
    last5 = all_years.slice(-5)
    year_list = last5.getInfo()
    print(f"Using years for 5-year median: {year_list}")

    # Filter to last 5 years
    alpha_5yr = alpha_col.filter(ee.Filter.inList("year", last5))
    median_img = alpha_5yr.median()

    for pdk in paddocks:
        out_path = emb_dir / f"{pdk['name']}_5yr_median.tif"
        if out_path.exists():
            print(f"  Skipping {pdk['name']} (already exists)")
            continue

        print(f"  Downloading {pdk['name']}...")
        geom = shape_to_ee_polygon(pdk["points"], pdk["parts"])
        clipped = median_img.clip(geom)

        # Use getDownloadURL (proven pattern from existing scripts)
        url = clipped.getDownloadURL({
            "scale": 10,
            "region": geom,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
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
            signature = f.read(4)

        if signature.startswith(b"PK"):
            with zipfile.ZipFile(tmp_path, "r") as z:
                tif_names = [n for n in z.namelist() if n.lower().endswith(".tif")]
                if not tif_names:
                    raise RuntimeError(f"No GeoTIFF found in zip for {pdk['name']}")
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

        print(f"    Saved: {out_path.name}")

    return year_list


# ============================================================================
# Step 3: Run Ensemble Inference on Rasters
# ============================================================================

def predict_raster(
    ensemble: SoilEnsemble,
    input_path: Path,
    output_dir: Path,
    block_size: int = 512,
) -> Dict[str, Path]:
    """Generate prediction GeoTIFFs (2 bands: mean + uncertainty) for each target."""
    import rasterio
    from rasterio.windows import Window

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        n_bands = src.count

        if n_bands != 64:
            print(f"  WARNING: Expected 64 bands, got {n_bands} in {input_path.name}")

        # Output profile: 2 bands (prediction + uncertainty)
        profile.update(count=2, dtype="float32", nodata=NODATA, compress="deflate")

        output_files = {}
        writers = {}

        for target in ensemble.target_names:
            out_path = output_dir / f"{input_path.stem}_{target}.tif"
            writers[target] = rasterio.open(out_path, "w", **profile)
            output_files[target] = out_path

        # Process in blocks
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                win_h = min(block_size, height - row)
                win_w = min(block_size, width - col)
                window = Window(col, row, win_w, win_h)

                block = src.read(window=window)  # (64, h, w)
                n_pixels = win_h * win_w
                X = block.reshape(n_bands, -1).T  # (n_pixels, 64)

                # Valid pixel mask (no NaN, no zeros across all bands)
                valid_mask = np.isfinite(X).all(axis=1)
                if src.nodata is not None and np.isfinite(src.nodata):
                    valid_mask &= ~(X == src.nodata).any(axis=1)

                mean_out = np.full((len(ensemble.target_names), n_pixels), NODATA, dtype=np.float32)
                std_out = np.full((len(ensemble.target_names), n_pixels), NODATA, dtype=np.float32)

                if valid_mask.sum() > 0:
                    mean_preds, std_preds = ensemble.predict_batch(X[valid_mask])
                    for i in range(len(ensemble.target_names)):
                        mean_out[i, valid_mask] = mean_preds[:, i]
                        std_out[i, valid_mask] = std_preds[:, i]

                for i, target in enumerate(ensemble.target_names):
                    writers[target].write(mean_out[i].reshape(win_h, win_w), 1, window=window)
                    writers[target].write(std_out[i].reshape(win_h, win_w), 2, window=window)

        for writer in writers.values():
            writer.close()

    return output_files


# ============================================================================
# Step 4: Create Farm Mosaics
# ============================================================================

def create_mosaics(pred_dir: Path, mosaic_dir: Path, target_names: List[str], paddock_names: List[str]):
    """Merge per-paddock predictions into farm-wide mosaics."""
    import rasterio
    from rasterio.merge import merge

    mosaic_dir.mkdir(parents=True, exist_ok=True)

    for target in target_names:
        # Find all prediction files for this target
        pred_files = []
        for name in paddock_names:
            p = pred_dir / f"{name}_5yr_median_{target}.tif"
            if p.exists():
                pred_files.append(p)

        if not pred_files:
            print(f"  No prediction files found for {target}, skipping mosaic")
            continue

        # Merge
        srcs = [rasterio.open(p) for p in pred_files]
        mosaic, out_transform = merge(srcs, nodata=NODATA)
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "nodata": NODATA,
            "count": mosaic.shape[0],
            "dtype": "float32",
            "compress": "deflate",
        })

        out_path = mosaic_dir / f"farm_mosaic_{target}_5yr_median.tif"
        with rasterio.open(out_path, "w", **out_meta) as dest:
            for band_idx in range(mosaic.shape[0]):
                dest.write(mosaic[band_idx].astype(np.float32), band_idx + 1)

        for s in srcs:
            s.close()

        print(f"  Mosaic: {out_path.name} ({len(pred_files)} paddocks)")


# ============================================================================
# Summary Statistics
# ============================================================================

def print_summary(pred_dir: Path, target_names: List[str], paddock_names: List[str]):
    """Print summary statistics per paddock per target."""
    import rasterio

    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")

    header = f"{'Paddock':<22}"
    for t in target_names:
        header += f"{'  ' + t.upper():>10} {'(unc)':>7}"
    print(header)
    print("-" * len(header))

    for name in paddock_names:
        row = f"{name:<22}"
        for target in target_names:
            p = pred_dir / f"{name}_5yr_median_{target}.tif"
            if not p.exists():
                row += f"{'N/A':>10} {'N/A':>7}"
                continue
            with rasterio.open(p) as src:
                mean_band = src.read(1)
                std_band = src.read(2)
                valid = mean_band != NODATA
                if valid.sum() > 0:
                    avg = np.mean(mean_band[valid])
                    avg_unc = np.mean(std_band[valid])
                    row += f"{avg:10.2f} {avg_unc:7.3f}"
                else:
                    row += f"{'empty':>10} {'':>7}"
        print(row)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run ResNet ensemble soil predictions on Speirs Farm paddocks"
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="GEE project ID (e.g., 'ee-yourproject'). "
             "If not set, uses default credentials."
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip GEE download, use existing embeddings"
    )
    parser.add_argument(
        "--skip-predict", action="store_true",
        help="Skip prediction, only create mosaics from existing predictions"
    )
    parser.add_argument(
        "--block-size", type=int, default=512,
        help="Block size for raster processing (default: 512)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ResNet Soil Model - Speirs Farm - 5-Year Median AlphaEarth Embeddings")
    print("=" * 70)

    # ---- Step 1: Load boundaries ----
    print(f"\n[Step 1] Loading paddock boundaries from {BOUNDARIES_SHP.name}")
    if not BOUNDARIES_SHP.exists():
        print(f"ERROR: Shapefile not found: {BOUNDARIES_SHP}")
        sys.exit(1)

    paddocks = load_paddocks(BOUNDARIES_SHP)
    paddock_names = [p["name"] for p in paddocks]
    print(f"  Found {len(paddocks)} paddocks: {', '.join(paddock_names)}")

    # ---- Step 2: Download embeddings ----
    if not args.skip_download:
        print(f"\n[Step 2] Downloading 5-year median AlphaEarth embeddings")
        years = download_embeddings(paddocks, EMB_DIR, project=args.project)
    else:
        print(f"\n[Step 2] Skipping download (--skip-download)")
        # Verify embeddings exist
        missing = [n for n in paddock_names if not (EMB_DIR / f"{n}_5yr_median.tif").exists()]
        if missing:
            print(f"  WARNING: Missing embeddings for: {', '.join(missing)}")

    # ---- Step 3: Run ensemble inference ----
    if not args.skip_predict:
        print(f"\n[Step 3] Running ResNet ensemble inference")
        print(f"  Models dir: {MODELS_DIR}")

        ensemble = SoilEnsemble(MODELS_DIR)

        for pdk in paddocks:
            emb_path = EMB_DIR / f"{pdk['name']}_5yr_median.tif"
            if not emb_path.exists():
                print(f"  Skipping {pdk['name']} (no embedding file)")
                continue

            print(f"\n  Predicting {pdk['name']}...")
            output_files = predict_raster(
                ensemble=ensemble,
                input_path=emb_path,
                output_dir=PRED_DIR,
                block_size=args.block_size,
            )
            for target, path in output_files.items():
                print(f"    {target}: {path.name}")
    else:
        print(f"\n[Step 3] Skipping prediction (--skip-predict)")
        # Still need target names for mosaics
        ensemble = SoilEnsemble(MODELS_DIR)

    # ---- Step 4: Create farm mosaics ----
    print(f"\n[Step 4] Creating farm-wide mosaics")
    create_mosaics(PRED_DIR, MOSAIC_DIR, ensemble.target_names, paddock_names)

    # ---- Summary ----
    print_summary(PRED_DIR, ensemble.target_names, paddock_names)

    print(f"\n{'='*70}")
    print("Complete!")
    print(f"  Embeddings: {EMB_DIR}")
    print(f"  Predictions: {PRED_DIR}")
    print(f"  Mosaics: {MOSAIC_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
