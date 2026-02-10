# Google Earth Engine — AlphaEarth Embedding Extraction

## Authentication

1. Install the Earth Engine Python API:
   ```bash
   pip install earthengine-api
   ```

2. Authenticate:
   ```bash
   earthengine authenticate
   ```

3. Initialize with your project:
   ```python
   import ee
   ee.Initialize(project='your-project-id')
   ```

## AlphaEarth Embeddings

This project uses AlphaEarth foundation model embeddings from Google Earth Engine.

- **Image Collection**: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- **Bands**: 64 embedding dimensions (A00–A63)
- **Temporal composites**: 5-year median (typically 2020–2024)
- **Spatial resolution**: 10 m (Sentinel-2 native)
- **Coverage**: Global annual mosaics

## Scripts in this Directory

| Script | Description |
|--------|-------------|
| `sample_embeddings.py` | Consolidated GEE sampling script for extracting embeddings at soil points |
| `gee_sample_alphaearth_by_year.py` | Year-specific sampling (produces per-year CSVs) |
| `resnet_colab.py` | Standalone ResNet training script for Google Colab |
| `sodic_classifier_colab.py` | XGBoost sodic risk classifier for Google Colab |

## Usage

### Extract embeddings at soil sample locations

```bash
python gee/sample_embeddings.py \
    --csv data/raw/soil_data_export.csv \
    --output data/processed/features.csv \
    --project your-gee-project
```

### Run ResNet training on Google Colab

1. Upload `resnet_colab.py` and `data/processed/features_normalized.csv` to Colab
2. Run the notebook `notebooks/03_resnet_training.ipynb` which calls `resnet_colab.py`

## All GEE Scripts (in archive/)

The full set of 9 GEE sampling scripts is in `archive/gee_scripts/`. These are the individual variants that were consolidated into `sample_embeddings.py`.
