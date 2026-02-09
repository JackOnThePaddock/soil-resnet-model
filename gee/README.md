# Google Earth Engine Setup

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

- Image Collection: `projects/sat-io/open-datasets/STAC/alphaearth-embeddings-sentinel2`
- Bands: 64 embedding dimensions
- Temporal composites: 5-year median (2019-2024)
- Spatial resolution: 10m (Sentinel-2 native)

## Sampling Script

Use `gee/sample_embeddings.py` to extract embeddings at soil sample locations:

```bash
python gee/sample_embeddings.py --csv data/raw/soil_data_export.csv --output data/processed/features.csv --project your-gee-project
```
