# External Data Sources

This project uses the following external data sources:

## ANSIS (Australian National Soil Information System)
- URL: https://www.ansis.net/
- Provides soil observation data from Australian soil surveys
- Accessed via TERN Soil & Landscape API

## AlphaEarth Satellite Embeddings
- Platform: Google Earth Engine
- 64-band foundation model embeddings derived from Sentinel-2 imagery
- 5-year median composites at soil sample locations

## SoilGrids
- URL: https://soilgrids.org/
- Global soil property predictions at 250m resolution
- Used for comparison / gap-filling (optional)
