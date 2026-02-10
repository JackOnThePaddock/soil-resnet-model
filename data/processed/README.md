# Processed Data

## Files

### features.csv (3,534 × 74)
Complete training dataset with **raw** target values and AlphaEarth satellite features.

- **Metadata columns**: `id`, `lat`, `lon`
- **Target columns** (7): `ph`, `cec`, `esp`, `soc`, `ca`, `mg`, `na` — real-world units (pH units, cmol/kg, %)
- **Feature columns** (64): `band_0` through `band_63` — AlphaEarth 5-year median embeddings

Reconstructed from 9 year-specific files (2017–2025). Rows with any missing band values were excluded (3,669 → 3,534).

### features_normalized.csv (3,534 × 71)
The **normalized** version used directly for model training.

- **Target columns** (7): `ph`, `cec`, `esp`, `soc`, `ca`, `mg`, `na` — z-scored (mean=0, std=1)
- **Feature columns** (64): `band_0` through `band_63` — same as raw (already normalized by GEE)

Target normalization parameters (computed from full dataset):

| Target | Mean | Std |
|--------|------|-----|
| ph | 6.095 | 1.037 |
| cec | 14.249 | 15.106 |
| esp | 2.260 | 2.945 |
| soc | 3.401 | 3.854 |
| ca | 7.447 | 8.513 |
| mg | 3.460 | 4.508 |
| na | 0.290 | 0.584 |

To convert normalized predictions back to raw: `raw = normalized * std + mean`

## How to Regenerate

```bash
python scripts/run_data_extraction.py --output data/processed/
python scripts/run_gee_sampling.py --csv data/processed/soil_sites.csv --output data/processed/features.csv
```
