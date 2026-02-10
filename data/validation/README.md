# Independent Validation Data

Datasets used for true out-of-sample evaluation of the ResNet ensemble. Zero overlap with training data confirmed by feature vector comparison.

## Files

| File | Samples | Source | Description |
|------|---------|--------|-------------|
| `national_independent_1368.csv` | 1,368 | ANSIS top-10cm soil cores | National independent validation set with 5-year median AlphaEarth embeddings. Targets: pH, CEC, ESP, Na |

## Speirs Farm Validation

The Speirs farm data (60 samples, in `data/farm/speirs_training_points_alphaearth_5yr.csv`) was also used as an independent validation set for the ResNet model. It contains pH, CEC, ESP, and Na from real lab soil tests at a farm the national model has never seen.

## Key Results

| Target | National (1,368) R² | Speirs Farm (60) R² |
|--------|---------------------|---------------------|
| pH     | 0.606               | -0.914              |
| CEC    | 0.817               | 0.214               |
| ESP    | 0.363               | 0.464               |
| Na     | 0.559               | 0.536               |

See `RESULTS.md` in the repo root for full analysis.
