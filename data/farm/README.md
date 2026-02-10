# Farm-Level Data

Local soil test data from Speirs Farm (NSW, Australia) used for RF model training and ResNet independent validation.

## Files

| File | Samples | Description |
|------|---------|-------------|
| `training_data_combined_1x1.csv` | 69 | Combined training data (Hill PDK n=33, Lightning Tree n=27, 400 n=9) with 1x1 AlphaEarth bands + pH, CEC, ESP |
| `training_data_combined_3x3.csv` | 69 | Same samples with 3x3 focal-mean smoothed AlphaEarth bands |
| `speirs_training_points_alphaearth_5yr.csv` | 60 | Hill PDK + Lightning Tree points with 5-year median AlphaEarth embeddings (used for ResNet independent validation) |

## Paddocks

- **Hill Paddock (HILLPDK)**: 33 samples
- **Lightning Tree**: 27 samples
- **Paddock 400**: 9 samples (excluded in some analyses due to small size)

## Variables

- **pH** (CaCl2) — soil acidity
- **CEC** (cmol+/kg) — cation exchange capacity
- **ESP** (%) — exchangeable sodium percentage, calculated as (Na_cmol / CEC) × 100
- **Na** (mg/kg or cmol+/kg) — exchangeable sodium
