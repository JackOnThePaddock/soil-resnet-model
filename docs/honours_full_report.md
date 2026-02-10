# Honours Project Report - Digital Soil Mapping With AlphaEarth Embeddings

## Title
High-resolution mapping of pH (CaCl2), CEC, and ESP using AlphaEarth embeddings and Random Forest models in mixed broadacre paddocks, NSW, Australia

**Author:** [Your Name]

**Supervisors:** [Supervisor Names]

**Institution:** [University]

**Date:** 2026-01-26

---

## Abstract
This project built a reproducible digital soil mapping pipeline to predict pH (CaCl2), cation exchange capacity (CEC), and exchangeable sodium percentage (ESP) at 10 m resolution using AlphaEarth 5-year median embeddings and local soil tests. The workflow integrates point-based soil test data, feature selection, model evaluation, and spatial prediction with paddock clipping and NDVI masking from bare-earth Sentinel-2 composites. Random Forest models with band selection outperformed all-64-band models across targets. A 3x3 neighborhood extraction was evaluated but did not universally improve accuracy. Cross-paddock generalization remains limited, reinforcing the need for targeted sampling. A constrained cLHS sampling design (n=50) was generated to guide future field sampling, excluding trees (NDVI > 0.30) and avoiding proximity to existing samples. Outputs include paddock-level and whole-farm GeoTIFFs, bias-corrected variants, and KML sampling files.

---

## 1. Project Aims
1. Build a robust pipeline to map pH (CaCl2), CEC, and ESP using AlphaEarth embeddings and local soil tests.
2. Quantify model performance using multiple validation schemes (LOOCV, k-fold, leave-one-paddock-out).
3. Compare 1x1 vs 3x3 pixel extraction and identify the most reliable configuration.
4. Generate operational outputs (paddock maps, whole-farm mosaics, NDVI-masked products).
5. Design a constrained cLHS sampling plan for 50 new samples across the farm.

---

## 2. Study Area and Data
**Paddocks (boundary layer):** 300, 400, AIRSTRIPE, CREEK PADDOCK, DEER PADDOCKS, HILLPDK, LIGHTNING TREE, Lease 1-3, LYNN DENE MAILBOX, MERRIMU DRIVEWAY, TURKEY YARD (EPSG:4326)

**Soil tests used in model training:**
- HILLPDK: 33 points
- LIGHTNING TREE: 27 points
- 400: 9 points
- Total: 69 points

**Variables used:** pH (CaCl2), CEC (cmol+/kg), Na (mg/kg or cmol+/kg), ESP

**ESP computation:**
- If Na is mg/kg: Na_cmol = Na / 230
- ESP = (Na_cmol / CEC) * 100

**AlphaEarth embeddings:**
- Dataset: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
- 5-year median: 2020-2024
- 64 bands, 10 m resolution

**NDVI mask:**
- Source: GA Barest Earth (Sentinel-2) composite
- NDVI calculation using red band 1 and NIR band 7
- NDVI threshold: 0.35 for map outputs; 0.30 for sampling design

---

## 3. Methods
### 3.1 Data preparation
- Soil test point shapefiles ingested and merged by paddock.
- ESP calculated from Na and CEC where required.
- Points joined to AlphaEarth embeddings for model training.

### 3.2 Feature extraction
- Two extraction modes tested:
  - 1x1 pixel value at the point
  - 3x3 neighborhood mean (focal mean) to reduce GPS jitter and pixel noise

### 3.3 Modeling
- Model: RandomForestRegressor
- Parameters: 300 trees, max_features=sqrt, random_state=42
- Feature selection: recursive feature elimination (RFE) for combined dataset; RF feature importance ranking for hill+lightning-only scenario

### 3.4 Validation
- LOOCV (leave-one-out cross-validation)
- 5-fold CV
- Leave-one-paddock-out (LOPO)
- Per-paddock holdout errors

### 3.5 Mapping
- Predictions generated for each paddock embedding GeoTIFF
- Clip to paddock boundaries
- Apply NDVI mask (<= 0.35) to remove vegetation and trees
- ESP clamped to non-negative values for map outputs

### 3.6 Sampling design
- Constrained cLHS sampling (n=50)
- NDVI <= 0.30 mask and paddock boundary enforcement
- Excluded points within 20 m of existing Hill and Lightning samples
- PCA reduction (6 components) + correlation penalty in cLHS

---

## 4. Results

### 4.1 Combined dataset (HILLPDK + LIGHTNING + 400) - 1x1 best-bands
**LOOCV**
| Target | RMSE | MAE | R2 | Bands |
|---|---:|---:|---:|---:|
| pH  | 0.390 | 0.226 | 0.238 | 10 |
| CEC | 2.749 | 1.885 | 0.358 | 12 |
| ESP | 1.892 | 1.509 | 0.546 | 12 |

**5-fold and LOPO**
| Target | 5-fold RMSE | 5-fold R2 | LOPO RMSE | LOPO R2 |
|---|---:|---:|---:|---:|
| pH  | 0.428 | 0.080 | 0.393 | 0.224 |
| CEC | 3.116 | 0.176 | 3.322 | 0.063 |
| ESP | 1.933 | 0.527 | 2.831 | -0.016 |

### 4.2 Combined dataset (HILLPDK + LIGHTNING + 400) - 3x3 best-bands
**LOOCV**
| Target | RMSE | MAE | R2 | Bands |
|---|---:|---:|---:|---:|
| pH  | 0.407 | 0.235 | 0.169 | 15 |
| CEC | 2.861 | 1.872 | 0.305 | 15 |
| ESP | 1.960 | 1.588 | 0.513 | 15 |

**5-fold and LOPO**
| Target | 5-fold RMSE | 5-fold R2 | LOPO RMSE | LOPO R2 |
|---|---:|---:|---:|---:|
| pH  | 0.443 | 0.018 | 0.423 | 0.101 |
| CEC | 3.136 | 0.165 | 3.355 | 0.044 |
| ESP | 2.013 | 0.486 | 2.892 | -0.060 |

### 4.3 Per-paddock holdout (combined dataset, 1x1 best-bands)
| Target | Paddock | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| pH  | 400 | 0.494 | 0.362 | -0.029 |
| pH  | HILLPDK | 0.427 | 0.209 | 0.052 |
| pH  | LIGHTNING_TREE | 0.302 | 0.238 | 0.211 |
| CEC | 400 | 4.389 | 3.510 | -0.424 |
| CEC | HILLPDK | 3.517 | 2.802 | -0.352 |
| CEC | LIGHTNING_TREE | 2.581 | 2.038 | 0.203 |
| ESP | 400 | 2.602 | 2.404 | -0.624 |
| ESP | HILLPDK | 3.238 | 2.920 | -0.747 |
| ESP | LIGHTNING_TREE | 2.328 | 1.887 | -0.050 |

### 4.4 Bias correction (combined dataset, 1x1)
LOOCV improvement using paddock mean error correction:
- pH RMSE: 0.3908 -> 0.3875
- CEC RMSE: 2.7512 -> 2.7150
- ESP RMSE: 1.8958 -> 1.8820

### 4.5 Hill + Lightning only (400 removed)
Best accuracy by target across scenarios:
- **pH:** 1x1 best-bands (10 bands)
  - LOOCV RMSE 0.390, MAE 0.216, R2 0.038
  - LOPO RMSE 0.444, MAE 0.288, R2 -0.247
- **CEC:** 3x3 best-bands (20 bands)
  - LOOCV RMSE 2.744, MAE 1.680, R2 0.220
  - LOPO RMSE 2.973, MAE 2.189, R2 0.085
- **ESP:** 1x1 best-bands (20 bands)
  - LOOCV RMSE 1.912, MAE 1.516, R2 0.560
  - LOPO RMSE 2.526, MAE 2.156, R2 0.232

### 4.6 GPR baseline (for comparison)
GPR LOOCV (combined dataset) underperformed RF and was dropped:
- pH RMSE 0.392, R2 0.071
- CEC RMSE 2.853, R2 0.218

### 4.7 Outputs generated
- Paddock-level GeoTIFFs for all paddocks (pH, CEC, ESP)
- NDVI-masked and boundary-clipped versions
- Whole-farm mosaics (NDVI masked)
- ESP clamped to non-negative values

Whole-farm mosaics (1x1 best-bands, NDVI masked):
- `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\all_paddocks_1x1\farm_mosaics_ndvi35\farm_pH_rf_bestbands_1x1_ndvi35.tif`
- `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\all_paddocks_1x1\farm_mosaics_ndvi35\farm_CEC_rf_bestbands_1x1_ndvi35.tif`
- `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\all_paddocks_1x1\farm_mosaics_ndvi35\farm_ESP_rf_bestbands_1x1_ndvi35_nonneg.tif`

---

## 5. Sampling Design (cLHS)
A constrained cLHS sampling design (n=50) was generated across the whole farm:
- NDVI <= 0.30 (bare-earth mask)
- Points clipped to paddock boundaries
- Excluded points within 20 m of existing Hill and Lightning samples

Sampling KML:
- `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\clhs_sampling\clhs_sampling_points_50_inside_boundaries_ndvi30.kml`

Sampling distribution by paddock:
```
Lease_2              5
Lease_1              5
MERRIMU_DRIVEWAY     5
400                  5
Lease_3              4
LIGHTNING_TREE       4
HILLPDK              4
TURKEY_YARD          4
LYNN_DENE_MAILBOX    3
DEER_PADDOCKS        3
AIRSTRIPE            3
300                  3
CREEK_PADDOCK        2
```

---

## 6. Discussion and Findings
1. **Best overall model** for the full dataset remains the 1x1 best-bands RF, with stronger LOOCV performance than 3x3.
2. **3x3 extraction** did not consistently improve accuracy across the combined dataset but did improve CEC slightly when 400 was excluded.
3. **Cross-paddock transferability** (LOPO) is weak for some targets, highlighting the need for local calibration.
4. **ESP prediction** is the most reliable target overall (highest R2), but still requires non-negative clamping.
5. **Bias correction** improves LOOCV error slightly and is useful for paddock-specific calibration.
6. **cLHS sampling plan** offers a robust path to improve model stability by targeting under-sampled areas and ensuring spatial representativeness.

---

## 7. Limitations
- Small and unbalanced sample counts (notably 400).
- Limited depth information used in modeling.
- No explicit terrain or radiometric covariates in the final models.
- NDVI masking assumes bare-earth composite reliability.
- ESP constraints are enforced post-hoc rather than modeled physically.

---

## 8. Recommendations for First Class Honours Completion
1. **Collect the 50 cLHS samples** and re-train the model with expanded ground truth.
2. **Add terrain or radiometric covariates** once sample size increases to avoid overfitting.
3. **Report uncertainty** (e.g., RF variance or quantile forests) for decision support.
4. **Validate with independent paddock samples** (true external validation).
5. **Include spatial visual validation** with yield maps or soil management zones.

---

## 9. Thesis Deliverables Checklist
- [ ] Final methodology chapter with pipeline diagram
- [ ] Results chapter with maps + error tables
- [ ] Discussion linking agronomic meaning to model outputs
- [ ] Limitation section and future work
- [ ] Data and code archive for reproducibility

---

## 10. Data and Code Locations (Key Files)
- Combined training: `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp\training_data_combined.csv`
- 3x3 training: `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\training_data_combined_3x3.csv`
- Hill+Lightning evaluation: `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_hill_lightning_only\cv_accuracy_summary.csv`
- Sampling KML: `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\clhs_sampling\clhs_sampling_points_50_inside_boundaries_ndvi30.kml`

---

## Appendix A: Combined Dataset Metrics (1x1)
See `cv_accuracy_summary.csv`, `loocv_accuracy_summary.csv`, and `cv_paddock_holdout_summary.csv`.

## Appendix B: Combined Dataset Metrics (3x3)
See `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_combined_ph_cec_esp_3x3\cv_accuracy_summary.csv`.

## Appendix C: Hill + Lightning Only Metrics
See `C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\rf_hill_lightning_only\cv_accuracy_summary.csv`.
