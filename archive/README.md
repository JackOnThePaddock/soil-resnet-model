# Archive

Original scripts preserved for reference. These are the source scripts from the project's development, organized by category. The consolidated, refactored versions live in `src/`.

## Directory Index

### resnet_ensemble/ (5 scripts)

The deep learning model — a multi-target ResNet ensemble predicting 7 soil properties from 64-band AlphaEarth embeddings, trained on 3,534 national ANSIS samples.

| Script | Lines | Description |
|--------|-------|-------------|
| `resnet_soil_trainer.py` | 657 | Core training script: SoilDataset, ResidualBlock, NationalSoilNet, masked_mse_loss, K-fold CV + ensemble training |
| `ensemble_inference.py` | 410 | SoilEnsemble loader with point and raster prediction modes, uncertainty via ensemble std |
| `run_speirs_soil_model.py` | 563 | End-to-end farm pipeline: GEE download, paddock inference, mosaic creation (contains inline model copies) |
| `resnet_colab.py` | 354 | Colab-optimized training script (standalone, no imports from src/) |
| `sodic_classifier_colab.py` | 132 | XGBoost binary classifier for sodic risk (ESP > threshold) |

### random_forest/ (20+ scripts)

Local farm RF models trained on 69 Speirs farm soil samples (Hill Paddock n=33, Lightning Tree n=27, Paddock 400 n=9). Predicts pH, CEC, ESP using RFE-selected AlphaEarth bands.

| Script | Description |
|--------|-------------|
| `rf_train_predict_combined.py` | Main RF pipeline: collect points, RFE feature selection, LOOCV, raster prediction (1x1) |
| `cv_metrics.py` | 5-fold CV and Leave-One-Group-Out evaluation comparing all-64 vs best-bands |
| `cv_paddock_holdout.py` | Leave-one-paddock-out validation |
| `cv_paddock_holdout_bias.py` | Paddock holdout with bias correction |
| `loocv_bias_corrected_metrics.py` | LOOCV with paddock mean error correction |
| `rf_predict_turkey_yard.py` | Prediction for Turkey Yard paddock |
| `rf_predict_all_paddocks_1x1.py` | Batch prediction for all paddocks (1x1) |
| `rf_eval_hill_lightning_scenarios.py` | Comprehensive Hill+Lightning evaluation: LOOCV, 3-fold, LOPO, feature ranking |
| `rf_exchangeable_sodium_hill_lightning.py` | Na-specific RF model (Hill+Lightning) |
| `rf_exchangeable_sodium_400.py` | Na-specific RF model (with 400) |
| `rf_minmax_calibrate_paddocks.py` | Min-max calibration approach |
| `clip_ndvi35_*.py` | NDVI masking scripts (threshold 0.35) |
| `clamp_esp_nonneg*.py` | Post-hoc ESP clamping to non-negative |
| `apply_bias_to_ndvi_clips.py` | Apply paddock bias correction to NDVI-clipped rasters |
| `mosaic_farm_ndvi35.py` | Merge paddock rasters into farm-wide mosaic |

### random_forest_3x3/ (6 scripts)

Same RF pipeline but with 3x3 neighborhood averaging of AlphaEarth embeddings (focal mean to reduce GPS jitter).

| Script | Description |
|--------|-------------|
| `rf_train_predict_combined_3x3.py` | Full 3x3 pipeline with smooth_embeddings(), all CV methods |
| `rf_train_eval_combined_3x3.py` | Training + evaluation variant |
| `evaluate_3x3_metrics.py` | Metrics comparison for 3x3 models |
| `clip_ndvi35_rf_combined_3x3.py` | NDVI masking for 3x3 outputs |
| `clamp_esp_nonneg_3x3.py` | ESP clamping for 3x3 outputs |
| `apply_bias_to_ndvi_clips_3x3.py` | Bias correction for 3x3 outputs |

### gpr/ (9 scripts)

Gaussian Process Regression with Matern kernels — tested as a baseline but underperformed RF.

| Script | Description |
|--------|-------------|
| `gpr_pipeline.py` | Complete GPR pipeline: GEE download, Matern kernel, LOOCV, raster prediction |
| `gpr_loocv_metrics_refresh.py` | Refreshed LOOCV metrics |
| `step1_sample_merge.py` | Data preparation for GPR (merge soil tests with embeddings) |
| `gpr_train_predict_full.py` | Full GPR training with all soil tests |
| `step1_sample_full_soiltests.py` | Data preparation (full soil test set) |
| `apply_ndvi_mask_and_mosaic*.py` | NDVI masking and mosaic creation |
| `clip_gpr_to_boundaries*.py` | Clip GPR predictions to paddock boundaries |

### classical_ml/ (13 scripts)

National-scale classical ML models trained on ANSIS data: SVR, Random Forest, Extra Trees, Cubist.

| Script | Description |
|--------|-------------|
| `ph_model_search.py` | Feature selection + model search for pH (RF, ExtraTrees, RFE) |
| `svr_alphaearth_top10cm.py` | SVR with RBF kernel, grid search for pH/CEC/ESP/Na |
| `svr_model_search*.py` | SVR hyperparameter search variants (3 scripts) |
| `cubist_model_search_with400.py` | Cubist rule-based regression with grid search |
| `alphaearth_models_top10cm.py` | Multi-model comparison on ANSIS top-10cm data |
| `esp_models_alphaearth_ansis_top10cm.py` | ESP-specific model comparison on ANSIS data |
| `na_models_alphaearth_top10cm.py` | Na prediction models |
| `na_models_alphaearth_top10cm_physics.py` | Na models with physics-based features |
| `esp_derived_vs_direct.py` | ESP direct prediction vs derived from Na/CEC ratio (SVR LOOCV comparison) |
| `run_models_alphaearth_merged.py` | Models on merged AlphaEarth by-attribute data |
| `merge_alphaearth_by_attribute.py` | Merge per-year AlphaEarth data by soil attribute |

### calibration/ (9 scripts)

National-to-local calibration: base model trained on national ANSIS data, CatBoost calibrator fine-tuned on local farm data.

| Script | Description |
|--------|-------------|
| `national_local_calibration.py` | Base calibration pipeline |
| `national_local_calibration_calib_catboost.py` | Two-level: best base model + CatBoost calibrator (6 model types grid searched) |
| `national_local_calibration_calib_catboost_by_year.py` | Year-specific calibration |
| `national_local_calibration_calib_catboost_clipped.py` | Clipped variant (removes outliers) |
| `national_local_calibration_catboost_base.py` | CatBoost as base model |
| `national_local_calibration_catboost_base_ph_cec.py` | CatBoost base for pH and CEC only |
| `national_local_calibration_recompute.py` | Recomputed metrics |
| `national_local_calibration_recompute_cb_cubist.py` | CatBoost + Cubist variant |
| `train_rfe_catboost_local_calibration.py` | RFE + CatBoost local calibration |

### gee_scripts/ (9 scripts)

Google Earth Engine AlphaEarth embedding extraction scripts.

| Script | Description |
|--------|-------------|
| `gee_sample_alphaearth_by_year.py` | Sample embeddings per year (2017+), produces annual CSVs |
| `gee_sample_alphaearth_top10cm.py` | 5-year median composite for top-10cm soil data |
| `gee_sample_alphaearth_local_points.py` | Sample at local farm points |
| `gee_sample_alphaearth_local_points_2024.py` | 2024-specific sampling |
| `gee_sample_alphaearth_ansis_esp_top10cm.py` | ANSIS ESP samples |
| `gee_sample_alphaearth_ansis_na_top10cm.py` | ANSIS Na samples |
| `gee_sample_alphaearth_top10cm_physics.py` | With additional physics-based bands |
| `gee_test_small.py` | Small test/debug script |
| `gee_sample_alphaearth_by_year_metrics.py` | By-year sampling for per-attribute metrics |

### data_pipeline/ (25+ scripts)

ANSIS/TERN data extraction, ESPADE database extraction, cleaning, and standardization.

| Script | Description |
|--------|-------------|
| `extract_soil_data.py` | Main consolidator: merges AlphaEarth + soil CSVs + shapefiles into annual exports |
| `_extract_soil_data.py` / `_extract_soil_data_15cm.py` | ANSIS API extraction (original and 15cm variant) |
| `add_alphaearth_embeddings.py` | GEE AlphaEarth sampling for soil points (single + batch mode, multi-threaded) |
| `sample_gee_covariates.py` | GEE covariate sampling |
| `extract_espade_*.py` (4 scripts) | ESPADE database extraction variants |
| `download_espade_*.py` (2 scripts) | ESPADE download scripts |
| `extract_ph_top10cm_by_year.py` / `extract_top10cm_ph_cec_esp.py` | ESPADE data extraction |
| `build_soil_points_all_no400.py` | Build master soil points dataset (excluding paddock 400) |
| `clean_*.py` (4 scripts) | Data cleaning: ANSIS ESP, Na, soil points standardization |
| `ansis_extract_*.py` (3 scripts) | ANSIS data extraction: ESP, Na, lab measurements |
| `clip_national_ph_esp*.py` (2 scripts) | National data clipping/filtering |
| `scan_outliers_ph_esp.py` | Outlier detection for pH/ESP |
| `_count_dups_by_site.py`, `_latlon_dups.py`, etc. | Data quality analysis scripts |

### farm_applications/ (12 scripts)

Operational farm outputs: lime/gypsum rates, cLHS sampling design.

| Script | Description |
|--------|-------------|
| `lime_rate_ph55.py` | Lime rate (t/ha) to raise pH to 5.5 using pH and CEC maps |
| `lime_rate_totals.py` | Total lime per paddock (area × rate) |
| `lime_rate_totals_geodesic.py` | Geodesic-corrected pixel areas for lime totals |
| `lime_rate_transform.py` | Lime rate transformation |
| `lime_rate_debug.py` | Lime rate debugging |
| `gypsum_rate_esp6.py` | Gypsum rate (t/ha) to reduce ESP to 6% |
| `lime_gypsum_totals_all_paddocks.py` | Combined lime + gypsum totals for all paddocks |
| `clhs_sampling_points.py` | Base cLHS sampling (50 points, PCA, simulated annealing) |
| `clhs_sampling_points_inside_boundaries.py` | cLHS with boundary constraints |
| `clhs_sampling_points_with_boundaries.py` | cLHS with paddock boundary enforcement |
| `clhs_sampling_points_ndvi30_existing.py` | cLHS with NDVI ≤ 0.30 mask + 20m buffer around existing samples |
| `fix_kml_ndvi30.py` | KML export fix for NDVI-filtered sampling |

### predictions/ (8 scripts)

Raster prediction and mosaic generation scripts.

| Script | Description |
|--------|-------------|
| `predict_paddock_medians_from_embeddings.py` | Paddock-level median predictions from embeddings |
| `predict_rasters_mosaic_from_embeddings.py` | Per-paddock rasters + farm mosaics (2024, trained models: SVR/CatBoost/RF) |
| `predict_rasters_mosaic_multiyear.py` | Multi-year raster predictions |
| `predict_rasters_mosaic_rfe_calib_multiyear.py` | Multi-year with RFE + calibration |
| `evaluate_local_accuracy_from_mosaics.py` | Evaluate local accuracy by sampling mosaic at soil test points |
| `evaluate_local_accuracy_from_rfe_calib_mosaics.py` | Same for RFE-calibrated mosaics |
| `compute_mean_mosaic_from_yearly.py` | Average multi-year mosaics |
| `smooth_boundaries.py` | Boundary smoothing for output rasters |

---

## Script Count Summary

| Category | Scripts |
|----------|---------|
| ResNet ensemble | 5 |
| Random Forest (1x1) | 20+ |
| Random Forest (3x3) | 6 |
| GPR | 9 |
| Classical ML | 13 |
| Calibration | 9 |
| GEE scripts | 9 |
| Data pipeline | 25+ |
| Farm applications | 12 |
| Predictions | 8 |
| **Total** | **~116** |

## Source Locations

All scripts originate from `C:\Users\jackc\Documents\SOIL AI TRAINED MODELS\`:

- `SOIL Tests/exports/resnet_ensemble/` — ResNet code
- `SOIL Tests/exports/rf_combined_ph_cec_esp/` — RF 1x1
- `SOIL Tests/exports/rf_combined_ph_cec_esp_3x3/` — RF 3x3
- `SOIL Tests/exports/gpr_alphaearth/` — GPR
- `SOIL Tests/exports/clhs_sampling/` — cLHS
- `SOIL Testing model Data/scripts/` — National models, GEE, calibration
- `Soil Data/` — Data extraction pipeline
