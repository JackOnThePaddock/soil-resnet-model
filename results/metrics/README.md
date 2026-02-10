# Results Metrics

All model evaluation CSVs, organized by model type and experiment.

## ResNet Ensemble (National Model — 3,534 training samples)

| File | Description |
|------|-------------|
| `holdout_metrics_ensemble.csv` | Ensemble mean R², RMSE, MAE on 80/20 random holdout (in-sample) |
| `holdout_metrics_per_model.csv` | Per-model (1–5) holdout metrics showing ensemble consistency |

## Random Forest — Local Farm (Combined 1x1, 69 samples)

| File | Description |
|------|-------------|
| `rf_combined_1x1_cv_summary.csv` | 5-fold CV and LOPO metrics (all-64 vs best-bands) |
| `rf_combined_1x1_loocv_summary.csv` | LOOCV metrics for pH, CEC, ESP |
| `rf_combined_1x1_loocv_bias_corrected.csv` | LOOCV with paddock mean error correction |
| `rf_combined_1x1_paddock_holdout.csv` | Leave-one-paddock-out (400, HILLPDK, Lightning Tree) |
| `rf_combined_1x1_paddock_bias.csv` | Mean bias per paddock from LOOCV |
| `rf_combined_1x1_best_features.csv` | RFE-selected bands per target |
| `rf_combined_1x1_feature_count.csv` | Feature count vs accuracy sweep |

## Random Forest — Local Farm (Combined 3x3, 69 samples)

| File | Description |
|------|-------------|
| `rf_combined_3x3_cv_summary.csv` | 5-fold CV and LOPO for 3x3-smoothed inputs |
| `rf_combined_3x3_paddock_holdout.csv` | Paddock holdout for 3x3 |
| `rf_combined_3x3_paddock_bias.csv` | Paddock bias for 3x3 |
| `rf_combined_3x3_best_features.csv` | Best bands for 3x3 |
| `rf_combined_3x3_feature_count.csv` | Feature count sweep for 3x3 |

## Random Forest — Hill + Lightning Only (60 samples)

| File | Description |
|------|-------------|
| `rf_hill_lightning_cv_summary.csv` | All CV schemes for Hill+Lightning (no paddock 400) |
| `rf_hill_lightning_paddock_holdout.csv` | LOPO between Hill and Lightning |
| `rf_hill_lightning_best_features.csv` | Best features (Hill+Lightning) |
| `rf_hill_lightning_feature_count.csv` | Feature count sweep |

## Gaussian Process Regression

| File | Description |
|------|-------------|
| `gpr_cv_metrics.csv` | GPR LOOCV metrics (combined dataset) |
| `gpr_cv_metrics_refresh.csv` | Refreshed GPR metrics |
| `gpr_cv_metrics_full.csv` | GPR with full soil test set |

## Na-Specific Models

| File | Description |
|------|-------------|
| `na_rf_loocv_metrics.csv` | Na RF LOOCV (Hill+Lightning) |
| `na_rf_loocv_metrics_400.csv` | Na RF LOOCV (including 400) |

## National Classical ML Models (ANSIS data)

| File | Description |
|------|-------------|
| `national_alphaearth_model_metrics.csv` | RF/ExtraTrees/SVR on ANSIS top-10cm |
| `national_svr_metrics.csv` | SVR grid search results |
| `national_cubist_metrics.csv` | Cubist model results |

## National-to-Local Calibration

| File | Description |
|------|-------------|
| `calibration_catboost.csv` | Two-level calibration: base model + CatBoost fine-tuning |
| `calibration_catboost_base.csv` | CatBoost as base model |
| `ansis_calibration_metrics.csv` | ANSIS-based calibration |
| `aus_soil_db_cv_metrics.csv` | Australian Soil DB model metrics |
| `aus_soil_filtered_metrics.csv` | Filtered/calibrated Aus Soil DB metrics |
| `rf_minmax_calibration_summary.csv` | Min-max calibration approach |

## ESP Experiments

| File | Description |
|------|-------------|
| `esp_derived_vs_direct.csv` | Direct ESP prediction vs derived from Na/CEC (SVR LOOCV) |

## Feature Selection

| File | Description |
|------|-------------|
| `rfe_dimensionality_sweep.csv` | RFE feature count sweep |
| `rfe_ranking_ph.csv` | Band importance ranking for pH |
| `rfe_ranking_cec.csv` | Band importance ranking for CEC |

## Farm Applications

| File | Description |
|------|-------------|
| `lime_gypsum_totals.csv` | Lime and gypsum requirements per paddock |
| `clhs_sampling_50_ndvi30.csv` | 50 cLHS sampling points (NDVI ≤ 0.30, boundary-constrained) |
| `clhs_pca_variance.csv` | PCA variance explained (used in cLHS) |
| `local_accuracy_from_mosaics.csv` | Accuracy at soil test points sampled from prediction mosaics |
