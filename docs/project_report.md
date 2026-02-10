# High-Resolution Mapping of pH (CaCl2), CEC, and ESP Using AlphaEarth Embeddings and Random Forest Regression in Mixed Cropping Paddocks, NSW, Australia

## Abstract
This project developed a reproducible pipeline to map soil pH (CaCl2), cation exchange capacity (CEC), and exchangeable sodium percentage (ESP) at 10 m resolution using Google AlphaEarth embeddings and a small set of field soil tests. Soil test points (n=69) from three paddocks (HILLPDK, LIGHTNING TREE, and 400) were combined with 64-band AlphaEarth 5-year median embeddings (2020–2024). Random Forest (RF) regressors were trained for each soil property with recursive feature elimination (RFE) to mitigate overfitting in small-sample settings. The workflow included cross-validation (LOOCV, 5-fold CV, and leave-one-paddock-out), per-paddock bias estimation, NDVI-based masking of non-bare soil, and spatial clipping to paddock boundaries. Best-band RF models improved predictive accuracy versus full-band models across all targets. Bias correction provided small but consistent gains. The outputs are paddock-level GeoTIFFs suitable for variable-rate lime and gypsum decision support, with transparent limitations around cross-paddock transferability.

**Keywords:** AlphaEarth embeddings, Random Forest, soil mapping, pH (CaCl2), CEC, ESP, NDVI masking, variable-rate lime, gypsum, Australia

## 1. Introduction
Spatial variability in soil acidity and sodicity drives productivity and amelioration requirements in Australian broadacre systems. Traditional interpolation of sparse soil tests can under-represent fine-scale spatial patterns. Recent satellite foundation models such as AlphaEarth provide stable multi-temporal embeddings that can be paired with local soil tests to generate predictive surfaces. This thesis evaluates whether AlphaEarth embeddings, combined with Random Forest models and minimal sample sizes, can deliver farm-scale maps of pH, CEC, and ESP suitable for variable-rate management.

## 2. Objectives
- Build a reproducible pipeline to predict pH (CaCl2), CEC, and ESP at 10 m resolution using AlphaEarth embeddings and local soil tests.
- Quantify predictive performance using multiple cross-validation regimes, including leave-one-paddock-out.
- Evaluate feature selection and bias correction for improved accuracy and practical mapping outputs.
- Produce paddock-clipped, NDVI-masked GeoTIFFs for operational use.

## 3. Study Area and Data
- **Paddocks:** HILLPDK, LIGHTNING TREE, and 400.
- **Soil tests:** 69 total points (HILLPDK n=33, LIGHTNING TREE n=27, 400 n=9).
- **Targets:** pH (CaCl2), CEC (cmol+/kg), ESP (%).
- **ESP calculation:** `ESP = (Na_cmol / CEC) * 100` with Na converted from mg/kg to cmol+/kg using `/ 230` when required.
- **Remote sensing:** AlphaEarth embeddings (64 bands), 5-year median (2020–2024), 10 m.
- **Masking:** Sentinel-2 bare-earth composite used to compute NDVI (red band 1, NIR band 7); NDVI > 0.35 masked to exclude vegetation and tree cover.
- **Boundaries:** Paddock polygons in EPSG:4326.

## 4. Methods
### 4.1 Feature extraction
- AlphaEarth 5-year median embeddings were sampled at soil test points.
- All 64 embedding bands used as candidate predictors.

### 4.2 Model training and feature selection
- Random Forest regressors (300 trees, `max_features="sqrt"`).
- RFE used to rank bands and select optimal feature counts.
- Selected bands: 10 for pH, 12 for CEC, 12 for ESP.

### 4.3 Validation strategies
- **LOOCV:** leave-one-out cross-validation across all samples.
- **5-fold CV:** random folds (shuffle, seed 42).
- **Leave-one-paddock-out (LOPO):** train on two paddocks, test on the third.
- **Per-paddock bias:** mean error per paddock estimated from LOOCV predictions.

### 4.4 Mapping workflow
- Apply trained RF models to AlphaEarth rasters for each paddock.
- Clip to paddock boundaries.
- Apply NDVI mask (NDVI = 0.35 retained).
- Clamp ESP to non-negative values.
- Optional bias correction: subtract paddock mean error from predictions.

## 5. Results
### 5.1 Sample summary
```
Paddock          Samples
HILLPDK          33
LIGHTNING TREE   27
400               9
Total            69
```

### 5.2 Selected bands (best-bands models)
```
pH  (10): A24,A00,A39,A30,A35,A18,A53,A04,A44,A06
CEC (12): A09,A00,A39,A44,A24,A52,A60,A35,A53,A33,A30,A18
ESP (12): A35,A58,A05,A51,A09,A44,A31,A60,A56,A23,A41,A10
```

### 5.3 LOOCV accuracy (all-64 vs best-bands)
```
Target  Feature set  RMSE   MAE    R²
pH      all_64       0.4288 0.2653 0.0781
pH      best_bands   0.3898 0.2262 0.2384
CEC     all_64       3.0361 2.0821 0.2174
CEC     best_bands   2.7493 1.8851 0.3582
ESP     all_64       2.1170 1.7285 0.4319
ESP     best_bands   1.8920 1.5093 0.5462
```

### 5.4 5-fold CV and LOPO (all-64 vs best-bands)
```
Target  Feature set  CV type                 RMSE   MAE    R²
pH      all_64       5-fold                  0.4784 0.3028 -0.1473
pH      all_64       leave-one-paddock-out   0.5074 0.3853 -0.2908
pH      best_bands   5-fold                  0.4285 0.2558  0.0797
pH      best_bands   leave-one-paddock-out   0.3933 0.2403  0.2245

CEC     all_64       5-fold                  3.3873 2.3119  0.0258
CEC     all_64       leave-one-paddock-out   3.8937 3.2468 -0.2872
CEC     best_bands   5-fold                  3.1157 2.0532  0.1758
CEC     best_bands   leave-one-paddock-out   3.3219 2.5958  0.0631

ESP     all_64       5-fold                  2.1571 1.7542  0.4102
ESP     all_64       leave-one-paddock-out   3.1479 2.7431 -0.2561
ESP     best_bands   5-fold                  1.9326 1.5319  0.5266
ESP     best_bands   leave-one-paddock-out   2.8314 2.4483 -0.0162
```

### 5.5 Per-paddock holdout (best-bands)
```
Target  Paddock          RMSE   MAE    R²
pH      400              0.4941 0.3621 -0.0288
pH      HILLPDK          0.4271 0.2092  0.0524
pH      LIGHTNING TREE   0.3017 0.2378  0.2112

CEC     400              4.3894 3.5101 -0.4236
CEC     HILLPDK          3.5169 2.8024 -0.3524
CEC     LIGHTNING TREE   2.5811 2.0385  0.2027

ESP     400              2.6016 2.4041 -0.6244
ESP     HILLPDK          3.2376 2.9196 -0.7469
ESP     LIGHTNING TREE   2.3280 1.8870 -0.0503
```

### 5.6 Bias-corrected LOOCV (best-bands)
```
Target  RMSE   MAE    R²    ?   RMSE   MAE    R² (bias-corrected)
pH      0.3908 0.2269 0.234 ?   0.3875 0.2257 0.247
CEC     2.7512 1.8805 0.357 ?   2.7150 1.8161 0.374
ESP     1.8958 1.5171 0.544 ?   1.8820 1.5009 0.551
```

### 5.7 Mean bias by paddock (best-bands, prediction - actual)
```
Target  400        HILLPDK    LIGHTNING TREE
pH      -0.262     0.031      0.026
CEC     -2.033     1.960     -0.580
ESP     -1.005     2.438     -1.435
```

## 6. Discussion
- Feature selection is critical for small datasets; best-bands models consistently outperformed full-band models across all validation regimes.
- LOPO results reveal limited transferability across paddocks, with negative R² for CEC and ESP in some holdouts. This indicates that embeddings capture local soil-vegetation relationships but extrapolate poorly to new paddocks without local calibration.
- Paddock-specific bias correction modestly improves overall accuracy and aligns predictions with local lab means, which is operationally useful for VR prescriptions.
- NDVI masking removes vegetated/tree areas and improves interpretability but does not directly increase statistical accuracy; it improves spatial plausibility of the outputs.
- ESP values were clamped to non-negative because RF is unconstrained; this is a practical fix but does not address underlying model physics.

## 7. Limitations
- Small and imbalanced sample size (n=69; only n=9 in paddock 400).
- No explicit terrain or soil covariate layers used; only AlphaEarth embeddings.
- Cross-paddock generalization is weak, indicating a need for local calibration per paddock.

## 8. Conclusions
AlphaEarth embeddings combined with RF and RFE can generate usable paddock-scale pH, CEC, and ESP surfaces from sparse soil tests. Within-farm accuracy is moderate (pH RMSE ~0.39; CEC ~2.75; ESP ~1.89) and improves slightly with paddock-level bias correction. Transferability across paddocks remains limited, reinforcing the value of local calibration for commercial-grade mapping.

## 9. Recommendations for Future Work
- Increase stratified sampling in under-represented paddocks (e.g., 400).
- Test constrained modelling for ESP via predicted Na and CEC to enforce physical bounds.
- Evaluate additional covariates (terrain, radiometrics) once sample size increases.
- Incorporate uncertainty mapping for decision-making (e.g., RF variance or quantile forests).

## 10. Data and Code Availability
Outputs are GeoTIFFs for each paddock and target, with NDVI-masked and bias-corrected variants. Scripts and CSVs are stored in the project directory and can be packaged into a reproducible archive on request.
