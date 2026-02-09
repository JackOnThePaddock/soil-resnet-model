# NationalSoilNet -- Methodology

This document provides a detailed account of the data collection, feature engineering, model architecture, training procedure, baseline comparisons, calibration strategy, uncertainty quantification, and evaluation approach used in the NationalSoilNet system.

---

## Table of Contents

1. [Data Collection](#1-data-collection)
2. [Feature Engineering](#2-feature-engineering)
3. [Model Architecture](#3-model-architecture)
4. [Training Procedure](#4-training-procedure)
5. [Baseline Models](#5-baseline-models)
6. [National-to-Local Calibration](#6-national-to-local-calibration)
7. [Uncertainty Quantification](#7-uncertainty-quantification)
8. [Evaluation](#8-evaluation)
9. [Applications](#9-applications)

---

## 1. Data Collection

### Source

Soil samples were obtained from the Australian National Soil Information System (ANSIS) via the TERN Soil Data API. The TERN portal provides programmatic access to harmonised laboratory measurements from multiple state and federal soil survey programs.

### Sample Selection Criteria

- **Depth interval**: 0--15 cm (topsoil). All samples with upper depth between 0 and 5 cm and lower depth between 10 and 20 cm were included and attributed to the 0--15 cm layer. Samples outside this range were excluded.
- **Temporal window**: 2017--2025. This window aligns with the availability period of AlphaEarth satellite embeddings and ensures that land use conditions at the time of soil sampling are reasonably consistent with the satellite imagery.
- **Geographic extent**: Continental Australia, spanning arid, semi-arid, temperate, subtropical, and tropical climate zones.
- **Final dataset size**: 2,625 unique sample locations after deduplication, quality filtering, and removal of records with no valid target values.

### Target Variables

Seven soil properties were extracted as prediction targets:

| Property | Abbreviation | Unit | Description |
|----------|-------------|------|-------------|
| pH (CaCl2) | pH | -- | Soil acidity measured in calcium chloride solution |
| Cation Exchange Capacity | CEC | cmol(+)/kg | Total capacity of soil to hold exchangeable cations |
| Exchangeable Sodium Percentage | ESP | % | Proportion of CEC occupied by sodium; indicator of sodicity |
| Soil Organic Carbon | SOC | % | Organic carbon content by weight |
| Exchangeable Calcium | Ca | cmol(+)/kg | Plant-available calcium |
| Exchangeable Magnesium | Mg | cmol(+)/kg | Plant-available magnesium |
| Exchangeable Sodium | Na | cmol(+)/kg | Exchangeable sodium contributing to sodicity |

Not all targets are available for every sample. Missing values are handled at training time via a masked loss function (see Section 4).

### Automated Extraction

Data extraction was automated using the TERN Soil Data API. The pipeline queries sample locations within bounding boxes, filters by depth and date, harmonises measurement methods where needed, and exports a consolidated Parquet file with coordinates, target values, and metadata.

---

## 2. Feature Engineering

### AlphaEarth Satellite Embeddings

The sole input features are 64-band embedding vectors derived from the AlphaEarth foundation model. AlphaEarth is a self-supervised vision model trained on large-scale Google Earth Engine imagery. Its embeddings capture spectral, spatial, and temporal patterns in a compact 64-dimensional representation per pixel.

### Composite Construction

For each soil sample location, a five-year median composite of AlphaEarth embeddings was constructed:

1. All available AlphaEarth scenes within a five-year window centred on the soil sampling date were collected.
2. A per-band pixel-wise median was computed to produce a single 64-band composite, reducing noise from clouds, shadows, and seasonal variation.
3. The median composite was sampled at the exact geographic coordinates of each soil sample.

### NDVI Vegetation Masking

To reduce interference from standing crop and pasture biomass (which reflects vegetation state rather than underlying soil properties), an NDVI-based vegetation mask was applied:

1. NDVI was computed from the corresponding Sentinel-2 or Landsat imagery.
2. Scenes where NDVI at the sample location exceeded a threshold (indicating dense green vegetation cover) were excluded from the composite.
3. This biases the composite towards bare-soil or low-vegetation conditions, improving the signal-to-noise ratio for soil property prediction.

### Feature Summary

Each sample is represented by a single 64-dimensional vector (the AlphaEarth embedding composite). No hand-engineered spectral indices, terrain derivatives, or climate covariates are included -- the model learns all necessary representations from the embedding space.

---

## 3. Model Architecture

### NationalSoilNet

NationalSoilNet is a multi-target residual network with the following structure:

```
Input: x in R^64
    |
    v
Linear(64 -> 128)
BatchNorm1d(128)
SiLU()
    |
    v
ResidualBlock_1(128)
    |
    v
ResidualBlock_2(128)
    |
    v
7 parallel output heads: Linear(128 -> 1) each
    |
    v
Output: [pH, CEC, ESP, SOC, Ca, Mg, Na]
```

### Residual Block

Each `ResidualBlock` implements the following computation:

```
Input: x in R^128
    |
    +---> identity (skip connection)
    |
    v
Linear(128 -> 128)
BatchNorm1d(128)
SiLU()
Dropout(p)
Linear(128 -> 128)
BatchNorm1d(128)
    |
    + identity  (element-wise addition)
    |
    v
SiLU()
    |
    v
Output: x' in R^128
```

The skip connection allows gradients to flow directly through the block, enabling effective training despite the depth of the network. Dropout is applied between the two linear layers within each block to regularise the learned representations.

### Output Heads

Seven independent linear heads project the shared 128-dimensional representation to scalar predictions:

- `head_ph`: Linear(128 -> 1)
- `head_cec`: Linear(128 -> 1)
- `head_esp`: Linear(128 -> 1)
- `head_soc`: Linear(128 -> 1)
- `head_ca`: Linear(128 -> 1)
- `head_mg`: Linear(128 -> 1)
- `head_na`: Linear(128 -> 1)

This multi-target design enables the shared backbone to learn correlations between soil properties (for example, the relationship between CEC, Ca, Mg, and Na) while keeping predictions independent at the output layer.

### Design Rationale

- **Residual connections** prevent degradation in deeper networks and improve gradient flow.
- **Batch normalisation** stabilises training by reducing internal covariate shift.
- **SiLU activation** (Sigmoid Linear Unit) provides smooth, non-monotonic activation that has been shown to outperform ReLU in similar regression tasks.
- **Multi-target architecture** is more data-efficient than training seven separate models, as the shared backbone benefits from information across all targets.

---

## 4. Training Procedure

### Ensemble Strategy

The final model is an ensemble of five independently trained NationalSoilNet instances. Each member is initialised with a different random seed, producing diversity in learned representations. Predictions are aggregated by averaging across the five models.

### Cross-Validation

Within each ensemble member, five-fold cross-validation is used:

1. The training data (excluding the holdout test set) is split into five stratified folds.
2. Each fold is used once as a validation set while the remaining four folds are used for training.
3. The best model checkpoint (by validation loss) from each fold is retained.
4. At inference time, predictions from all five folds of all five ensemble members (25 models total) can be averaged, or a single best-fold model per ensemble member can be used.

### Optimiser and Learning Rate

- **Optimiser**: AdamW with learning rate `1e-4` and weight decay `1e-5`. AdamW decouples weight decay from the gradient update, providing more consistent regularisation than L2 penalty in Adam.
- **Learning rate scheduler**: `ReduceLROnPlateau` monitoring validation loss, with patience of 20 epochs and a reduction factor of 0.5. The learning rate is halved each time validation loss fails to improve for 20 consecutive epochs.

### Regularisation

- **Early stopping**: Training terminates if validation loss does not improve for 50 consecutive epochs. The model checkpoint with the lowest validation loss is restored.
- **Gradient clipping**: Gradients are clipped to a maximum norm of 1.0 to prevent exploding gradients.
- **Dropout**: Applied within each residual block.
- **Weight decay**: Applied via AdamW (1e-5).

### Loss Function

The loss function is a **masked mean squared error (MSE)**. Because not all samples have measurements for all seven targets, the loss is computed only over targets where ground truth values are available:

```
L = (1 / N_valid) * sum_{i in valid} (y_i - y_hat_i)^2
```

where `N_valid` is the number of non-missing target values in the batch, and the sum runs only over targets with valid measurements. This approach allows the model to learn from partial observations without imputing missing values.

### Feature Normalisation

All 64 input features are normalised using `StandardScaler` (zero mean, unit variance), fitted on the training split only. The scaler parameters are saved alongside each model checkpoint to ensure consistent normalisation at inference time. Target values are not normalised.

### Training Budget

Each model is trained for up to 1,000 epochs, though early stopping typically terminates training well before this limit (typically between 200 and 500 epochs depending on the fold and target data availability).

### Hyperparameter Summary

| Parameter | Value |
|-----------|-------|
| Input dimension | 64 |
| Hidden dimension | 128 |
| Number of residual blocks | 2 |
| Number of output heads | 7 |
| Activation function | SiLU |
| Optimiser | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau |
| LR scheduler patience | 20 epochs |
| LR scheduler factor | 0.5 |
| Early stopping patience | 50 epochs |
| Gradient clipping max norm | 1.0 |
| Maximum epochs | 1000 |
| Ensemble size | 5 models |
| Cross-validation folds | 5 |
| Feature normalisation | StandardScaler |
| Loss function | Masked MSE |

---

## 5. Baseline Models

NationalSoilNet was compared against five conventional machine learning baselines, all trained on the same 64-dimensional AlphaEarth features:

### Support Vector Regression (SVR)

- Radial Basis Function (RBF) kernel
- Hyperparameters (C, gamma, epsilon) tuned via grid search with cross-validation
- Separate model trained per target variable

### Random Forest

- 500 estimators
- Maximum depth and minimum samples per leaf tuned via cross-validation
- Separate model per target

### Extra Trees

- 500 estimators with the same tuning approach as Random Forest
- Extra Trees use random split thresholds, providing additional variance reduction

### Cubist

- Rule-based regression model with committees and instance-based corrections
- Particularly well-suited to soil science applications where interpretability is valued

### Gaussian Process Regression (GPR)

- Matern kernel (nu=2.5) with automatic relevance determination
- Provides native uncertainty estimates
- Computationally limited to moderate dataset sizes

In per-target comparisons, NationalSoilNet consistently matched or exceeded the best baseline for each target, with the largest improvements on CEC (R-squared = 0.959 versus 0.91 for the best baseline) and Ca (R-squared = 0.929 versus 0.87).

---

## 6. National-to-Local Calibration

### Motivation

A nationally trained model captures broad-scale soil-landscape relationships but may exhibit systematic bias when applied to a specific farm or region. Local calibration corrects for this bias using a small number of site-specific soil samples.

### Approach

The calibration step uses CatBoost (gradient-boosted decision trees) as a second-stage model:

1. The national ensemble generates predictions for all available local calibration samples.
2. A CatBoost model is trained to map national predictions (and optionally their uncertainty) to observed local values.
3. The calibrated model is then applied to all prediction locations on the farm.

### Why CatBoost

CatBoost was selected for calibration because:

- It handles small datasets well without overfitting (typical local calibration sets have 20--100 samples).
- It natively supports ordered boosting, which reduces prediction shift.
- It requires minimal hyperparameter tuning for tabular regression tasks.

### Calibration Workflow

1. Collect local soil samples with laboratory analysis (minimum 15--20 recommended).
2. Extract AlphaEarth embeddings for the local sample locations.
3. Generate national model predictions and uncertainty for those locations.
4. Train a CatBoost calibration model mapping national predictions to local observations.
5. Apply the calibrated pipeline to all target locations (e.g., a raster grid across the farm).

---

## 7. Uncertainty Quantification

### Ensemble Disagreement

Predictive uncertainty is estimated via the standard deviation of predictions across the five ensemble members:

```
mu(x) = (1/5) * sum_{k=1}^{5} f_k(x)
sigma(x) = sqrt( (1/5) * sum_{k=1}^{5} (f_k(x) - mu(x))^2 )
```

where `f_k(x)` is the prediction of the k-th ensemble member for input `x`.

### Interpretation

- **Low standard deviation**: The five models agree, suggesting the input is well-represented in the training distribution and the prediction is reliable.
- **High standard deviation**: The models disagree, suggesting the input may be in a region of feature space with sparse training data, or the soil properties at that location are inherently variable.

### Usage in Practice

Uncertainty maps are generated alongside prediction maps for each soil property. These are used to:

- Identify areas requiring additional soil sampling.
- Weight predictions in downstream agronomic calculations.
- Inform confidence intervals for management recommendations.

---

## 8. Evaluation

### Holdout Test Set

A stratified holdout set was reserved before any model training. Stratification was performed by geographic region and target value distribution to ensure the test set is representative of the full data range. This set was never used for training, validation, or hyperparameter tuning.

### Metrics

Three metrics are reported for each target:

- **R-squared (R2)**: Coefficient of determination. Measures the proportion of variance in the target explained by the model. Values closer to 1.0 indicate better fit.
- **RMSE (Root Mean Squared Error)**: Square root of the mean squared prediction error. Penalises large errors more heavily than MAE.
- **MAE (Mean Absolute Error)**: Mean of absolute prediction errors. More robust to outliers than RMSE.

### Per-Target Evaluation

Because data availability varies across targets (e.g., 527 test samples for CEC but only 123 for Ca), metrics are computed independently for each target on its available test samples.

### Holdout Results

| Target | N Test | R-squared | RMSE   | MAE   |
|--------|--------|-----------|--------|-------|
| pH     | 133    | 0.809     | 0.638  | 0.421 |
| CEC    | 527    | 0.959     | 3.948  | 2.101 |
| ESP    | 524    | 0.841     | 10.327 | 3.994 |
| SOC    | 527    | 0.870     | 0.557  | 0.320 |
| Ca     | 123    | 0.929     | 4.905  | 2.685 |
| Mg     | 123    | 0.926     | 3.587  | 1.966 |
| Na     | 123    | 0.805     | 2.267  | 1.134 |

### Paddock-Holdout Cross-Validation

In addition to the standard holdout evaluation, a spatial cross-validation scheme was employed where entire paddocks (contiguous management units) were held out. This tests the model's ability to generalise to new spatial locations rather than interpolating between nearby training samples. Results from paddock-holdout CV are reported separately and provide a more conservative estimate of real-world predictive performance.

---

## 9. Applications

### Farm-Scale Raster Prediction

The primary application is generating continuous soil property maps across farm properties:

1. AlphaEarth embeddings are extracted on a regular grid (e.g., 10 m resolution) covering the farm.
2. The national ensemble (optionally calibrated) predicts all seven soil properties at each grid cell.
3. Ensemble uncertainty is computed at each cell.
4. Results are exported as GeoTIFF rasters for integration with GIS and precision agriculture platforms.

### Lime Rate Calculation

Predicted pH, CEC, and Ca values feed into agronomic lime requirement models:

- The Shoemaker-McLean-Pratt (SMP) buffer method or equivalent is applied using predicted soil chemistry.
- Lime rates (tonnes/ha of calcium carbonate equivalent) are computed spatially, enabling variable-rate lime application maps.

### Gypsum Rate Calculation

For sodic soils, predicted ESP, CEC, Ca, and Na values are used to calculate gypsum (calcium sulfate) requirements:

- The target ESP reduction is specified (e.g., from current ESP to below 6%).
- Gypsum rate is computed as a function of CEC, current ESP, target ESP, and soil bulk density.
- Spatial gypsum maps enable targeted amelioration of sodic areas.

### Conditioned Latin Hypercube Sampling (cLHS)

Predicted soil property maps and their uncertainties are used to design optimal soil sampling campaigns:

- cLHS selects sample locations that representatively cover the multivariate feature space.
- Uncertainty layers can be incorporated to preferentially sample areas where model confidence is low.
- This closes the loop: model predictions guide where to sample next, and new samples improve the model.
