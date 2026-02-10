# Results

## Training Data Metrics (Ensemble Mean)

These metrics were computed on the **same data used for training** (3,534 samples). Because the final ensemble models were each trained on the full dataset after cross-validation, these results reflect in-sample performance and are **optimistically biased**.

| Target | N | R² | RMSE | MAE | Unit |
|--------|---|-----|------|-----|------|
| pH | 3,511 | 0.815 | 0.446 | 0.337 | pH |
| CEC | 694 | 0.952 | 3.321 | 2.425 | cmol/kg |
| ESP | 563 | 0.785 | 1.367 | 0.827 | % |
| SOC | 755 | 0.904 | 1.193 | 0.727 | % |
| Ca | 1,823 | 0.918 | 2.435 | 1.752 | cmol/kg |
| Mg | 1,775 | 0.924 | 1.243 | 0.866 | cmol/kg |
| Na | 1,309 | 0.894 | 0.190 | 0.102 | cmol/kg |

---

## Independent Validation

To assess true generalisation, the ensemble was evaluated on data that was **never seen during training** (zero feature-vector overlap confirmed).

### National Independent Dataset (1,368 ANSIS samples)

Source: ANSIS top-10cm soil cores with AlphaEarth 5-year median embeddings, excluding any overlap with the training set.

| Target | N | R² | RMSE | MAE | Unit |
|--------|---|-----|------|-----|------|
| pH | 1,367 | 0.606 | 0.577 | 0.448 | pH |
| CEC | 364 | 0.817 | 8.022 | 5.533 | cmol/kg |
| ESP | 380 | 0.363 | 2.257 | 1.541 | % |
| Na | 988 | 0.559 | 0.449 | 0.197 | cmol/kg |

SOC, Ca, and Mg were not available in this dataset.

### Speirs Farm — Real Lab Results (60 samples)

Source: Independent farm-level soil testing (Hill Paddock: 33 samples, Lightning Tree: 27 samples) with laboratory analysis. This tests the model's ability to predict at a specific site it has never seen.

| Target | N | R² | RMSE | MAE | Unit |
|--------|---|-----|------|-----|------|
| pH | 60 | -0.914 | 0.550 | 0.455 | pH |
| CEC | 60 | 0.214 | 2.754 | 1.735 | cmol/kg |
| ESP | 60 | 0.464 | 2.110 | 1.801 | % |
| Na | 60 | 0.536 | 0.244 | 0.193 | cmol/kg |

Per-paddock breakdown:

| Paddock | pH R² | CEC R² | ESP R² | Na R² |
|---------|-------|--------|--------|-------|
| Hill Paddock (33) | -0.509 | 0.105 | 0.430 | 0.605 |
| Lightning Tree (27) | -1.747 | 0.179 | -0.108 | 0.200 |

The negative pH R² indicates the model predicts worse than the mean for this farm. The Speirs farm has predominantly acidic soils (pH 4.3–7.0) that are underrepresented in the national training data.

---

## Overfitting Analysis

| Target | Training R² | Independent R² | Drop |
|--------|-------------|----------------|------|
| pH | 0.815 | 0.606 | -0.209 |
| CEC | 0.952 | 0.817 | -0.135 |
| ESP | 0.785 | 0.363 | -0.422 |
| Na | 0.894 | 0.559 | -0.335 |

The model shows significant overfitting, with R² dropping 0.13–0.42 on truly independent data. This is because the final ensemble models were trained on the entire dataset (all 3,534 samples) — the K-fold cross-validation was used only during training to select checkpoints, and then the final saved models were retrained on all data.

---

## ESP: Direct Prediction vs Derived from Na/CEC

Since ESP = (Na / CEC) × 100, we compared the model's direct ESP prediction head against computing ESP from the Na and CEC prediction heads.

| Dataset | Direct ESP R² | Derived (Na/CEC) R² | Winner |
|---------|---------------|---------------------|--------|
| National Independent (380) | 0.363 | 0.277 | Direct |
| Speirs Farm (60) | 0.464 | 0.552 | Derived |
| Speirs — Hill Paddock (33) | 0.430 | 0.422 | ~Tied |
| Speirs — Lightning Tree (27) | -0.108 | 0.218 | Derived |

The derived ESP approach helps at the farm level, particularly for Lightning Tree where the direct head gives negative R² but Na/CEC gives 0.22. Neither approach provides reliable ESP on unseen data.

---

## Per-Model Performance (80/20 Random Split)

Individual model performance on a random 20% holdout (707 samples). Note: this split may partially overlap with training data due to the full-data retraining approach.

| Model | pH | CEC | ESP | SOC | Ca | Mg | Na | Avg R² |
|-------|-----|------|------|------|------|------|------|--------|
| 1 | 0.795 | 0.926 | 0.676 | 0.904 | 0.887 | 0.861 | 0.758 | 0.830 |
| 2 | 0.791 | 0.912 | 0.663 | 0.929 | 0.880 | 0.892 | 0.774 | 0.835 |
| 3 | 0.784 | 0.924 | 0.668 | 0.923 | 0.879 | 0.876 | 0.792 | 0.835 |
| 4 | 0.791 | 0.935 | 0.705 | 0.893 | 0.884 | 0.879 | 0.806 | 0.842 |
| 5 | 0.787 | 0.935 | 0.666 | 0.926 | 0.904 | 0.896 | 0.793 | 0.844 |

The five models are consistent (avg R² 0.830–0.844), confirming the ensemble provides meaningful diversity through different random seeds.

---

## Key Findings

1. **CEC is the strongest generaliser** — R² = 0.817 on independent national data, the only target that remains robust.
2. **pH is moderate nationally (R² = 0.606) but fails on specific farms** — the Speirs farm has very acidic soils outside the training distribution.
3. **ESP is unreliable on unseen data** — R² = 0.363 nationally, driven by the difficulty of predicting a ratio of two uncertain quantities.
4. **The originally reported holdout metrics (R² 0.81–0.96) were inflated** due to the final models being trained on all data, including the holdout.
5. **National-to-local calibration is essential** — the raw national model is insufficiently accurate for site-specific farm applications without local calibration data.

Regenerate this file: `python scripts/generate_report.py`
