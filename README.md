# NationalSoilNet: Multi-Target Soil Property Prediction from Satellite Embeddings

NationalSoilNet is a ResNet-based ensemble model that predicts seven key soil properties from 64-band AlphaEarth satellite embeddings. Trained on 2,625 Australian soil samples (0--15 cm depth) sourced from the ANSIS/TERN portal, the model delivers strong predictive performance across pH, CEC, ESP, SOC, Ca, Mg, and Na, with R-squared values ranging from 0.805 to 0.959 on a held-out test set.

The system uses an ensemble of five independently seeded models, each trained with five-fold cross-validation, to produce both point predictions and calibrated uncertainty estimates. A downstream national-to-local calibration step using CatBoost enables adaptation to individual farm sites with minimal local data.

---

## Architecture

The core model, `NationalSoilNet`, is a multi-target residual network with seven parallel output heads -- one per soil property.

```
Input (64 AlphaEarth bands)
    |
    v
Linear(64 -> 128) --> BatchNorm(128) --> SiLU
    |
    v
ResidualBlock(128)
    |   Linear(128 -> 128) --> BatchNorm --> SiLU --> Dropout
    |   Linear(128 -> 128) --> BatchNorm
    |   + skip connection
    |   SiLU
    |
    v
ResidualBlock(128)
    |   Linear(128 -> 128) --> BatchNorm --> SiLU --> Dropout
    |   Linear(128 -> 128) --> BatchNorm
    |   + skip connection
    |   SiLU
    |
    v
+-----+-----+-----+-----+-----+-----+-----+
|     |     |     |     |     |     |     |
v     v     v     v     v     v     v     v
pH   CEC   ESP   SOC   Ca    Mg    Na
(1)  (1)   (1)   (1)   (1)   (1)   (1)

Each output head: Linear(128 -> 1)
```

Each `ResidualBlock` consists of two linear layers with batch normalisation, SiLU activation, and dropout, wrapped with a skip (identity) connection. The multi-target design allows the network to learn shared representations across correlated soil properties while maintaining independent prediction heads.

---

## Results

Holdout test set performance for each soil property target:

| Target | N Test | R-squared | RMSE   | MAE   |
|--------|--------|-----------|--------|-------|
| pH     | 133    | 0.809     | 0.638  | 0.421 |
| CEC    | 527    | 0.959     | 3.948  | 2.101 |
| ESP    | 524    | 0.841     | 10.327 | 3.994 |
| SOC    | 527    | 0.870     | 0.557  | 0.320 |
| Ca     | 123    | 0.929     | 4.905  | 2.685 |
| Mg     | 123    | 0.926     | 3.587  | 1.966 |
| Na     | 123    | 0.805     | 2.267  | 1.134 |

All metrics are computed on a stratified holdout set that was excluded from training and cross-validation. The varying test set sizes reflect per-target data availability, as the model uses masked loss to handle missing values.

---

## Quick Start

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-org/soil-resnet-model.git
cd soil-resnet-model
pip install -e .
```

### Dependencies

Core dependencies include PyTorch, scikit-learn, pandas, NumPy, and CatBoost. See `pyproject.toml` or `requirements.txt` for the full list.

---

## Usage

### Training

Train a full ensemble (5 models, 5-fold CV each):

```python
from soil_resnet.train import train_ensemble

train_ensemble(
    data_path="data/processed/soil_samples.parquet",
    output_dir="models/ensemble",
    n_models=5,
    n_folds=5,
    max_epochs=1000,
    lr=1e-4,
    weight_decay=1e-5,
    patience=50,
)
```

### Evaluation

Evaluate the ensemble on a holdout set:

```python
from soil_resnet.evaluate import evaluate_ensemble

metrics = evaluate_ensemble(
    model_dir="models/ensemble",
    test_data="data/processed/test_set.parquet",
)

for target, m in metrics.items():
    print(f"{target}: R2={m['r2']:.3f}, RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}")
```

### Prediction

Generate predictions (with uncertainty) for new satellite embeddings:

```python
from soil_resnet.predict import predict

results = predict(
    model_dir="models/ensemble",
    embeddings="data/new_site_embeddings.parquet",
)

# results is a DataFrame with columns: pH_mean, pH_std, CEC_mean, CEC_std, ...
print(results.head())
```

### National-to-Local Calibration

Fine-tune predictions for a specific farm using local calibration samples:

```python
from soil_resnet.calibrate import calibrate_to_local

calibrated = calibrate_to_local(
    national_model_dir="models/ensemble",
    local_data="data/farm/local_samples.parquet",
    embeddings="data/farm/paddock_embeddings.parquet",
)
```

---

## Data Sources

- **Soil samples**: 2,625 samples at 0--15 cm depth from the Australian National Soil Information System (ANSIS) via the TERN Soil Data API. Samples span 2017--2025 and cover diverse Australian soil types and land uses.
- **Satellite features**: 64-band AlphaEarth embedding composites extracted from Google Earth Engine. Five-year median composites are used with NDVI-based vegetation masking to reduce crop/pasture signal interference.
- **Targets**: pH (CaCl2), Cation Exchange Capacity (CEC, cmol(+)/kg), Exchangeable Sodium Percentage (ESP, %), Soil Organic Carbon (SOC, %), Exchangeable Calcium (Ca, cmol(+)/kg), Exchangeable Magnesium (Mg, cmol(+)/kg), Exchangeable Sodium (Na, cmol(+)/kg).

---

## Project Structure

```
soil-resnet-model/
|-- README.md                  Project overview and quick start
|-- METHODOLOGY.md             Detailed methodology documentation
|-- pyproject.toml             Package configuration and dependencies
|-- requirements.txt           Pinned dependency versions
|
|-- soil_resnet/
|   |-- __init__.py
|   |-- model.py               NationalSoilNet architecture definition
|   |-- train.py               Ensemble training with k-fold CV
|   |-- evaluate.py            Holdout evaluation and metrics
|   |-- predict.py             Inference with ensemble uncertainty
|   |-- calibrate.py           National-to-local CatBoost calibration
|   |-- dataset.py             Data loading and preprocessing
|   |-- utils.py               Shared utilities and constants
|
|-- data/
|   |-- raw/                   Raw ANSIS exports and GEE extractions
|   |-- processed/             Cleaned and merged datasets
|
|-- models/
|   |-- ensemble/              Trained model checkpoints and scalers
|
|-- notebooks/
|   |-- exploration.ipynb      Data exploration and visualisation
|   |-- baselines.ipynb        Baseline model comparisons
|
|-- tests/
|   |-- test_model.py          Model unit tests
|   |-- test_dataset.py        Data pipeline tests
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
