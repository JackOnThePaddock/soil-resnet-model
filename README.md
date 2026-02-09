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

| Target | N Test | R-squared | RMSE (raw) | MAE (raw) |
|--------|--------|-----------|------------|-----------|
| pH     | 698    | 0.809     | 0.446      | 0.343     |
| CEC    | 130    | 0.959     | 3.434      | 2.607     |
| ESP    | 100    | 0.841     | 1.609      | 0.901     |
| SOC    | 150    | 0.870     | 1.524      | 0.739     |
| Ca     | 338    | 0.929     | 2.603      | 1.900     |
| Mg     | 327    | 0.926     | 1.187      | 0.814     |
| Na     | 234    | 0.805     | 0.239      | 0.115     |

All metrics are computed on a stratified holdout set that was excluded from training and cross-validation. The varying test set sizes reflect per-target data availability, as the model uses masked loss to handle missing values.

---

## Quick Start

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/jackcoombs/soil-resnet-model.git
cd soil-resnet-model
pip install -e .
```

For all optional dependencies (baselines, GEE, notebooks, dev tools):

```bash
pip install -e ".[all]"
```

---

## Usage

### Training

Train a full ensemble (5 models, 5-fold CV each):

```bash
python scripts/train_resnet_ensemble.py \
    --data data/processed/features.csv \
    --config configs/resnet.yaml \
    --output models/resnet_ensemble
```

Or from Python:

```python
from src.training.train_resnet import train_ensemble

train_ensemble(
    data_path="data/processed/features.csv",
    output_dir="models/resnet_ensemble",
    n_models=5, n_splits=5, epochs=1000, lr=1e-4,
)
```

### Evaluation

Evaluate and compare models:

```bash
python scripts/evaluate_models.py --metrics-dir results/metrics/
```

### Prediction

Generate predictions with uncertainty for new satellite embeddings:

```python
from src.models.ensemble import SoilEnsemble

ensemble = SoilEnsemble("models/resnet_ensemble")
predictions, uncertainty = ensemble.predict(features, return_std=True)
# predictions shape: (N, 7) -- one column per target
# uncertainty shape: (N, 7) -- ensemble standard deviation
```

### National-to-Local Calibration

Fine-tune predictions for a specific farm using local calibration samples:

```python
from src.training.calibration import calibrate_national_to_local

calibrate_national_to_local(
    national_data="data/processed/features.csv",
    local_data="data/farm/local_samples.csv",
    output_dir="models/calibrated",
)
```

### Farm Pipeline

Run end-to-end predictions for a farm:

```bash
python scripts/predict_farm.py \
    --shp data/paddocks.shp \
    --models models/resnet_ensemble \
    --output results/farm_predictions
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
|-- README.md                         Project overview and quick start
|-- METHODOLOGY.md                    Detailed methodology documentation
|-- RESULTS.md                        Model performance metrics
|-- pyproject.toml                    Package configuration and dependencies
|-- Makefile                          Common commands
|-- configs/
|   |-- resnet.yaml                   ResNet hyperparameters
|   |-- baselines.yaml                Baseline model configs
|
|-- src/
|   |-- models/
|   |   |-- resnet.py                 NationalSoilNet + ResidualBlock
|   |   |-- dataset.py                SoilDataset (PyTorch Dataset)
|   |   |-- ensemble.py               SoilEnsemble loader + predict
|   |   |-- baselines/                SVR, RF, Cubist, GPR
|   |-- training/
|   |   |-- train_resnet.py           Ensemble training loop
|   |   |-- train_baselines.py        Baseline training
|   |   |-- losses.py                 Masked MSE loss
|   |   |-- calibration.py            National-to-local CatBoost
|   |-- inference/
|   |   |-- predict_points.py         CSV point predictions
|   |   |-- predict_raster.py         GeoTIFF raster predictions
|   |   |-- mosaic.py                 Farm mosaic creation
|   |-- evaluation/
|   |   |-- metrics.py                R², RMSE, MAE
|   |   |-- cross_validation.py       K-fold, paddock-holdout CV
|   |   |-- model_comparison.py       Side-by-side comparison
|   |-- features/
|   |   |-- gee_sampler.py            GEE AlphaEarth sampling
|   |   |-- ndvi_mask.py              Vegetation masking
|   |   |-- feature_selection.py      RFE + RF importance
|   |-- data/
|   |   |-- tern_client.py            TERN API client
|   |   |-- ansis_parser.py           ANSIS JSON-LD parser
|   |   |-- data_cleaner.py           Cleaning + validation
|   |   |-- db_handler.py             SQLite handler
|   |   |-- extract_pipeline.py       Data extraction orchestrator
|   |-- applications/
|       |-- farm_pipeline.py          End-to-end farm predictions
|       |-- lime_rates.py             Lime application rates
|       |-- gypsum_rates.py           Gypsum application rates
|       |-- clhs_sampling.py          cLHS sampling design
|       |-- pedotransfer.py           Saxton & Rawls functions
|
|-- scripts/                          CLI entry points
|-- data/raw/                         Raw soil data + SQLite DB
|-- models/resnet_ensemble/           Trained weights + scaler
|-- results/metrics/                  Holdout evaluation CSVs
|-- notebooks/                        EDA and analysis notebooks
|-- tests/                            Unit tests
|-- gee/                              GEE setup and sampling
|-- archive/                          Original scripts for reference
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
