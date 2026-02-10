# NationalSoilNet: Multi-Target Soil Property Prediction from Satellite Embeddings

NationalSoilNet is a ResNet-based ensemble model that predicts seven key soil properties from 64-band AlphaEarth satellite embeddings. Trained on 3,534 Australian soil samples (0--15 cm depth, 2017--2025) sourced from the ANSIS/TERN portal, the model predicts pH, CEC, ESP, SOC, Ca, Mg, and Na.

The system uses an ensemble of five independently seeded models with five-fold cross-validation to produce both point predictions and uncertainty estimates (ensemble standard deviation). A downstream national-to-local calibration step using CatBoost is recommended for farm-level applications.

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

Each output head: Linear(128 -> 64) -> SiLU -> Linear(64 -> 1)
```

Each `ResidualBlock` consists of two linear layers with batch normalisation, SiLU activation, and dropout, wrapped with a skip (identity) connection. The multi-target design allows the network to learn shared representations across correlated soil properties while maintaining independent prediction heads.

---

## Results

### Training Data Performance

In-sample metrics (3,534 samples -- the final models were trained on all data):

| Target | N | R2 | RMSE | MAE | Unit |
|--------|---|-----|------|-----|------|
| pH | 3,511 | 0.815 | 0.446 | 0.337 | pH |
| CEC | 694 | 0.952 | 3.321 | 2.425 | cmol/kg |
| ESP | 563 | 0.785 | 1.367 | 0.827 | % |
| SOC | 755 | 0.904 | 1.193 | 0.727 | % |
| Ca | 1,823 | 0.918 | 2.435 | 1.752 | cmol/kg |
| Mg | 1,775 | 0.924 | 1.243 | 0.866 | cmol/kg |
| Na | 1,309 | 0.894 | 0.190 | 0.102 | cmol/kg |

### Independent Validation

True generalisation performance on data **never seen during training** (zero overlap confirmed):

| Target | National (1,368) R2 | Speirs Farm (60) R2 |
|--------|---------------------|---------------------|
| pH | 0.606 | -0.914 |
| CEC | 0.817 | 0.214 |
| ESP | 0.363 | 0.464 |
| Na | 0.559 | 0.536 |

CEC generalises best (R2 = 0.817). pH and ESP show significant performance drops on unseen data, and the model fails on the Speirs farm for pH (negative R2), highlighting the need for national-to-local calibration for farm-level applications. See [RESULTS.md](RESULTS.md) for full analysis including per-paddock breakdowns and ESP derivation experiments.

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

This training path now supports:
- `group_kfold` CV by location/site
- weighted Huber loss for sparse targets
- per-target transforms (e.g. `log1p` for skewed chemistry)
- ESP consistency regularization (`ESP ~= 100 * Na / CEC`)
- optional specialist models for sparse targets (`CEC/ESP/SOC`)

Or from Python (for custom workflows):

```python
from pathlib import Path

from src.training.train_resnet import train_ensemble
from src.training.train_resnet import load_training_data

targets = ["ph", "cec", "esp", "soc", "ca", "mg", "na"]
X, y, valid_targets = load_training_data(
    csv_path="data/processed/features.csv",
    target_cols=targets,
    feature_prefix="band_",
    n_features=64,
)

config = {
    "input_dim": 64,
    "hidden_dim": 128,
    "num_res_blocks": 2,
    "dropout": 0.2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 300,
    "patience": 30,
    "grad_clip": 1.0,
    "ensemble_size": 5,
    "n_folds": 5,
    "random_seed": 42,
}

train_ensemble(
    X=X,
    y=y,
    target_names=valid_targets,
    config=config,
    output_dir=Path("models/resnet_ensemble"),
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

### Bare Earth + SpectralGPT Fusion

Download GA Barest Earth (Sentinel-2) via DEA WCS:

```bash
python scripts/download_bare_earth.py \
    --bbox 148.9 -35.5 149.3 -35.1 \
    --output data/external/ga_barest_earth_canberra.tif \
    --discover
```

Build fused features (AlphaEarth + BareEarth bands + SpectralGPT embeddings + optional management):

```bash
python scripts/build_fused_features.py \
    --soil-csv data/processed/features.csv \
    --bare-earth-raster data/external/ga_barest_earth_canberra.tif \
    --output-csv data/processed/features_fused.csv \
    --spectral-method spectral_gpt \
    --spectral-dim 16 \
    --applications-csv data/farm/management_applications.csv
```

Use the official pretrained SpectralGPT checkpoint (`SpectralGPT+.pth`) for point chips:

```bash
python scripts/pull_bare_earth_embeddings.py \
    --normalized-csv data/processed/features_normalized.csv \
    --points-csv data/processed/features.csv \
    --output-csv data/processed/features_normalized_bareearth_sgpt.csv \
    --output-embeddings-csv data/processed/features_normalized_sgpt_embeddings.csv \
    --spectral-backend official_pretrained \
    --spectral-dim 16 \
    --official-request-chunk-size 64 \
    --output-official-raw-csv data/processed/features_normalized_sgpt_official_raw.csv
```

Notes:
- The script will clone the official SpectralGPT repo and download `SpectralGPT+.pth` automatically if not present.
- DEA bare-earth has 10 Sentinel-2 bands; official SpectralGPT expects 12. Missing B1 and B9 are proxied from nearest available bands during preprocessing.

Train fused model:

```bash
python scripts/train_resnet_ensemble.py \
    --data data/processed/features_fused.csv \
    --config configs/resnet_fused.yaml \
    --output models/resnet_fused
```

Quick ablation (does Bare Earth help?) using grouped RF CV:

```bash
python scripts/ablate_bare_earth_gain.py \
    --alpha-csv data/processed/features.csv \
    --fused-csv data/processed/features_fused.csv \
    --alpha-prefix band_ \
    --fused-prefix feat_ \
    --group-mode latlon \
    --output-csv results/metrics/bare_earth_ablation.csv
```

Management application CSV should include at least:
- `date`
- `rate_t_ha`
- one of `type` or `material` (e.g. lime, gypsum)
- plus either `site_id` or `lat`/`lon` for spatial matching

---

## Data Sources

- **Soil samples**: 3,534 samples (2,823 unique locations) at 0--15 cm depth from the Australian National Soil Information System (ANSIS) via the TERN Soil Data API. Samples span 2017--2025 and cover diverse Australian soil types and land uses.
- **Satellite features**: 64-band AlphaEarth embedding composites extracted from Google Earth Engine. Five-year median composites are used with NDVI-based vegetation masking to reduce crop/pasture signal interference.
- **Targets**: pH (CaCl2), Cation Exchange Capacity (CEC, cmol(+)/kg), Exchangeable Sodium Percentage (ESP, %), Soil Organic Carbon (SOC, %), Exchangeable Calcium (Ca, cmol(+)/kg), Exchangeable Magnesium (Mg, cmol(+)/kg), Exchangeable Sodium (Na, cmol(+)/kg).

---

## Project Structure

```
soil-resnet-model/
|-- README.md                         Project overview and quick start
|-- METHODOLOGY.md                    Detailed methodology documentation
|-- RESULTS.md                        Model performance metrics and findings
|-- pyproject.toml                    Package configuration and dependencies
|-- Makefile                          Common commands
|-- configs/
|   |-- resnet.yaml                   ResNet hyperparameters
|   |-- baselines.yaml                Baseline model configs
|
|-- src/                              Refactored source code
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
|   |   |-- metrics.py                R2, RMSE, MAE
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
|
|-- scripts/                          CLI entry points
|-- data/
|   |-- raw/                          Raw soil data + SQLite DB
|   |-- processed/                    Training datasets (features.csv, features_normalized.csv)
|   |-- farm/                         Speirs farm training data (69 samples, 1x1 and 3x3)
|   |-- validation/                   Independent validation data (1,368 national samples)
|-- models/resnet_ensemble/           Trained weights (5 models) + scaler
|-- results/metrics/                  40+ evaluation CSVs (all models, all experiments)
|-- notebooks/                        Colab training notebook (with outputs), sodic classifier
|-- docs/                             Honours report, project report
|-- gee/                              GEE setup, sampling scripts, Colab training code
|-- tests/                            Unit tests
|-- archive/                          ~116 original scripts organized by category
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
