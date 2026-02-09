# Archive

Original scripts preserved for reference. These are the source scripts that were consolidated and refactored into the `src/` package.

## Directory Index

### resnet_ensemble/
- `resnet_soil_trainer.py` - Original ResNet training script (658 lines)
- `ensemble_inference.py` - Original ensemble inference (411 lines)
- `run_speirs_soil_model.py` - End-to-end farm pipeline (563 lines)
- `resnet_colab.ipynb` - Google Colab training notebook
- `sodic_classifier_colab.ipynb` - Sodic soil classifier notebook

### classical_ml/
- `svr_alphaearth_top10cm.py` - SVR with RBF kernel
- `alphaearth_models_top10cm.py` - Multi-model comparison
- `ph_model_search.py` - pH model search with RFE
- `cubist_model_search_with400.py` - Cubist model search
- `gpr_pipeline.py` - Gaussian Process Regression

### calibration/
- `national_local_calibration_calib_catboost.py` - CatBoost calibration

### farm_applications/
- `lime_rate_ph55.py` - Lime rate calculation
- `gypsum_rate_esp6.py` - Gypsum rate calculation
- `clhs_sampling_points.py` - cLHS sampling design
- `cv_paddock_holdout.py` - Paddock holdout cross-validation

### data_pipeline/
- `main.py` - Original extraction orchestrator
- `soil_data_fetcher.py` - TERN API client
- `ansis_parser.py` - ANSIS JSON-LD parser
- `data_cleaner.py` - Data cleaning/validation
- `db_handler.py` - SQLite handler
- `config.py` - Configuration constants
