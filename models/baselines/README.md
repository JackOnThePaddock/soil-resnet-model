# Baseline Models

Trained baseline models are stored here after running:

```bash
python scripts/train_baselines.py --data data/processed/features.csv --output models/baselines/
```

Baseline models (SVR, Random Forest, Extra Trees) are trained via `src/training/train_baselines.py`.
Trained baseline model files are not tracked in git. Only the ResNet ensemble weights are included.
