.PHONY: install train-resnet train-baselines evaluate predict-farm report test lint clean

install:
	pip install -e ".[all]"

train-resnet:
	@echo "Usage: python scripts/train_resnet_ensemble.py --data <data.csv> --config configs/resnet.yaml"
	@echo "Example: python scripts/train_resnet_ensemble.py --data data/processed/features.csv --config configs/resnet.yaml --output models/resnet_ensemble"

train-baselines:
	@echo "Usage: python scripts/train_baselines.py --data <data.csv> --config configs/baselines.yaml"
	@echo "Example: python scripts/train_baselines.py --data data/processed/features.csv --config configs/baselines.yaml --output models/baselines"

evaluate:
	python scripts/evaluate_models.py

predict-farm:
	@echo "Usage: python scripts/predict_farm.py --shp <paddocks.shp> --output <output_dir>"
	@echo "Example: python scripts/predict_farm.py --shp data/paddocks.shp --models models/resnet_ensemble --output results/farm_predictions"

report:
	python scripts/generate_report.py

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
