.PHONY: install train-resnet train-baselines evaluate predict-farm test lint clean

install:
	pip install -e ".[all]"

train-resnet:
	python scripts/train_resnet_ensemble.py --config configs/resnet.yaml

train-baselines:
	python scripts/train_baselines.py --config configs/baselines.yaml

evaluate:
	python scripts/evaluate_models.py

predict-farm:
	python scripts/predict_farm.py

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
