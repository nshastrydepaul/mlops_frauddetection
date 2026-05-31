.PHONY: install dev data train predict test lint format clean docker_build docker_run docs

# Note: 'uv' is a faster alternative to pip. Install with: pip install uv
# Then replace 'pip install' with 'uv pip install' in the commands below.

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -r requirements_dev.txt
	pre-commit install

data:
	python -m mlops_frauddetection.data.make_dataset

train:
	python -m mlops_frauddetection.train_model

predict:
	python -m mlops_frauddetection.predict_model

test:
	pytest tests/ -v 2>&1 | tee reports/figures/pytest_section1.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

docker_build:
	docker build -t mlops_frauddetection -f dockerfiles/Dockerfile .

docker_run:
	docker run --rm \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/reports:/app/reports \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/data:/app/data \
		-e PYTHONPATH=/app/src \
		mlops_frauddetection

docker_compose_build:
	docker compose build --no-cache

docker_compose_up:
	docker compose up

docker_compose_down:
	docker compose down

docs:
	mkdocs serve
