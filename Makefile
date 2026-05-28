# FreeAlphaRadar — developer convenience targets.
.PHONY: help install run scorer seed test lint format precommit docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install all pinned dependencies
	pip install -r requirements.txt

run: ## Launch the Streamlit dashboard
	streamlit run streamlit_app.py

scorer: ## Run the batch scorer (refreshes cache + scores)
	python run_scorer.py

seed: ## (Re)seed the offline sample dataset into SQLite
	python run_scorer.py --seed-sample --no-refresh

test: ## Run the offline test-suite (no network)
	FAR_OFFLINE=1 pytest

lint: ## Check formatting with black & isort (no changes)
	black --check --line-length 88 freealpharadar tests *.py
	isort --check-only --profile black freealpharadar tests *.py

format: ## Auto-format with black & isort
	isort --profile black freealpharadar tests *.py
	black --line-length 88 freealpharadar tests *.py

precommit: ## Run all pre-commit hooks against all files
	pre-commit run --all-files

docker-build: ## Build the Docker image
	docker build -t freealpharadar:latest .

docker-up: ## Run via docker-compose
	docker compose up --build

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache **/__pycache__ build dist *.egg-info
	find . -name '*.pyc' -delete
