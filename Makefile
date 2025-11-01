.PHONY: setup format lint type test coverage pre-commit-all

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black .

lint:
	ruff check .

type:
	mypy .

test:
	pytest

coverage:
	pytest --cov=. --cov-report=term-missing

pre-commit-all:
	pre-commit run --all-files
