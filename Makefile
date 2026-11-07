.PHONY: install check lint test report

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest --cov --cov-report=term-missing

check: lint test

report:
	crisk train --out artifacts --report docs/model-validation-report.md --check
