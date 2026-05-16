.PHONY: lint test

lint:
	ruff check src tests --line-length 120
	flake8 src tests --max-line-length 120

test:
	PYTHONPATH=. pytest -q
