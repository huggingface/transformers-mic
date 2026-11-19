.PHONY: build build-release test format lint clean

build:
	pip install -e ".[runtime,dev]"

build-release:
	rm -rf build/ dist/ src/*.egg-info
	python -m build

test:
	python -m pytest tests/ -q

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	ruff format --check .

clean:
	rm -rf build/ dist/ src/*.egg-info .pytest_cache
