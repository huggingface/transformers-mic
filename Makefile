.PHONY: build build-release test clean

build:
	pip install -e ".[runtime,dev]"

build-release:
	rm -rf build/ dist/ src/*.egg-info
	python -m build

test:
	python -m pytest tests/ -q

clean:
	rm -rf build/ dist/ src/*.egg-info .pytest_cache
