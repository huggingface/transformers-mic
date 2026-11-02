import importlib.util

import pytest

from transformers_mic import workspace


@pytest.fixture(autouse=True)
def catalog_environment(monkeypatch):
    # Keep workspace creation offline; test_fleet_environment covers the Hub lookup.
    monkeypatch.setattr(
        workspace, "fleet_environment", lambda catalog: catalog["environment"]
    )



# Tests that load or introspect real models need the `runtime` extra
# (pip install -e ".[dev,runtime]"); without it, @pytest.mark.requires_runtime
# tests are skipped with the missing packages named.
MISSING_RUNTIME = [
    name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_runtime: needs torch and transformers (the runtime extra)"
    )


def pytest_collection_modifyitems(config, items):
    if not MISSING_RUNTIME:
        return
    skip = pytest.mark.skip(
        reason=f"needs the runtime extra (missing: {', '.join(MISSING_RUNTIME)})"
    )
    for item in items:
        if "requires_runtime" in item.keywords:
            item.add_marker(skip)
