import importlib.util

import pytest

from transformers_mic import workspace


@pytest.fixture(autouse=True)
def catalog_environment(monkeypatch):
    # Keep workspace creation offline; test_fleet_environment covers the Hub lookup.
    monkeypatch.setattr(
        workspace, "fleet_environment", lambda catalog: catalog["environment"]
    )


def _missing_runtime():
    return [n for n in ("torch", "transformers") if importlib.util.find_spec(n) is None]


# Tests that load or introspect real models need the `runtime` extra.
requires_runtime = pytest.mark.skipif(
    bool(_missing_runtime()),
    reason=f"needs the runtime extra (missing: {', '.join(_missing_runtime())})",
)
