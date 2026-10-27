import pytest

from transformers_mic import workspace


@pytest.fixture(autouse=True)
def catalog_environment(monkeypatch):
    # Keep workspace creation offline; test_fleet_environment covers the Hub lookup.
    monkeypatch.setattr(
        workspace, "fleet_environment", lambda catalog: catalog["environment"]
    )
