"""Locate canonical sources in an editable checkout or bundled wheel resources."""

from pathlib import Path

PACKAGE = Path(__file__).parent
CHECKOUT = PACKAGE.parents[1]


def backend():
    return (
        PACKAGE / "_backend"
        if (PACKAGE / "_backend").is_dir()
        else CHECKOUT / "example"
    )


def catalog():
    if (PACKAGE / "_catalog").is_dir():
        return PACKAGE / "_catalog"
    return CHECKOUT / "example"


def catalog_lock():
    root = catalog()
    return (
        root / "contracts/fleet.lock"
        if (root / "contracts").is_dir()
        else root / "transformers/tests/contracts/fleet.lock"
    )


def catalog_expectations():
    return catalog_lock().parent / "expectations"
