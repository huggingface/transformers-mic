"""The locks nightly.yml and shadow.yml run: the Hub control fleet, plus the catalog's rest.

    PYTHONPATH=src python .github/scripts/control_locks.py --out out >> "$GITHUB_ENV"

The control fleet (huggingface-mic/transformers-fleet, fleet ``control``) is what
the project ships: umbrella ``*-mic`` repos whose reviewed baselines live on the
Hub. It is read at the dataset revision pinned in
example/transformers/tests/contracts.lock, as a Transformers checkout pins it, so a
fleet update reaches these workflows only through a reviewed pin bump here.

The in-tree catalog lock (example/transformers/tests/contracts/fleet.lock) stays
the source of ``tmic create`` and the publishing scripts. Its contracts that the
control fleet has no entry for (by contract name) keep running from it, against the
in-tree baselines.

Prints, for $GITHUB_ENV:
  CONTROL_LOCK=<out>/control.lock   the control fleet's lock at the pinned revision
  CONTROL_REVISION=<dataset sha>
  CATALOG_ONLY=<names>              catalog contracts the control fleet lacks
"""

import argparse
import json
import shutil
from pathlib import Path

from transformers_mic import fleet

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "example/transformers/tests"
PIN = CONTRACTS / "contracts.lock"
CATALOG = CONTRACTS / "contracts/fleet.lock"


def catalog_only(catalog, control):
    """Catalog contract names the control fleet does not pin."""
    return sorted(set(catalog["contracts"]) - set(control["contracts"]))


def main(argv=None, download=fleet.download):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pin", type=Path, default=PIN)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args(argv)
    revision = fleet.read_pin(args.pin)
    path, resolved = download("control", revision)
    control = json.loads(Path(path).read_text())
    if control.get("framework") != "transformers" or not control.get("contracts"):
        raise ValueError(f"control fleet at {revision} is not a Transformers lock")
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "control.lock"
    shutil.copyfile(path, target)
    rest = catalog_only(json.loads(args.catalog.read_text()), control)
    print(f"CONTROL_LOCK={target.resolve()}")
    print(f"CONTROL_REVISION={resolved}")
    print(f"CATALOG_ONLY={' '.join(rest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
