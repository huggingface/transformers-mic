"""Shared fleet snapshots and the framework's single revision pin.

The dataset stores each fleet as readable CSV tables under ``<fleet>/``; tmic
converts them to and from the JSON lock the runners read. Revisions from before
the tables only have the JSON files, which are still read as they are.
"""

import csv
import io
import json
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REPOSITORY = "huggingface-mic/transformers-fleet"
FILES = {"control": "fleet-lock.json", "prototypes": "prototype-fleet.lock"}
FRAMEWORK = "transformers"
COLUMNS = {
    "contracts": (
        "contract",
        "target",
        "mic_repo",
        "revision",
        "contract_sha256",
        "fixture_sha256",
        "checkpoints",
    ),
    "environment": ("package", "version"),
    "known_failures": ("contract", "target", "issue", "match", "skip_outputs"),
}
# Transition: the JSON locks are still written next to the tables until the
# Transformers CI runner (transformers-ci, pinned by ci_sha) reads the tables.
WRITE_JSON = True


def table_path(name, table):
    return f"{name}/{table}.csv"


def _csv(table, rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, COLUMNS[table], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _rows(table, text):
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != COLUMNS[table]:
        raise ValueError(
            f"{table}.csv must have the columns {', '.join(COLUMNS[table])}"
        )
    return list(reader)


def _split(cell):
    return [part.strip() for part in cell.split(";") if part.strip()]


def to_tables(lock):
    """The lock as {table: CSV text}; refuses anything the tables cannot hold exactly."""
    if lock.get("framework", FRAMEWORK) != FRAMEWORK:
        raise ValueError(f"The fleet dataset only holds {FRAMEWORK} locks")
    contracts = []
    for name, contract in lock["contracts"].items():
        extra = set(contract) - {"integration", "fixtures", "checkpoints"}
        if extra:
            raise ValueError(
                f"{name}: fleet tables cannot hold {', '.join(sorted(extra))}"
            )
        integration = contract["integration"]
        fixtures = contract.get("fixtures") or {}
        if set(integration) != {"repo", "revision", "files_sha256"}:
            raise ValueError(
                f"{name}: a fleet contract needs a published repo, revision and files_sha256"
            )
        owner = fixtures.get(integration["repo"])
        if not owner or owner["revision"] != integration["revision"]:
            raise ValueError(
                f"{name}: the contract must be pinned in one of its fixture snapshots"
            )
        for repo, fixture in fixtures.items():
            if set(fixture) - {"target"} != {"revision", "files_sha256"}:
                raise ValueError(
                    f"{name}: fixture {repo} needs exactly a revision and files_sha256 (and optional target)"
                )
            if ";" in fixture.get("target", ""):
                raise ValueError(f"{name}: fixture target names cannot contain ';'")
            carries = repo == integration["repo"]
            contracts.append(
                {
                    "contract": name,
                    "target": fixture.get("target", ""),
                    "mic_repo": repo,
                    "revision": fixture["revision"],
                    "contract_sha256": integration["files_sha256"] if carries else "",
                    "fixture_sha256": fixture["files_sha256"],
                    "checkpoints": "; ".join(
                        f"{model}@{revision}"
                        for model, revision in contract.get("checkpoints", {}).items()
                    )
                    if carries
                    else "",
                }
            )
    known = []
    for key, entry in lock.get("known_failures", {}).items():
        contract, _, target = key.partition("/")
        entry = {"issue": entry} if isinstance(entry, str) else entry
        if not target or set(entry) - {"issue", "match", "skip_outputs"}:
            raise ValueError(
                f"known_failures[{key}] cannot be stored in the fleet tables"
            )
        known.append(
            {
                "contract": contract,
                "target": target,
                "issue": entry["issue"],
                "match": entry.get("match", ""),
                "skip_outputs": "; ".join(entry.get("skip_outputs", [])),
            }
        )
    environment = [
        {"package": package, "version": version}
        for package, version in lock["environment"].items()
    ]
    return {
        "contracts": _csv("contracts", contracts),
        "environment": _csv("environment", environment),
        "known_failures": _csv("known_failures", known),
    }


def from_tables(tables):
    """The JSON lock for {table: CSV text}, as the runners read it."""
    contracts = {}
    for row in _rows("contracts", tables["contracts"]):
        contract = contracts.setdefault(row["contract"], {"fixtures": {}})
        fixture = {"revision": row["revision"], "files_sha256": row["fixture_sha256"]}
        if row["target"]:
            fixture["target"] = row["target"]
        contract["fixtures"][row["mic_repo"]] = fixture
        if row["contract_sha256"]:
            if "integration" in contract:
                raise ValueError(
                    f"{row['contract']}: more than one row carries the contract"
                )
            contract["integration"] = {
                "repo": row["mic_repo"],
                "revision": row["revision"],
                "files_sha256": row["contract_sha256"],
            }
            contract["checkpoints"] = dict(
                pin.rsplit("@", 1) for pin in _split(row["checkpoints"])
            )
        elif row["checkpoints"]:
            raise ValueError(
                f"{row['contract']}: checkpoints belong on the row carrying the contract"
            )
    for name, contract in contracts.items():
        if "integration" not in contract:
            raise ValueError(f"{name}: no row carries the contract (contract_sha256)")
        contracts[name] = {
            "integration": contract["integration"],
            "fixtures": contract["fixtures"],
            "checkpoints": contract["checkpoints"],
        }
    known = {}
    for row in _rows("known_failures", tables["known_failures"]):
        entry = {"issue": row["issue"]}
        if row["match"]:
            entry["match"] = row["match"]
        if row["skip_outputs"]:
            entry["skip_outputs"] = _split(row["skip_outputs"])
        known[f"{row['contract']}/{row['target']}"] = (
            entry["issue"] if len(entry) == 1 else entry
        )
    lock = {
        "framework": FRAMEWORK,
        "environment": {
            row["package"]: row["version"]
            for row in _rows("environment", tables["environment"])
        },
        "contracts": contracts,
    }
    if known:
        lock["known_failures"] = known
    return lock


def operations(name, lock):
    """Dataset files for one fleet: its tables, plus the JSON lock during the transition."""
    files = {table_path(name, table): text for table, text in to_tables(lock).items()}
    if WRITE_JSON:
        files[FILES[name]] = json.dumps(from_tables(to_tables(lock)), indent=2) + "\n"
    return files


def resolve(revision="main"):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    return HfApi().dataset_info(REPOSITORY, revision=revision).sha


def read_pin(path):
    revision = Path(path).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{path} must contain one immutable dataset commit SHA")
    return revision


def download(name="control", revision="main"):
    """A local JSON lock for the fleet at an immutable revision."""
    revision = resolve(revision)

    def fetch(filename):
        return Path(
            hf_hub_download(
                REPOSITORY, filename, repo_type="dataset", revision=revision
            )
        )

    try:
        tables = {
            table: fetch(table_path(name, table)).read_text() for table in COLUMNS
        }
    except EntryNotFoundError:  # a revision from before the tables
        return fetch(FILES[name]), revision
    lock = from_tables(tables)
    if WRITE_JSON:
        # The runners still read the JSON; an older tmic would have updated only that.
        legacy = json.loads(fetch(FILES[name]).read_text())
        legacy.pop("comment", None)
        if legacy != lock:
            raise ValueError(
                f"{REPOSITORY}@{revision}: the {name} tables and {FILES[name]} disagree; "
                "an export by an older tmic updated only the JSON. Re-export with this tmic."
            )
    path = Path(tempfile.gettempdir()) / "tmic-fleet" / revision / FILES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(lock, indent=2) + "\n")
    partial.replace(path)
    return path, revision


def pin(checkout, revision):
    path = Path(checkout).resolve() / "tests/contracts.lock"
    # Resolve named refs before writing; a framework pin must always be immutable.
    revision = resolve(revision)
    for name in FILES:
        download(name, revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(revision + "\n")
    return path
