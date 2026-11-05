"""Shared fleet snapshots and the framework's single revision pin.

The dataset stores each fleet as readable CSV tables under ``<fleet>/``; tmic
converts them to and from the JSON lock the runners read. Revisions from before
the tables only have the JSON files (``FILES``), which are still read as they are.

Sentence Transformers contracts that have no Transformers usage live in their own
dataset (``ST_REPOSITORY``, fleet ``ST_FLEET``) with the same tables, so they never
count as Transformers coverage.
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
ST_REPOSITORY = "huggingface-mic/sentence-transformers-fleet"
ST_FLEET = "sentence_transformers"
ST_FRAMEWORK = "sentence_transformers"
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

INTEGRATION_COLUMNS = ("contract", "revision", "integrations")


def table_path(name, table):
    return f"{name}/{table}.csv"


def _csv(table, rows):
    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        INTEGRATION_COLUMNS if table == "integrations" else COLUMNS[table],
        lineterminator="\n",
    )
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


def to_tables(lock, framework=FRAMEWORK):
    """The lock as {table: CSV text}; refuses anything the tables cannot hold exactly."""
    if lock.get("framework", framework) != framework:
        raise ValueError(f"This fleet dataset only holds {framework} locks")
    contracts = []
    for name, contract in lock["contracts"].items():
        extra = set(contract) - {
            "integration",
            "fixtures",
            "checkpoints",
            "integrations",
        }
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
    tables = {
        "contracts": _csv("contracts", contracts),
        "environment": _csv("environment", environment),
        "known_failures": _csv("known_failures", known),
    }

    advertised = [
        {
            "contract": name,
            "revision": contract["integration"]["revision"],
            "integrations": "; ".join(contract["integrations"]),
        }
        for name, contract in lock["contracts"].items()
        if "integrations" in contract
    ]
    if advertised:
        tables["integrations"] = _csv("integrations", advertised)
    return tables


def from_tables(tables, framework=FRAMEWORK):
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
    if "integrations" in tables:
        reader = csv.DictReader(io.StringIO(tables["integrations"]))
        if tuple(reader.fieldnames or ()) != INTEGRATION_COLUMNS:
            raise ValueError("Invalid integrations.csv columns")
        for row in reader:
            contract = contracts.get(row["contract"])
            # Older exporters do not update this optional table. Never trust stale metadata.
            if contract and contract["integration"]["revision"] == row["revision"]:
                contract["integrations"] = _split(row["integrations"])
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
        "framework": framework,
        "environment": {
            row["package"]: row["version"]
            for row in _rows("environment", tables["environment"])
        },
        "contracts": contracts,
    }
    if known:
        lock["known_failures"] = known
    return lock


def operations(name, lock, framework=FRAMEWORK):
    """Dataset files for one fleet: its tables."""
    return {
        table_path(name, table): text
        for table, text in to_tables(lock, framework).items()
    }


def resolve(revision="main", repository=REPOSITORY):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    return HfApi().dataset_info(repository, revision=revision).sha


def download_st(revision="main"):
    """The Sentence Transformers fleet's JSON lock at an immutable revision."""
    revision = resolve(revision, ST_REPOSITORY)

    def fetch(table):
        return Path(
            hf_hub_download(
                ST_REPOSITORY,
                table_path(ST_FLEET, table),
                repo_type="dataset",
                revision=revision,
            )
        ).read_text()

    lock = from_tables({table: fetch(table) for table in COLUMNS}, ST_FRAMEWORK)
    path = Path(tempfile.gettempdir()) / "tmic-fleet" / revision / f"{ST_FLEET}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2) + "\n")
    return path, revision


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
    try:
        tables["integrations"] = fetch(table_path(name, "integrations")).read_text()
    except EntryNotFoundError:
        pass
    lock = from_tables(tables)
    try:
        legacy = json.loads(fetch(FILES[name]).read_text())
    except EntryNotFoundError:
        legacy = None
    if legacy is not None:
        # A tmic from before the tables writes only the JSON; never shadow that with stale tables.
        legacy.pop("comment", None)
        comparable = {
            **lock,
            "contracts": {
                name: {
                    key: value
                    for key, value in contract.items()
                    if key != "integrations"
                }
                for name, contract in lock["contracts"].items()
            },
        }
        if legacy != comparable:
            raise ValueError(
                f"{REPOSITORY}@{revision}: the {name} tables and {FILES[name]} disagree; "
                "an export by an older tmic updated only the JSON. Re-export with this tmic."
            )
    path = Path(tempfile.gettempdir()) / "tmic-fleet" / revision / FILES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    # One temporary file per call: concurrent downloads (a batch of `tmic create`)
    # shared one .partial, lost the rename race, and fell back to the catalog's
    # stale environment, which export then refused after the whole cloud run.
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".partial", delete=False
    ) as partial:
        partial.write(json.dumps(lock, indent=2) + "\n")
    Path(partial.name).replace(path)
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
