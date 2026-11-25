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
import os
import re
import tempfile
import urllib.request
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
# Expected values re-recorded for one open Transformers PR (`record-mic`): a file on the
# dataset's main that moves named contracts to new mic-repo commits for that PR's runs only.
OVERLAYS = "overlays"


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
        raise ValueError(f"{table}.csv must have the columns {', '.join(COLUMNS[table])}")
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
            raise ValueError(f"{name}: fleet tables cannot hold {', '.join(sorted(extra))}")
        integration = contract["integration"]
        fixtures = contract.get("fixtures") or {}
        if set(integration) != {"repo", "revision", "files_sha256"}:
            raise ValueError(f"{name}: a fleet contract needs a published repo, revision and files_sha256")
        owner = fixtures.get(integration["repo"])
        if not owner or owner["revision"] != integration["revision"]:
            raise ValueError(f"{name}: the contract must be pinned in one of its fixture snapshots")
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
                        f"{model}@{revision}" for model, revision in contract.get("checkpoints", {}).items()
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
            raise ValueError(f"known_failures[{key}] cannot be stored in the fleet tables")
        known.append(
            {
                "contract": contract,
                "target": target,
                "issue": entry["issue"],
                "match": entry.get("match", ""),
                "skip_outputs": "; ".join(entry.get("skip_outputs", [])),
            }
        )
    environment = [{"package": package, "version": version} for package, version in lock["environment"].items()]
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
                raise ValueError(f"{row['contract']}: more than one row carries the contract")
            contract["integration"] = {
                "repo": row["mic_repo"],
                "revision": row["revision"],
                "files_sha256": row["contract_sha256"],
            }
            contract["checkpoints"] = dict(pin.rsplit("@", 1) for pin in _split(row["checkpoints"]))
        elif row["checkpoints"]:
            raise ValueError(f"{row['contract']}: checkpoints belong on the row carrying the contract")
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
        known[f"{row['contract']}/{row['target']}"] = entry["issue"] if len(entry) == 1 else entry
    lock = {
        "framework": framework,
        "environment": {row["package"]: row["version"] for row in _rows("environment", tables["environment"])},
        "contracts": contracts,
    }
    if known:
        lock["known_failures"] = known
    return lock


def operations(name, lock, framework=FRAMEWORK):
    """Dataset files for one fleet: its tables."""
    return {table_path(name, table): text for table, text in to_tables(lock, framework).items()}


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


def overlay_path(pr):
    return f"{OVERLAYS}/transformers-pr-{int(pr)}.json"


def download_overlay(pr, revision):
    """The PR's overlay at a dataset revision, or None."""
    try:
        path = hf_hub_download(REPOSITORY, overlay_path(pr), repo_type="dataset", revision=revision)
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text())


def pins(contract, repo):
    """The pins of `contract` on mic repo `repo`: its integration and fixtures there."""
    found = [contract["integration"]] if contract["integration"]["repo"] == repo else []
    return found + [pin for name, pin in contract.get("fixtures", {}).items() if name == repo]


def apply_overlay(lock, overlay, name):
    """Move the overlay's contracts of fleet `name` to their re-recorded commits.

    An entry applies only while the fleet still pins its `from` commit: once main moved the
    contract (another update, a merged record), the PR's values are stale and the case is
    red again until re-recorded. Returns the lock and {"applied": [...], "stale": [...]}.
    """
    lock = json.loads(json.dumps(lock))
    report = {"applied": [], "stale": []}
    for key, moves in sorted((overlay or {}).get("contracts", {}).items()):
        fleet_name, _, contract_name = key.partition("/")
        if fleet_name != name:
            continue
        contract = lock["contracts"].get(contract_name)
        for repo, move in sorted(moves.items()):
            found = pins(contract, repo) if contract else []
            if found and all(pin["revision"] == move["from"] for pin in found):
                for pin in found:
                    pin["revision"] = move["to"]
                report["applied"].append(f"{contract_name}:{repo}")
            else:
                report["stale"].append(f"{contract_name}:{repo}")
    return lock, report


# Which Transformers commit each pin move applies from. A run of Transformers code X meets, for each
# contract, the newest revision whose row's commit X contains: the history is the source of truth and
# the pin is its newest row. Every pin move writes a row (record-mic's merge, `tmic export --since`,
# reserialize), with `since_transformers` = the commit on Transformers main, or ALWAYS for a move
# that applies to any code (weights re-serialized, a tolerance, a new target). Kept apart from
# COLUMNS: revisions from before it have no such table, and it is optional.
HISTORY = "history"
HISTORY_COLUMNS = ("contract", "mic_repo", "from_revision", "revision", "since_transformers")
HISTORY_KEEP = 5  # rows kept per (contract, mic repo): far more than a run of old code needs
ALWAYS = "always"
TRANSFORMERS = "huggingface/transformers"


def history_csv(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, HISTORY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def download_history(name, revision):
    """The fleet's history rows at a revision, oldest first ([] when it has none)."""
    try:
        text = Path(
            hf_hub_download(REPOSITORY, table_path(name, HISTORY), repo_type="dataset", revision=revision)
        ).read_text()
    except EntryNotFoundError:
        return []
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != HISTORY_COLUMNS:
        raise ValueError(f"{HISTORY}.csv must have the columns {', '.join(HISTORY_COLUMNS)}")
    return list(reader)


def with_history(rows, added):
    """`rows` plus `added`, keeping the newest HISTORY_KEEP per (contract, mic repo)."""
    rows = [*rows, *added]
    kept, seen = [], {}
    for row in reversed(rows):
        key = (row["contract"], row["mic_repo"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= HISTORY_KEEP:
            kept.append(row)
    return list(reversed(kept))


def check_since(since):
    """`since` for a pin move: a full Transformers commit SHA, or ALWAYS."""
    if since != ALWAYS and not re.fullmatch(r"[0-9a-f]{40}", since or ""):
        raise ValueError(f"--since takes a full Transformers commit SHA or {ALWAYS!r}, got {since!r}")
    return since


def moves(old, new):
    """[(contract, mic repo, from, to)] for every pin of a contract in both locks whose revision moved."""
    found = []
    for name, contract in sorted(new["contracts"].items()):
        before = old["contracts"].get(name)
        if not before:
            continue  # a new contract has no earlier value to keep for older code
        repos = {contract["integration"]["repo"], *contract.get("fixtures", {})}
        for repo in sorted(repos):
            frm = {p["revision"] for p in pins(before, repo)}
            to = {p["revision"] for p in pins(contract, repo)}
            if len(frm) == 1 and len(to) == 1 and frm != to:
                found.append((name, repo, frm.pop(), to.pop()))
    return found


def history_rows(old, new, since):
    """The history rows for moving fleet lock `old` to `new`, applying from Transformers `since`."""
    return [
        {"contract": c, "mic_repo": r, "from_revision": f, "revision": t, "since_transformers": since}
        for c, r, f, t in moves(old, new)
    ]


def apply_history(lock, rows, is_ancestor):
    """Pin each contract to the newest revision that applies to the code under test.

    A contract's rows form a chain ending at its pin (each row's `from_revision` is the row before's
    `revision`). The code meets the newest row of that chain whose `since_transformers` it contains
    (`is_ancestor(sha)`; ALWAYS applies to any code), or the chain's first `from_revision` when none
    applies. A pin that is not its newest row's revision moved without one: it is left as it is.
    Returns the lock and {"reverted": ["<contract>:<repo>", ...], "unrecorded": [...]}.
    """
    lock = json.loads(json.dumps(lock))
    report = {"reverted": [], "unrecorded": []}
    chains = {}
    for row in rows:
        chains.setdefault((row["contract"], row["mic_repo"]), []).append(row)
    for (name, repo), chain in sorted(chains.items()):
        contract = lock["contracts"].get(name)
        found = pins(contract, repo) if contract else []
        current = {pin["revision"] for pin in found}
        if not found:
            continue
        if current != {chain[-1]["revision"]}:
            report["unrecorded"].append(f"{name}:{repo}")
            continue
        linked = [chain[-1]]
        for row in reversed(chain[:-1]):
            if row["revision"] != linked[0]["from_revision"]:
                break  # a move with no row: older rows describe another chain
            linked.insert(0, row)
        revision = linked[0]["from_revision"]
        for row in reversed(linked):
            if row["since_transformers"] == ALWAYS or is_ancestor(row["since_transformers"]):
                revision = row["revision"]
                break
        if revision != chain[-1]["revision"]:
            for pin in found:
                pin["revision"] = revision
            report["reverted"].append(f"{name}:{repo}")
    return lock, report


def _github(path, repository=TRANSFORMERS):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/{path}", headers={"Accept": "application/vnd.github+json"}
    )
    if token := os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def github_ancestor(head, repository=TRANSFORMERS):
    """is_ancestor(sha) for commit `head` through GitHub's compare API (GH_TOKEN/GITHUB_TOKEN if set).

    An answer the API cannot give (network, rate limit, unknown commit) counts as an ancestor: the
    run then meets the latest values, as it did before history existed.
    """
    cache = {}

    def is_ancestor(sha):
        if sha not in cache:
            try:
                cache[sha] = _github(f"compare/{sha}...{head}", repository).get("status") in ("ahead", "identical")
            except (OSError, ValueError):
                cache[sha] = True
        return cache[sha]

    return is_ancestor


def github_merged(pr, repository=TRANSFORMERS):
    """PR `pr`'s commit on main if it is merged, else None (also when GitHub cannot answer)."""
    try:
        data = _github(f"pulls/{int(pr)}", repository)
    except (OSError, ValueError):
        return None
    return data.get("merge_commit_sha") if data.get("merged") else None


def list_overlays(revision):
    """The PR numbers with an overlay on the fleet dataset at `revision`."""
    files = HfApi().list_repo_files(REPOSITORY, repo_type="dataset", revision=revision)
    found = []
    for path in files:
        match = re.fullmatch(rf"{re.escape(OVERLAYS)}/transformers-pr-(\d+)\.json", path)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def apply_merged_overlays(lock, name, revision, code_is_ancestor, skip=None, merged=github_merged, prs=None):
    """Apply the overlays of PRs merged into the code under test that the reconciler has not merged yet.

    Between a PR's merge and the reconciler's next pass, its values are still an overlay: code that
    contains its commit on main meets them now, code that does not keeps the fleet's.
    Returns the lock and {"merged": [...], "merged_stale": [...]}.
    """
    report = {"merged": [], "merged_stale": []}
    for pr in list_overlays(revision) if prs is None else prs:
        if pr == skip:
            continue
        sha = merged(pr)
        if not sha or not code_is_ancestor(sha):
            continue
        lock, applied = apply_overlay(lock, download_overlay(pr, revision), name)
        report["merged"] += [f"{key}@transformers-pr-{pr}" for key in applied["applied"]]
        report["merged_stale"] += [f"{key}@transformers-pr-{pr}" for key in applied["stale"]]
    return lock, report


def download_for_run(name, revision, pr=None, code_revision=None, is_ancestor=None, merged=github_merged):
    """download() for one run: the values the code under test contains, then the PR's overlay.

    (lock path, revision, report): report has "reverted"/"unrecorded" (history), "merged"/
    "merged_stale" (merged PRs not reconciled yet) and "applied"/"stale" (the PR's own overlay).
    """
    path, revision = download(name, revision)
    lock, report = json.loads(Path(path).read_text()), {}
    tag = []
    if code_revision:
        is_ancestor = is_ancestor or github_ancestor(code_revision)
        lock, report = apply_history(lock, download_history(name, revision), is_ancestor)
        lock, pending = apply_merged_overlays(lock, name, revision, is_ancestor, skip=pr, merged=merged)
        report.update(pending)
        tag.append(code_revision[:12])
    if pr is not None:
        lock, overlay = apply_overlay(lock, download_overlay(pr, revision), name)
        report.update(overlay)
        tag.append(f"transformers-pr-{int(pr)}")
    if not tag:
        return path, revision, report
    out = path.with_name(f"{'.'.join(tag)}.{path.name}")
    out.write_text(json.dumps(lock, indent=2) + "\n")
    return out, revision, report


def download_for_pr(name, revision, pr):
    """download() with PR `pr`'s overlay applied: (lock path, revision, overlay report)."""
    path, revision = download(name, revision)
    lock, report = apply_overlay(json.loads(Path(path).read_text()), download_overlay(pr, revision), name)
    overlaid = path.with_name(f"transformers-pr-{int(pr)}.{path.name}")
    overlaid.write_text(json.dumps(lock, indent=2) + "\n")
    return overlaid, revision, report


def download(name="control", revision="main"):
    """A local JSON lock for the fleet at an immutable revision."""
    revision = resolve(revision)

    def fetch(filename):
        return Path(hf_hub_download(REPOSITORY, filename, repo_type="dataset", revision=revision))

    try:
        tables = {table: fetch(table_path(name, table)).read_text() for table in COLUMNS}
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
                name: {key: value for key, value in contract.items() if key != "integrations"}
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
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".partial", delete=False) as partial:
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
