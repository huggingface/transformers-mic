"""Current Hub ownership; preserve immutable historical manifests and pins."""

from functools import cache

DEFAULT_NAMESPACE = "huggingface-mic"


def canonical_repo(repo):
    prefix = "hf-internal-testing/mic-"
    return (
        DEFAULT_NAMESPACE + "/mic-" + repo[len(prefix) :]
        if repo.startswith(prefix)
        else repo
    )


def canonical_references(value):
    """Update exported identifiers without changing hashes or revisions."""
    if isinstance(value, dict):
        return {canonical_repo(k): canonical_references(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical_references(v) for v in value]
    return canonical_repo(value) if isinstance(value, str) else value


@cache
def consolidation():
    import json
    from pathlib import Path

    path = Path(__file__).with_name("consolidation.json")
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"aliases": {}, "fixtures": {}, "integrations": {}}
    )


def umbrella_references(value):
    aliases = consolidation()["aliases"]
    if isinstance(value, dict):
        return {aliases.get(k, k): umbrella_references(v) for k, v in value.items()}
    if isinstance(value, list):
        return [umbrella_references(v) for v in value]
    return aliases.get(value, value) if isinstance(value, str) else value


def upgrade_contract(contract):
    import copy

    result = copy.deepcopy(contract)
    migration = consolidation()
    original = result["integration"]["repo"]
    primary = migration["integrations"].get(original)
    if primary is None:
        return rename_known_targets(result, migration)
    result["integration"].update(
        repo=primary["repo"],
        revision=primary["revision"],
        files_sha256=primary["contract_files_sha256"],
    )
    fixtures = {}
    for repo, pin in result.get("fixtures", {}).items():
        row = migration["fixtures"].get(repo)
        if row:
            pin.update(
                revision=row["revision"],
                files_sha256=row["fixture_files_sha256"],
                target=row.get("target", pin.get("target", repo.split("/")[-1])),
            )
            fixtures[row["repo"]] = pin
        else:
            fixtures[repo] = pin
    result["fixtures"] = fixtures
    return rename_known_targets(result, migration)


def rename_known_targets(contract, migration):
    """Advance known snapshots, preserving custom revisions and contract content."""
    targets = migration.get("targets", {})
    integration = contract["integration"]
    pins = [(integration["repo"], integration), *contract.get("fixtures", {}).items()]
    for repo, pin in pins:
        row = targets.get(repo)
        if row and pin["revision"] not in [row["revision"], *row["previous_revisions"]]:
            return contract
    row = targets.get(integration["repo"])
    if (
        row
        and row.get("previous_contract_files_sha256")
        and integration.get("files_sha256")
        not in [row["previous_contract_files_sha256"], row["contract_files_sha256"]]
    ):
        return contract
    if row and integration["revision"] in row["previous_revisions"]:
        integration["revision"] = row["revision"]
        if row.get("contract_files_sha256"):
            integration["files_sha256"] = row["contract_files_sha256"]
    for repo, pin in contract.get("fixtures", {}).items():
        row = targets.get(repo)
        if row and pin["revision"] in [row["revision"], *row["previous_revisions"]]:
            pin.update(revision=row["revision"], target=row["target"])
    return contract
