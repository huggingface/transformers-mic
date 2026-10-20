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
        return result
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
                target=pin.get("target", repo.split("/")[-1]),
            )
            fixtures[row["repo"]] = pin
        else:
            fixtures[repo] = pin
    result["fixtures"] = fixtures
    return result
