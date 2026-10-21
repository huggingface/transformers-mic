import copy

import pytest

from transformers_mic.hub import consolidation, upgrade_contract


@pytest.mark.parametrize("repo", sorted(consolidation()["targets"]))
def test_known_canonical_snapshot_advances_without_changing_hashes(repo):
    row = consolidation()["targets"][repo]
    original = {
        "integration": {
            "repo": repo,
            "revision": row["previous_revisions"][0],
            "files_sha256": row.get("previous_contract_files_sha256", "a" * 64),
        },
        "fixtures": {
            repo: {
                "revision": row["previous_revisions"][0],
                "files_sha256": "b" * 64,
                "target": "legacy-slim",
            }
        },
    }
    source = copy.deepcopy(original)
    updated = upgrade_contract(original)
    assert original == source
    assert updated["integration"]["revision"] == row["revision"]
    assert updated["integration"]["files_sha256"] == row.get(
        "contract_files_sha256", "a" * 64
    )
    assert updated["fixtures"][repo]["revision"] == row["revision"]
    assert updated["fixtures"][repo]["files_sha256"] == "b" * 64
    assert updated["fixtures"][repo]["target"] == repo.split("/")[-1]


def test_unrecognized_custom_snapshot_is_preserved():
    repo = "huggingface-mic/bert-mic"
    source = {
        "integration": {"repo": repo, "revision": "f" * 40},
        "fixtures": {repo: {"revision": "f" * 40, "target": "custom-target"}},
    }
    assert upgrade_contract(source) == source


def test_date_fix_preserves_custom_contract_content():
    repo = "huggingface-mic/mistral3-mic"
    row = consolidation()["targets"][repo]
    source = {
        "integration": {
            "repo": repo,
            "revision": row["previous_revisions"][0],
            "files_sha256": "f" * 64,
        },
        "fixtures": {
            repo: {"revision": row["previous_revisions"][0], "target": "mistral3-mic"}
        },
    }
    assert upgrade_contract(source) == source


def test_legacy_alias_uses_renamed_baseline_identity():
    result = upgrade_contract(
        {
            "integration": {
                "repo": "huggingface-mic/mic-bert-integration",
                "revision": "old",
            },
            "fixtures": {
                "huggingface-mic/mic-bert-slim": {
                    "revision": "old",
                    "target": "mic-bert-slim",
                }
            },
        }
    )
    row = consolidation()["targets"]["huggingface-mic/bert-mic"]
    assert result["integration"]["revision"] == row["revision"]
    assert result["fixtures"]["huggingface-mic/bert-mic"]["revision"] == row["revision"]
    assert result["fixtures"]["huggingface-mic/bert-mic"]["target"] == "bert-mic"


def test_mixed_custom_fixture_does_not_partially_advance_baseline_pin():
    repo = "huggingface-mic/bert-mic"
    row = consolidation()["targets"][repo]
    source = {
        "integration": {"repo": repo, "revision": row["previous_revisions"][0]},
        "fixtures": {repo: {"revision": "f" * 40, "target": "custom-target"}},
    }
    assert upgrade_contract(source) == source
