import json

import pytest
from huggingface_hub.errors import EntryNotFoundError

from transformers_mic import fleet

LOCK = {
    "framework": "transformers",
    "environment": {"python": "3.11", "torch": "2.14.0"},
    "contracts": {
        "TimesFM": {
            "integration": {
                "repo": "huggingface-mic/timesfm-mic",
                "revision": "a" * 40,
                "files_sha256": "1" * 64,
            },
            "fixtures": {
                "huggingface-mic/timesfm-mic": {
                    "revision": "a" * 40,
                    "files_sha256": "2" * 64,
                }
            },
            "checkpoints": {"google/timesfm": "b" * 40},
        },
        # Two slim models, three checkpoints: the contract lives in the first
        "SmolLM": {
            "integration": {
                "repo": "huggingface-mic/smollm-mic",
                "revision": "c" * 40,
                "files_sha256": "3" * 64,
            },
            "fixtures": {
                "huggingface-mic/smollm-mic": {
                    "revision": "c" * 40,
                    "files_sha256": "4" * 64,
                    "target": "mic-SmolLM-slim",
                },
                "huggingface-mic/smollm-1.7b-mic": {
                    "revision": "d" * 40,
                    "files_sha256": "5" * 64,
                    "target": "mic-SmolLM-1.7B-slim",
                },
            },
            "checkpoints": {
                "HuggingFaceTB/SmolLM-135M": "e" * 40,
                "HuggingFaceTB/SmolLM-1.7B": "f" * 40,
            },
        },
    },
    "known_failures": {
        "SmolLM/mic-SmolLM-slim": "https://github.com/org/repo/issues/1",
        "TimesFM/timesfm-mic": {
            "issue": "https://github.com/org/repo/issues/2",
            "match": "overflow",
            "skip_outputs": ["training.loss", "training.gradients"],
        },
    },
}


def test_tables_round_trip_exactly():
    tables = fleet.to_tables(LOCK)
    assert json.dumps(fleet.from_tables(tables)) == json.dumps(LOCK)


def test_contract_table_has_one_readable_row_per_slim_model():
    rows = fleet.to_tables(LOCK)["contracts"].splitlines()
    assert rows[0] == (
        "contract,target,mic_repo,revision,contract_sha256,fixture_sha256,checkpoints"
    )
    assert rows[1] == (
        f"TimesFM,,huggingface-mic/timesfm-mic,{'a' * 40},{'1' * 64},{'2' * 64},"
        f"google/timesfm@{'b' * 40}"
    )
    assert rows[3] == (
        f"SmolLM,mic-SmolLM-1.7B-slim,huggingface-mic/smollm-1.7b-mic,{'d' * 40},,{'5' * 64},"
    )
    assert fleet.to_tables(LOCK)["known_failures"].splitlines()[2] == (
        "TimesFM,timesfm-mic,https://github.com/org/repo/issues/2,overflow,"
        "training.loss; training.gradients"
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.update(artifacts={"org/x": "a" * 40}), "cannot hold artifacts"),
        (lambda c: c["integration"].update(path="example/hub/x"), "published repo"),
        (
            lambda c: c["integration"].update(repo="huggingface-mic/other"),
            "fixture snapshots",
        ),
        (lambda c: c["integration"].update(revision="9" * 40), "fixture snapshots"),
        (
            lambda c: c["fixtures"]["huggingface-mic/timesfm-mic"].pop("files_sha256"),
            "files_sha256",
        ),
    ],
)
def test_tables_refuse_what_they_cannot_hold(change, message):
    lock = json.loads(json.dumps(LOCK))
    change(lock["contracts"]["TimesFM"])
    with pytest.raises(ValueError, match=message):
        fleet.to_tables(lock)


def test_tables_reject_a_contract_without_its_carrying_row():
    tables = fleet.to_tables(LOCK)
    tables["contracts"] = (
        "contract,target,mic_repo,revision,contract_sha256,fixture_sha256,checkpoints\n"
        f"SmolLM,,huggingface-mic/smollm-1.7b-mic,{'d' * 40},,{'5' * 64},\n"
    )
    with pytest.raises(ValueError, match="SmolLM: no row carries the contract"):
        fleet.from_tables(tables)


def test_export_operations_write_only_the_tables():
    files = fleet.operations("prototypes", LOCK)
    assert sorted(files) == [
        "prototypes/contracts.csv",
        "prototypes/environment.csv",
        "prototypes/known_failures.csv",
    ]
    assert (
        fleet.from_tables(
            {t: files[fleet.table_path("prototypes", t)] for t in fleet.COLUMNS}
        )
        == LOCK
    )


def test_download_reads_tables_and_falls_back_to_json_for_old_revisions(
    tmp_path, monkeypatch
):
    tables = fleet.to_tables(LOCK)
    legacy = {
        "1" * 40: LOCK,
        "2" * 40: {"legacy": True},
        "3" * 40: {**LOCK, "environment": {}},
        "4" * 40: None,
    }
    has_tables = {"1" * 40: True, "2" * 40: False, "3" * 40: True, "4" * 40: True}

    def download(repo, filename, repo_type, revision):
        assert (repo, repo_type) == (fleet.REPOSITORY, "dataset")
        path = tmp_path / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "fleet-lock.json":
            if legacy[revision] is None:
                raise EntryNotFoundError(filename)
            path.write_text(json.dumps({"comment": "ignored", **legacy[revision]}))
        elif has_tables[revision]:
            path.write_text(tables[filename.split("/")[1].removesuffix(".csv")])
        else:
            raise EntryNotFoundError(filename)
        return str(path)

    monkeypatch.setattr(fleet, "hf_hub_download", download)
    monkeypatch.setattr(fleet.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    path, revision = fleet.download("control", "1" * 40)
    assert revision == "1" * 40 and path.parent.parent.name == "tmic-fleet"
    assert json.loads(path.read_text()) == LOCK
    path, _ = fleet.download("control", "2" * 40)
    assert json.loads(path.read_text())["legacy"]
    # A JSON-only update by a tmic from before the tables is refused, not shadowed
    with pytest.raises(ValueError, match="tables and fleet-lock.json disagree"):
        fleet.download("control", "3" * 40)
    # Tables alone: the dataset after the JSON locks were removed
    path, _ = fleet.download("control", "4" * 40)
    assert json.loads(path.read_text()) == LOCK
