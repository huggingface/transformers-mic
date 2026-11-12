import importlib.metadata
import json
import subprocess

import pytest
from packaging.requirements import Requirement

from transformers_mic import backend, cloud_environment
from transformers_mic.cli import main
from transformers_mic.workspace import Workspace

MAIN = "a72a1c385297bfaf1c8a07b5f4e689a315b90b77"
OLDER = "1" * 40
RUNTIME = [
    Requirement(text) for text in ("transformers>=5", "torch>=2.11", "num2words")
]


class Distribution:
    def __init__(self, version, direct_url=None):
        self.version = version
        self.direct_url = direct_url

    def read_text(self, name):
        assert name == "direct_url.json"
        return json.dumps(self.direct_url) if self.direct_url else None


def installed(**versions):
    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    return version


EVERYTHING = installed(transformers="5.23.0.dev0", torch="2.15.0", num2words="0.5.14")


def test_runtime_extra_is_what_the_cloud_runners_have():
    # Transformers' `testing` extra installs transformers-mic[runtime] into its CI images;
    # SmolVLM's processor needs num2words, manga-ocr's tokenizer fugashi + unidic-lite.
    names = {r.name for r in cloud_environment.runtime_requirements()}
    assert {"transformers", "torch", "num2words", "fugashi", "unidic-lite"} <= names


def test_main_commit_matches_without_problems():
    source = cloud_environment.transformers_source(
        Distribution(
            "5.23.0.dev0",
            {
                "url": "https://github.com/huggingface/transformers",
                "vcs_info": {"vcs": "git", "commit_id": MAIN},
            },
        )
    )
    assert source == ("5.23.0.dev0", MAIN, None)
    assert cloud_environment.compare(RUNTIME, source, MAIN, EVERYTHING) == ([], [], [])


def test_release_stops_with_the_install_command_for_main():
    # vip-llava-7b-hf-mic: 5.22.1 built a [64, 320] projector the fleet's commit stores as [64, 64].
    source = cloud_environment.transformers_source(Distribution("5.22.1"))
    assert source == ("5.22.1", None, None)
    problems, notes, install = cloud_environment.compare(
        RUNTIME,
        source,
        MAIN,
        installed(transformers="5.22.1", torch="2.15.0", num2words="0.5.14"),
    )
    assert "5.22.1 is a release" in problems[0] and MAIN in problems[0]
    assert install == [
        f"transformers @ git+https://github.com/huggingface/transformers@{MAIN}"
    ]
    assert notes == []


def test_missing_runtime_package_stops_and_is_named():
    # smolvlm-mic: ImportError: Package num2words is required.
    problems, _, install = cloud_environment.compare(
        RUNTIME,
        ("5.23.0.dev0", MAIN, None),
        MAIN,
        installed(transformers="5.23.0.dev0", torch="2.10.0"),
    )
    assert (
        "num2words" in problems[0] and "torch>=2.11 (installed 2.10.0)" in problems[0]
    )
    assert install == ["torch>=2.11", "num2words"]


def test_other_commit_only_notes_the_difference():
    problems, notes, install = cloud_environment.compare(
        RUNTIME, ("5.23.0.dev0", OLDER, None), MAIN, EVERYTHING
    )
    assert problems == [] and install == []
    assert (
        OLDER[:12] in notes[0] and MAIN[:12] in notes[0] and "pip install" in notes[0]
    )


def test_unreachable_main_does_not_stop_a_commit():
    problems, notes, _ = cloud_environment.compare(
        RUNTIME, ("5.23.0.dev0", OLDER, None), None, EVERYTHING
    )
    assert problems == [] and "Could not resolve" in notes[0]


def test_local_source_tree_is_traced_with_git(tmp_path):
    tree = tmp_path / "transformers"
    tree.mkdir()
    git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (tree / "setup.py").write_text("")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    origin = {"url": tree.as_uri(), "dir_info": {"editable": True}}
    assert cloud_environment.transformers_source(
        Distribution("5.23.0.dev0", origin)
    ) == ("5.23.0.dev0", head, None)
    (tree / "setup.py").write_text("changed")
    assert (
        cloud_environment.transformers_source(Distribution("5.23.0.dev0", origin))[2]
        == tree
    )


@pytest.fixture
def auto_workspace(tmp_path, monkeypatch):
    project = Workspace(tmp_path / "project")
    project.from_catalog("google-bert/bert-base-uncased", "BERT", "auto", False)
    calls = []
    monkeypatch.setattr(
        backend,
        "invoke",
        lambda *args, **kwargs: (
            calls.append(args) or subprocess.CompletedProcess(args, 0)
        ),
    )
    monkeypatch.setattr(cloud_environment, "runtime_requirements", lambda: RUNTIME)
    monkeypatch.setattr(cloud_environment, "main_commit", lambda: MAIN)
    monkeypatch.setattr(
        cloud_environment, "transformers_source", lambda: ("5.22.1", None, None)
    )
    monkeypatch.setattr(
        cloud_environment, "missing", lambda requirements, version: [(RUNTIME[2], None)]
    )
    return project, calls


def test_auto_match_cloud_stops_before_the_factory_check(auto_workspace, capsys):
    project, calls = auto_workspace
    assert main(["--workspace", str(project.root), "auto", "--match-cloud"]) == 2
    assert calls == []
    err = capsys.readouterr().err.strip().splitlines()
    # The stage summary shows the last line: the exact command.
    assert err[-1] == (
        f"Fix: python -m pip install num2words 'transformers @ git+https://github.com/huggingface/transformers@{MAIN}'"
    )
    assert "Traceback" not in "\n".join(err)  # never mistaken for a contract failure


def test_auto_without_match_cloud_and_with_override_runs(
    auto_workspace, monkeypatch, capsys
):
    project, calls = auto_workspace
    assert main(["--workspace", str(project.root), "auto"]) == 0
    monkeypatch.setenv(cloud_environment.ALLOW_DRIFT, "1")
    assert main(["--workspace", str(project.root), "auto", "--match-cloud"]) == 0
    assert len(calls) == 2
    assert "continuing" in capsys.readouterr().err
