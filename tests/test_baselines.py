from io import BytesIO
from pathlib import Path
import json
import stat
from types import SimpleNamespace
import zipfile

import pytest

from transformers_mic import baselines, baseline_worker, dashboard_auth, cloud_progress
from transformers_mic.cli import parser
from transformers_mic.workspace import read, write


JOB = {
    "id": "a" * 32,
    "fleet_revision": "b" * 40,
    "transformers_sha": "c" * 40,
    "mic_sha": "d" * 40,
    "fleets": ["control"],
    "fixtures_only": False,
}


def make_collection(root, *, changed=False):
    state = {
        "dashboard": "https://dashboard.test",
        "request": {
            "profiles": ["cpu", "cuda"],
            "fleets": ["control"],
            "fixtures_only": False,
            "transformers_ref": "main",
            "fleet_revision": JOB["fleet_revision"],
            "upload_repo": None,
            "request_id": "e" * 32,
        },
        "job": {**JOB, "status": "completed", "upload_repo": None},
    }
    write(root / "cloud.json", state)
    for profile, platform in (("cpu", "cpu-x86_64"), ("cuda", "cuda")):
        folder = root / "artifacts" / profile
        base = folder / "control/Demo"
        manifest = {
            "outputs": {"answer": {"compare": "allclose", "atol": 0.01, "rtol": 0.01}}
        }
        write(base / "integration.json", manifest)
        value = {
            "recorded_with": {
                "framework": "transformers",
                "target": "mic-demo",
                "variant": "default",
                "contract": "huggingface-mic/demo-mic",
                "environment": {},
            },
            "outputs": {"runtime": {"device": profile}, "answer": [1.0]},
        }
        candidate = base / "candidates" / platform / "mic-demo.candidate.json"
        write(candidate, value)
        old = {
            **value,
            "outputs": {**value["outputs"], "answer": [9.0] if changed else [1.0]},
        }
        write(base / "references" / platform / "mic-demo.json", old)
        files = {
            p.relative_to(folder).as_posix(): baselines.sha256(p)
            for p in (
                candidate,
                base / "integration.json",
                base / "references" / platform / "mic-demo.json",
            )
        }
        case = {
            "key": "control/Demo/mic-demo/default",
            "repo": "huggingface-mic/demo-mic",
            "revision": "f" * 40,
            "status": "validated",
            "platform": platform,
            "candidate": candidate.relative_to(folder).as_posix(),
            "files": files,
        }
        write(
            folder / "collection.json",
            {
                "request": JOB,
                "profile": profile,
                "complete": True,
                "expected_cases": [case["key"]],
                "cases": [case],
                "models": [],
            },
        )
    return state


def zip_folder(folder):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for path in folder.rglob("*"):
            if path.is_file():
                archive.writestr(path.relative_to(folder).as_posix(), path.read_bytes())
    return stream.getvalue()


def test_complete_collection_upload_is_previewed_and_idempotent(tmp_path, monkeypatch):
    make_collection(tmp_path)
    calls = []

    class Api:
        def create_repo(self, *a, **kw):
            calls.append(("create", a, kw))

        def upload_folder(self, **kw):
            calls.append(("upload", kw))
            return SimpleNamespace(oid="1" * 40)

    monkeypatch.setattr(baselines, "HfApi", Api)
    assert baselines.upload(tmp_path, "user/candidates", False) == 0
    assert calls == []
    assert baselines.upload(tmp_path, "user/candidates", True) == 0
    assert baselines.upload(tmp_path, "user/candidates", True) == 0
    assert [c[0] for c in calls] == ["create", "upload"]
    assert calls[1][1]["path_in_repo"] == "collections/" + JOB["id"]


@pytest.mark.parametrize(
    "problem",
    [
        "missing-case",
        "duplicate-case",
        "wrong-job",
        "tamper",
        "incomplete",
        "wrong-profile",
        "model-tamper",
    ],
)
def test_incomplete_or_modified_measurements_cannot_be_uploaded(tmp_path, problem):
    make_collection(tmp_path)
    folder = tmp_path / "artifacts/cuda"
    path = folder / "collection.json"
    report = read(path)
    if problem == "missing-case":
        report["expected_cases"].append("control/Demo/missing/default")
    elif problem == "duplicate-case":
        report["cases"].append(report["cases"][0])
    elif problem == "wrong-job":
        report["request"] = {**JOB, "id": "0" * 32}
    elif problem == "tamper":
        (folder / report["cases"][0]["candidate"]).write_text("{}")
    elif problem == "incomplete":
        report["complete"] = False
    elif problem == "model-tamper":
        write(folder / "models/config.json", {})
        report["models"] = [{"files": {"models/config.json": "0" * 64}}]
    else:
        report["profile"] = "cpu"
    write(path, report)
    with pytest.raises(ValueError):
        baselines.collection(tmp_path)


@pytest.mark.parametrize(
    "name", ["../outside", "/outside", "folder/../../outside", "folder\\outside"]
)
def test_artifact_extraction_rejects_unsafe_paths(tmp_path, name):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        baselines.extract_archive(stream.getvalue(), tmp_path)


def test_artifacts_reject_symlinks_and_normalized_duplicates(tmp_path):
    for entries in (("symlink",), ("a//b", "a/b")):
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name in entries:
                item = zipfile.ZipInfo(name)
                if name == "symlink":
                    item.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(item, "data")
        with pytest.raises(ValueError, match="Unsafe"):
            baselines.extract_archive(stream.getvalue(), tmp_path)


def test_review_binds_approval_to_measurements_and_changed_values_need_reason(tmp_path):
    make_collection(tmp_path, changed=True)
    assert baselines.review(tmp_path) == 3
    assert baselines.review(tmp_path, approve=True) == 3
    with pytest.raises(ValueError, match="--reason"):
        baselines.review(tmp_path, approve_changed=True)
    assert (
        baselines.review(
            tmp_path,
            approve=True,
            approve_changed=True,
            reason="Reviewed the framework change",
        )
        == 0
    )
    receipt = read(tmp_path / "reviews.json")
    assert len(receipt["approved"]) == 2
    assert all(
        value["reason"] == "Reviewed the framework change"
        for value in receipt["approved"].values()
    )


def test_publishing_opens_hub_prs_and_never_writes_unreviewed_values(
    tmp_path, monkeypatch
):
    make_collection(tmp_path)
    write(
        tmp_path / "reviews.json",
        {
            "collection": baselines.fingerprint(baselines.collection(tmp_path)[1]),
            "approved": {},
        },
    )
    with pytest.raises(ValueError, match="Unreviewed"):
        baselines.publish(tmp_path)
    assert baselines.review(tmp_path, approve=True) == 0
    commits = []

    class Api:
        def get_repo_discussions(self, repo, discussion_type):
            return []

        def create_commit(self, **kwargs):
            commits.append(kwargs)
            return SimpleNamespace(
                oid="1" * 40, pr_url="https://huggingface.co/example/discussions/1"
            )

    monkeypatch.setattr(baselines, "HfApi", Api)
    assert baselines.publish(tmp_path, False) == 0
    assert not commits
    assert baselines.publish(tmp_path, True) == 0
    assert commits[0]["create_pr"] is True
    assert commits[0]["parent_commit"] == "f" * 40
    assert len(commits[0]["operations"]) == 2
    assert baselines.publish(tmp_path, True) == 0
    assert len(commits) == 1


def test_publishing_adopts_a_pr_opened_by_a_stalled_call(tmp_path, monkeypatch):
    state = make_collection(tmp_path)
    assert baselines.review(tmp_path, approve=True) == 0
    title = "Review cloud MIC baselines " + state["job"]["id"]
    url = "https://huggingface.co/example/discussions/3"

    class Api:
        def get_repo_discussions(self, repo, discussion_type):
            return [
                SimpleNamespace(num=2, status="closed", title=title, url="closed"),
                SimpleNamespace(num=3, status="open", title=title, url=url),
            ]

        def get_discussion_details(self, repo, num):
            assert num == 3
            return SimpleNamespace(
                events=[SimpleNamespace(type="commit", oid="2" * 40)]
            )

        def create_commit(self, **kwargs):
            raise AssertionError("opened a duplicate PR")

    monkeypatch.setattr(baselines, "HfApi", Api)
    assert baselines.publish(tmp_path, True) == 0
    (publication,) = read(tmp_path / "cloud.json")["publications"].values()
    assert publication["revision"] == "2" * 40
    assert publication["url"] == url


def test_cli_submission_download_resume_and_upload_receipt(tmp_path, monkeypatch):
    source = tmp_path / "source"
    state = make_collection(source)
    job = {
        **state["job"],
        "artifacts": [{"profile": "cpu", "id": 1}, {"profile": "cuda", "id": 2}],
    }
    archives = {
        1: zip_folder(source / "artifacts/cpu"),
        2: zip_folder(source / "artifacts/cuda"),
    }
    calls = []

    def request(url, path, **kwargs):
        calls.append((path, kwargs))
        return job

    monkeypatch.setattr(dashboard_auth, "request", request)
    monkeypatch.setattr(
        dashboard_auth,
        "request_bytes",
        lambda url, path, **kw: archives[int(path.rsplit("/", 1)[1])],
    )
    monkeypatch.setattr(dashboard_auth, "access_token", lambda url: "dashboard-token")
    monkeypatch.setattr(baselines.fleet, "resolve", lambda ref: JOB["fleet_revision"])
    out = tmp_path / "local"
    args = parser().parse_args(
        [
            "baselines",
            "run",
            "--fleet",
            "control",
            "--out",
            str(out),
            "--dashboard",
            "https://dashboard.test",
        ]
    )
    assert baselines.run(args) == 0
    assert baselines.collection(out)[0]["job"]["id"] == JOB["id"]
    assert baselines.run(args) == 0
    assert sum(path == "/mic-jobs" for path, _ in calls) == 1
    args.fixtures_only = True
    with pytest.raises(ValueError, match="options changed"):
        baselines.run(args)


def test_cli_can_leave_models_in_cloud_and_return_only_hub_revision(
    tmp_path, monkeypatch
):
    job = {
        **JOB,
        "upload_repo": "huggingface-mic/candidates",
        "status": "completed",
        "publication_artifact": {"id": 3},
    }
    receipt = {
        "job_id": JOB["id"],
        "repository": job["upload_repo"],
        "revision": "1" * 40,
    }
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("publication.json", json.dumps(receipt))
    downloads = []
    monkeypatch.setattr(dashboard_auth, "request", lambda *a, **kw: job)
    monkeypatch.setattr(
        dashboard_auth,
        "request_bytes",
        lambda url, path, **kw: downloads.append(path) or stream.getvalue(),
    )
    monkeypatch.setattr(dashboard_auth, "access_token", lambda url: "dashboard-token")
    monkeypatch.setattr(baselines.fleet, "resolve", lambda ref: JOB["fleet_revision"])
    args = parser().parse_args(
        [
            "baselines",
            "run",
            "--fleet",
            "control",
            "--out",
            str(tmp_path),
            "--upload-repo",
            job["upload_repo"],
            "--no-download",
        ]
    )
    assert baselines.run(args) == 0
    assert len(downloads) == 1 and downloads[0].endswith("/3")
    assert read(tmp_path / "cloud.json")["remote_upload"] == receipt
    assert not (tmp_path / "artifacts").exists()


def test_case_accounting_rejects_missing_duplicate_and_hardware_skips():
    recorded = {
        "target": "demo",
        "variant": "default",
        "status": "recorded",
        "platform": "cuda",
    }
    checked = {**recorded, "status": "pass"}
    assert (
        baseline_worker.assess(["demo"], "default", [recorded], [checked], "cuda")[0][
            "status"
        ]
        == "validated"
    )
    for rows in ([], [recorded, recorded]):
        with pytest.raises(ValueError):
            baseline_worker.assess(["demo"], "default", rows, [checked], "cuda")
    hardware = {**recorded, "status": "unsupported", "skip": "hardware"}
    assert (
        baseline_worker.assess(["demo"], "default", [hardware], [], "cuda")[0]["status"]
        == "failed"
    )
    declared = {**hardware, "skip": "declared"}
    assert (
        baseline_worker.assess(["demo"], "default", [declared], [], "cuda")[0]["status"]
        == "declared-unsupported"
    )


def test_manifest_target_matrix_includes_checkpoints_unless_explicitly_excluded():
    contract = {
        "fixtures": {"org/slim": {"target": "tiny"}},
        "checkpoints": {"org/full": "a" * 40},
    }
    manifest = {
        "fixtures": {"org/slim": {}},
        "checkpoints": {"org/full": {}},
        "integrations": {"transformers": {"targets": ["fixtures", "checkpoints"]}},
    }
    assert baseline_worker.expected_targets(contract, manifest, False) == [
        "tiny",
        "full",
    ]
    assert baseline_worker.expected_targets(contract, manifest, True) == ["tiny"]
    manifest["integrations"]["transformers"]["resources"] = {
        "org/full": {"devices": ["cuda"], "gpu_memory_gb": 48}
    }
    assert baseline_worker.profile_exclusions(contract, manifest, "default", "cpu") == {
        "full"
    }
    assert (
        baseline_worker.profile_exclusions(contract, manifest, "default", "cuda")
        == set()
    )


@pytest.mark.parametrize("omit_case,runner_code", [(False, 0), (True, 0), (False, 2)])
def test_worker_records_verifies_and_accounts_for_the_whole_matrix(
    tmp_path, monkeypatch, omit_case, runner_code
):
    import sys

    model = tmp_path / "hub"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"slim weights")
    manifest = {
        "fixtures": {"huggingface-mic/demo-mic": {}},
        "checkpoints": {"org/full": {}},
        "integrations": {
            "transformers": {
                "targets": ["fixtures", "checkpoints"],
                "variants": {"default": {}, "bfloat16": {"dtype": "bfloat16"}},
            }
        },
        "outputs": {"answer": {"compare": "allclose", "atol": 0.01, "rtol": 0.01}},
    }
    write(model / "integration.json", manifest)
    contract = {
        "integration": {"repo": "huggingface-mic/demo-mic", "revision": "f" * 40},
        "fixtures": {
            "huggingface-mic/demo-mic": {"revision": "f" * 40, "target": "tiny"}
        },
        "checkpoints": {"org/full": "f" * 40},
    }
    lock = tmp_path / "fleet.lock"
    write(lock, {"contracts": {"Demo": contract}, "environment": {"python": "3.11"}})
    checkout = tmp_path / "framework"
    monkeypatch.setenv("MIC_TRANSFORMERS_CHECKOUT", str(checkout))
    monkeypatch.delenv("MIC_PROGRESS_URL", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__file__=str(checkout / "src/transformers/__init__.py")),
    )
    monkeypatch.setattr(
        baseline_worker, "runtime", lambda profile: {"profile": profile}
    )
    monkeypatch.setattr(
        baseline_worker.subprocess,
        "check_output",
        lambda *a, **kw: JOB["transformers_sha"],
    )
    monkeypatch.setattr(
        baseline_worker.fleet, "download", lambda *a: (lock, JOB["fleet_revision"])
    )
    monkeypatch.setattr(
        baseline_worker, "hf_hub_download", lambda *a, **kw: model / "integration.json"
    )
    monkeypatch.setattr(baseline_worker, "snapshot_download", lambda *a, **kw: model)
    current_profile = "cpu"
    stages = []

    def execute(root, command, stage):
        stages.append(stage)
        variant = command[command.index("--variant") + 1]
        output = Path(command[command.index("--output") + 1])
        folder = Path(command[command.index("--expectations") + 1])
        platform = "cuda" if current_profile == "cuda" else "cpu-x86_64"
        targets = ["tiny"] if omit_case else ["tiny", "full"]
        rows = []
        for target in targets:
            row = {
                "target": target,
                "variant": variant,
                "platform": platform,
                "status": "recorded" if "--record" in command else "pass",
            }
            if "--record" in command:
                suffix = "" if variant == "default" else "." + variant
                write(
                    folder / platform / (target + suffix + ".candidate.json"),
                    {
                        "recorded_with": row,
                        "outputs": {
                            "runtime": {"device": current_profile},
                            "answer": [1.0],
                        },
                    },
                )
            rows.append(row)
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return runner_code

    monkeypatch.setattr(baseline_worker, "execute", execute)
    out = tmp_path / "collection"
    for profile in ("cpu", "cuda"):
        current_profile = profile
        args = SimpleNamespace(
            job_id=JOB["id"],
            fleet_revision=JOB["fleet_revision"],
            transformers_sha=JOB["transformers_sha"],
            mic_sha=JOB["mic_sha"],
            fleets="control",
            fixtures_only=False,
            profile=profile,
            out=out / "artifacts" / profile,
        )
        assert baseline_worker.measure(args) == int(omit_case or bool(runner_code))
        report = read(args.out / "collection.json")
        assert len(report["expected_cases"]) == 4 and len(report["cases"]) == 4
        assert report["complete"] is (not omit_case and not runner_code)
        assert bool(report["models"]) is (profile == "cuda")
    assert len(stages) == 8
    write(out / "cloud.json", {"job": JOB, "request": {"profiles": ["cpu", "cuda"]}})
    if omit_case or runner_code:
        with pytest.raises(ValueError, match="incomplete"):
            baselines.collection(out)
    else:
        baselines.collection(out)


def test_cli_progress_shows_stages_counts_and_suppresses_duplicate_updates(capsys):
    job = {
        "status": "running",
        "workflow_jobs": [
            {
                "name": "measure (cuda)",
                "status": "in_progress",
                "step": "Install environment",
            }
        ],
        "progress": {
            "cpu": {
                "profile": "cpu",
                "stage": "validating",
                "completed": 4,
                "total": 30,
                "detail": "Demo/bfloat16",
            }
        },
    }
    signature = cloud_progress.display(job)
    message = capsys.readouterr().err
    assert "Install environment" in message and "[cpu] validating 4/30" in message
    cloud_progress.display(job, signature)
    assert capsys.readouterr().err == ""


def test_terminal_progress_refreshes_same_block_and_explains_runner_queue(monkeypatch):
    import io

    class TTY(io.StringIO):
        def isatty(self):
            return True

    output = TTY()
    monkeypatch.setattr(cloud_progress.sys, "stderr", output)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setattr(cloud_progress.time, "monotonic", lambda: 100.0)
    job = {
        "status": "running",
        "workflow_jobs": [{"name": "measure (cpu)", "status": "queued", "step": ""}],
    }
    first = cloud_progress.display(job)
    assert "CPU: Waiting for a runner" in output.getvalue()
    offset = output.tell()
    job["workflow_jobs"][0].update(
        status="in_progress",
        step="Install fleet environment",
        completed_steps=2,
        total_steps=7,
    )
    second = cloud_progress.display(job, first)
    assert "\x1b[2A\r\x1b[J" in output.getvalue()[offset:]
    assert (
        "2/7" in output.getvalue()[offset:]
        and "Install fleet environment" in output.getvalue()[offset:]
    )
    offset = output.tell()
    assert cloud_progress.display(job, second) == second
    assert output.tell() == offset


def test_redirected_progress_is_plain_and_only_changed_rows_are_printed(
    monkeypatch, capsys
):
    monkeypatch.delenv("TMIC_LIVE_PROGRESS", raising=False)
    job = {
        "status": "running",
        "workflow_jobs": [
            {"name": "measure (cpu)", "status": "queued", "step": ""},
            {
                "name": "measure (cuda)",
                "status": "in_progress",
                "step": "Measure baselines",
            },
        ],
    }
    first = cloud_progress.display(job)
    capsys.readouterr()
    job["workflow_jobs"][0].update(status="in_progress", step="Install environment")
    cloud_progress.display(job, first)
    text = capsys.readouterr().err
    assert "Install environment" in text
    assert "CUDA" not in text and "[cloud]" not in text and "\x1b" not in text


def test_progress_uses_cached_short_lived_oidc_and_never_forwards_hub_credentials(
    monkeypatch, capsys
):
    monkeypatch.setenv("MIC_PROGRESS_URL", "https://serge.test/dashboard/mic-progress")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.test/token?job=1")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runner-oidc-secret")
    monkeypatch.setenv("HF_TOKEN", "hub-secret")
    calls = []

    class Response:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return self.data

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            return Response(
                b'{"value":"short-oidc-token"}'
                if request.full_url.startswith("https://oidc.test")
                else b"{}"
            )

    monkeypatch.setattr(cloud_progress, "build_opener", lambda *a: Opener())
    reporter = cloud_progress.Progress(JOB["id"], "cuda")
    reporter.emit("recording", 0, 2)
    reporter.emit("validating", 1, 2)
    assert len(calls) == 3
    assert "audience=serge-mic-baselines" in calls[0].full_url
    assert calls[1].get_header("Authorization") == "Bearer short-oidc-token"
    assert json.loads(calls[2].data)["sequence"] == 2
    output = capsys.readouterr()
    assert "hub-secret" not in output.out + output.err
    assert "runner-oidc-secret" not in output.out + output.err


def test_progress_failure_is_nonfatal_and_uses_a_cooldown(monkeypatch, capsys):
    monkeypatch.setenv("MIC_PROGRESS_URL", "https://serge.test/dashboard/mic-progress")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc.test/token")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runner-token")
    calls = []

    class Opener:
        def open(self, *a, **kw):
            calls.append(1)
            raise OSError("offline")

    monkeypatch.setattr(cloud_progress, "build_opener", lambda *a: Opener())
    reporter = cloud_progress.Progress(JOB["id"], "cpu")
    reporter.emit("recording", 0, 2)
    reporter.emit("validating", 1, 2)
    assert calls == [1]
    assert capsys.readouterr().err.count("relay unavailable") == 1


def test_worker_leaves_excluded_contracts_out_of_the_matrix(tmp_path, monkeypatch):
    import sys

    model = tmp_path / "hub"
    model.mkdir()
    write(
        model / "integration.json",
        {
            "fixtures": {"huggingface-mic/demo-mic": {}},
            "checkpoints": {},
            "integrations": {
                "transformers": {"targets": ["fixtures"], "variants": {"default": {}}}
            },
            "outputs": {"answer": {"compare": "allclose", "atol": 0.01, "rtol": 0.01}},
        },
    )

    def contract(repo):
        return {
            "integration": {"repo": repo, "revision": "f" * 40},
            "fixtures": {repo: {"revision": "f" * 40, "target": repo.split("/")[1]}},
            "checkpoints": {},
        }

    lock = tmp_path / "fleet.lock"
    write(
        lock,
        {
            "contracts": {
                "Demo": contract("huggingface-mic/demo-mic"),
                "Broken": contract("huggingface-mic/broken-mic"),
            }
        },
    )
    checkout = tmp_path / "framework"
    monkeypatch.setenv("MIC_TRANSFORMERS_CHECKOUT", str(checkout))
    monkeypatch.delenv("MIC_PROGRESS_URL", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__file__=str(checkout / "src/transformers/__init__.py")),
    )
    monkeypatch.setattr(baseline_worker, "runtime", lambda profile: {})
    monkeypatch.setattr(
        baseline_worker.subprocess,
        "check_output",
        lambda *a, **kw: JOB["transformers_sha"],
    )
    monkeypatch.setattr(
        baseline_worker.fleet, "download", lambda *a: (lock, JOB["fleet_revision"])
    )
    monkeypatch.setattr(
        baseline_worker, "hf_hub_download", lambda *a, **kw: model / "integration.json"
    )
    monkeypatch.setattr(baseline_worker, "snapshot_download", lambda *a, **kw: model)
    measured = []

    def execute(root, command, stage):
        measured.append(command)
        return 1

    monkeypatch.setattr(baseline_worker, "execute", execute)
    args = SimpleNamespace(
        job_id=JOB["id"],
        fleet_revision=JOB["fleet_revision"],
        transformers_sha=JOB["transformers_sha"],
        mic_sha=JOB["mic_sha"],
        fleets="control",
        fixtures_only=True,
        exclude="Broken",
        profile="cpu",
        out=tmp_path / "out",
    )
    baseline_worker.measure(args)
    report = read(args.out / "collection.json")
    assert report["request"]["exclude"] == ["Broken"]
    assert report["expected_cases"] == ["control/Demo/demo-mic/default"]
    assert measured and not any("broken" in " ".join(map(str, c)) for c in measured)
