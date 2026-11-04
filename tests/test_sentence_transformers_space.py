import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "tools/update_sentence_transformers_space.py"
spec = importlib.util.spec_from_file_location("st_space", SOURCE)
space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(space)


def report(tmp_path, rows=(), conclusion="success", mode="check"):
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return space.result_run(
        tmp_path,
        run_id="123",
        attempt="1",
        conclusion=conclusion,
        mode=mode,
        repository="huggingface/transformers-mic",
        now="2026-11-04T10:00:00+00:00",
    )


def test_no_results_and_incomplete_runs_are_errors(tmp_path):
    assert report(tmp_path)["status"] == "error"
    assert (
        report(tmp_path, [{"status": "pass"}], conclusion="failure")["status"]
        == "error"
    )
    assert report(tmp_path, [{"status": "unsupported"}])["status"] == "error"


def test_recording_is_not_a_passing_check(tmp_path):
    assert (
        report(tmp_path, [{"status": "recorded"}], mode="record")["status"]
        == "recorded"
    )
    result = report(
        tmp_path,
        [
            {
                "status": "fail",
                "failures": ["scores differ"],
                "contract_name": "EmbeddingGemma",
            }
        ],
    )
    assert result["status"] == "fail"
    assert result["results"][0]["failures"] == ["scores differ"]
    assert (
        result["url"]
        == "https://github.com/huggingface/transformers-mic/actions/runs/123/attempts/1"
    )


def test_reports_versions_without_arbitrary_pip_urls(tmp_path):
    (tmp_path / "pip-freeze.txt").write_text(
        "torch==2.14.1\nprivate @ https://secret@example.com/package.whl\n"
    )
    (tmp_path / "fleet-revision.txt").write_text("a" * 40)
    result = report(tmp_path, [{"status": "pass"}])
    assert result["dependencies"] == ["torch==2.14.1"]
    assert result["fleet_revision"] == "a" * 40


def test_history_replaces_same_attempt_and_keeps_recent_90(tmp_path):
    latest = report(tmp_path, [{"status": "pass"}])
    previous = {
        "runs": [
            {**latest, "status": "fail"},
            *(
                {**latest, "id": str(i), "time": "2026-10-01T00:00:00+00:00"}
                for i in range(100)
            ),
        ]
    }
    result = space.history(previous, latest)
    assert len(result["runs"]) == 90
    assert result["runs"][0]["status"] == "pass"
    assert sum(row["id"] == latest["id"] for row in result["runs"]) == 1
