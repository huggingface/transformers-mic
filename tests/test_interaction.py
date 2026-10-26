import pytest

from transformers_mic.interaction import Interaction, Pending


def test_agent_baseline_answer_requires_reason_and_is_single_use():
    state = {}

    def context():
        return {"workspace_sha256": "content-A"}

    interaction = Interaction("agent", state, lambda: None, context)
    with pytest.raises(Pending) as first:
        interaction.ask(
            "Approve baseline?", "diff", kind="baseline-change", reason=True
        )
    question = first.value.record
    interaction.answer = {"question_id": question["question_id"], "answer": "approve"}
    with pytest.raises(Pending) as missing:
        interaction.ask(
            "Approve baseline?", "diff", kind="baseline-change", reason=True
        )
    assert "reason" in missing.value.error
    interaction.answer["reason"] = "Measured correction on the reference runtime"
    old_answer = interaction.answer.copy()
    assert (
        interaction.ask(
            "Approve baseline?", "diff", kind="baseline-change", reason=True
        )
        == old_answer["reason"]
    )
    assert not state.get("pending_question") and len(state["decisions"]) == 1
    interaction.answer = old_answer
    with pytest.raises(Pending) as replay:
        interaction.ask(
            "Approve baseline?", "diff", kind="baseline-change", reason=True
        )
    assert "already been used" in replay.value.error
    assert replay.value.record["question_id"] != question["question_id"]


def test_context_change_invalidates_question_even_with_same_text():
    state = {}
    content = {"base_revision": "old-dataset-revision"}
    interaction = Interaction("agent", state, lambda: None, lambda: content)
    with pytest.raises(Pending) as old:
        interaction.ask("Publish fleet?", "same text", kind="fleet-update")
    interaction.answer = {
        "question_id": old.value.record["question_id"],
        "answer": "approve",
    }
    content["base_revision"] = "new-dataset-revision"
    with pytest.raises(Pending) as current:
        interaction.ask("Publish fleet?", "same text", kind="fleet-update")
    assert current.value.record["question_id"] != old.value.record["question_id"]
    assert "does not match" in current.value.error
    assert not state.get("decisions")


def test_defer_preserves_question_and_does_not_approve():
    state = {}
    interaction = Interaction("agent", state, lambda: None, lambda: {})
    with pytest.raises(Pending) as first:
        interaction.ask("Publish?", "preview")
    interaction.answer = {
        "question_id": first.value.record["question_id"],
        "answer": "defer",
    }
    with pytest.raises(Pending) as deferred:
        interaction.ask("Publish?", "preview")
    assert deferred.value.record == first.value.record
    assert not state.get("decisions")


def test_always_approves_later_human_questions_and_records_authorization(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompts = []

    def choose(prompt):
        prompts.append(prompt)
        return "A"

    monkeypatch.setattr("builtins.input", choose)
    state = {}
    interaction = Interaction("human", state, lambda: None, lambda: {})
    assert (
        interaction.ask("Review contract?", "source", kind="source-review") == "approve"
    )
    authorization = state["decisions"][0]["question_id"]
    assert (
        interaction.ask("Publish model?", "preview", kind="model-publication")
        == "approve"
    )
    reason = interaction.ask(
        "Approve baseline?", "diff", kind="baseline-change", reason=True
    )
    assert "always yes" in reason
    assert len(prompts) == 1 and "A=always" in prompts[0]
    assert all(
        d["authorization_question_id"] == authorization for d in state["decisions"]
    )
    # A fresh invocation does not inherit a previous process's blanket approval.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    fresh = Interaction("human", state, lambda: None, lambda: {})
    with pytest.raises(Pending):
        fresh.ask("Publish another model?", "preview")


def test_always_stops_unchanged_retry_loop_even_if_log_changes(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "a")
    state = {}
    interaction = Interaction(
        "human", state, lambda: None, lambda: {"workspace_sha256": "unchanged"}
    )
    interaction.ask("Start?", "model")
    for version in ("first log", "second log"):
        assert (
            interaction.ask(
                "Retry build?",
                version,
                kind="retry",
                context={"failed_stage": "build", "exit_code": 1},
            )
            == "retry"
        )
    with pytest.raises(Pending):
        interaction.ask(
            "Retry build?",
            "third log",
            kind="retry",
            context={"failed_stage": "build", "exit_code": 1},
        )
    assert len(state["decisions"]) == 3


def test_details_can_be_opened_without_approving_or_changing_question(monkeypatch):
    from transformers_mic import terminal

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["d", "N"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    opened = []
    monkeypatch.setattr(terminal, "details", opened.append)
    state = {}
    interaction = Interaction("human", state, lambda: None, lambda: {})
    with pytest.raises(Pending) as pending:
        interaction.ask("Review?", "full diff\nline 2\nline 3\nline 4")
    assert opened == ["full diff\nline 2\nline 3\nline 4"]
    assert not state.get("decisions") and not interaction.always_yes
    assert pending.value.record == state["pending_question"]
