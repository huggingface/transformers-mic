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
