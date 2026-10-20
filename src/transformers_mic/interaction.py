"""Human prompts and the versioned JSON question/answer protocol."""

import hashlib
import json
import sys
import uuid

PROTOCOL_VERSION = 1


class Pending(Exception):
    def __init__(self, question, details, *, record=None, error=None):
        self.question, self.details = question, details
        self.record, self.error = record, error


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_answer(path):
    if path is None:
        return None
    with open(path) as handle:
        answer = json.load(handle)
    if not isinstance(answer, dict) or set(answer) - {
        "question_id",
        "answer",
        "reason",
    }:
        raise ValueError(
            "Answer must be a JSON object with question_id, answer, and optional reason"
        )
    if not isinstance(answer.get("question_id"), str) or not answer["question_id"]:
        raise ValueError("Answer requires a nonempty question_id")
    if answer.get("answer") not in ("approve", "retry", "defer"):
        raise ValueError("Answer must be approve, retry, or defer")
    if "reason" in answer and not isinstance(answer["reason"], str):
        raise ValueError("Answer reason must be a string")
    return answer


class Interaction:
    def __init__(self, mode, state, save, context, answer=None):
        self.mode, self.state, self.save, self.context = mode, state, save, context
        self.answer = answer

    def ask(
        self, question, details, *, reason=False, kind="confirmation", context=None
    ):
        snapshot = {
            "kind": kind,
            "question": question,
            "details": details,
            "choices": ["retry" if kind == "retry" else "approve", "defer"],
            "requires_reason": reason,
            "context": {**self.context(), **(context or {})},
        }
        context_hash = digest(snapshot)
        record = self.state.get("pending_question")
        if not record or record["context_sha256"] != context_hash:
            record = {
                "protocol_version": PROTOCOL_VERSION,
                "question_id": uuid.uuid4().hex,
                "context_sha256": context_hash,
                **snapshot,
            }
            self.state["pending_question"] = record
            self.save()

        if self.mode == "agent":
            if self.answer is None:
                raise Pending(question, details, record=record)
            answer = self.answer
            if answer["question_id"] != record["question_id"]:
                used = any(
                    d["question_id"] == answer["question_id"]
                    for d in self.state.get("decisions", [])
                )
                raise Pending(
                    question,
                    details,
                    record=record,
                    error=(
                        "Answer has already been used"
                        if used
                        else "Answer does not match the current question; review the new snapshot"
                    ),
                )
            if answer["answer"] not in record["choices"]:
                raise Pending(
                    question,
                    details,
                    record=record,
                    error="Answer is not one of this question's choices",
                )
            if answer["answer"] == "defer":
                raise Pending(question, details, record=record)
            explanation = answer.get("reason", "").strip()
            if reason and not explanation:
                raise Pending(
                    question,
                    details,
                    record=record,
                    error="This approval requires a nonempty reason",
                )
        else:
            print(question, flush=True)
            print(details, flush=True)
            if not sys.stdin.isatty():
                raise Pending(question, details, record=record)
            try:
                choice = input("Continue? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    raise Pending(question, details, record=record)
                explanation = (
                    input("Reason for approving this change: ").strip()
                    if reason
                    else ""
                )
            except EOFError:
                raise Pending(question, details, record=record) from None
            if reason and not explanation:
                raise Pending(question, details, record=record)
            answer = {
                "question_id": record["question_id"],
                "answer": record["choices"][0],
                "reason": explanation,
            }

        # Persist consumption before proceeding, so a crash cannot replay an approval.
        self.state.setdefault("decisions", []).append(
            {
                **record,
                "answer": answer["answer"],
                "reason": explanation,
                "mode": self.mode,
            }
        )
        self.state.pop("pending_question", None)
        self.answer = None
        self.save()
        return explanation if reason else answer["answer"]
