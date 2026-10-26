from transformers_mic.terminal import failure


def test_gate_failure_shows_cause_threshold_and_log_instead_of_large_dictionaries():
    text = "COMMAND [build]\nW1025 warning\nrandom weights fail the gate: {'random': {'pass': False}, 'overfit': {'weakest': 'pixel_values:halve', 'threshold': 0.001, 'deltas': {'pixel_values:halve': 0.000001}}}"
    summary = failure(text)
    assert "not sensitive enough" in summary
    assert "pixel_values:halve" in summary and "required 0.001" in summary
    assert "'deltas'" not in summary and "COMMAND" not in summary


def test_traceback_shows_exception_without_stack_frames():
    text = 'Traceback (most recent call last):\n  File "usage.md", line 53\n    named[bad]\nKeyError: missing.parameter'
    assert failure(text) == "KeyError: missing.parameter"
