"""Concise human summaries; complete diagnostics stay in stage logs."""

import ast
import json
import re
import textwrap

STAGE_NAMES = {
    "create": "Prepare workspace",
    "source": "Review contract",
    "build": "Build fixture",
    "auto": "Check model factories",
    "record": "Measure baselines",
    "review": "Review baseline changes",
    "check": "Validate baselines",
    "publish-preview": "Preview model upload",
    "publish": "Publish model",
    "published-check": "Verify published snapshot",
    "export-preview": "Preview fleet update",
    "export": "Update fleet",
    "pin": "Pin fleet revision",
}


def failure(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("COMMAND ")
    ]
    for line in reversed(lines):
        if "random weights fail the gate" in line:
            summary = (
                "Fixture outputs are not sensitive enough to changes in the inputs."
            )
            start = line.find("{'random':")
            if start >= 0:
                try:
                    gates = ast.literal_eval(line[start:])
                    gate = gates.get("overfit", gates["random"])
                    weakest = gate.get("weakest")
                    delta = gate.get("deltas", {}).get(weakest)
                    if weakest and isinstance(delta, (int, float)):
                        summary += f" Weakest: {weakest} ({delta:.3g}; required {gate['threshold']:.3g})."
                except (ValueError, SyntaxError, KeyError):
                    pass
            return summary
        if re.match(r"(?:[\w.]+(?:Error|Exception)|SystemExit):", line):
            return textwrap.shorten(line, width=240, placeholder="…")
        if line.startswith("{"):
            try:
                value = json.loads(line)
                if value.get("error"):
                    return textwrap.shorten(
                        str(value["error"]), width=240, placeholder="…"
                    )
            except (ValueError, AttributeError):
                pass
    meaningful = [line for line in lines if not re.match(r"[WEI]\d{4}\s", line)]
    return textwrap.shorten(
        meaningful[-1] if meaningful else "Stage exited unsuccessfully.",
        width=240,
        placeholder="…",
    )
