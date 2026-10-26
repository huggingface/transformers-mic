"""Concise human summaries; complete diagnostics stay in stage logs."""

import ast
import json
import os
import pydoc
import re
import shutil
import sys
import textwrap
import threading
import time

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


def interactive(stream=None):
    return bool((stream or sys.stdout).isatty()) and os.getenv("TERM") != "dumb"


def color(text, style="dim", stream=None):
    codes = {"dim": "2", "title": "1", "question": "1", "ok": "32", "error": "31"}
    if not interactive(stream) or "NO_COLOR" in os.environ:
        return text
    return f"\033[{codes[style]}m{text}\033[0m"


def short_path(path):
    value = os.fspath(path)
    relative = os.path.relpath(value)
    if not relative.startswith(".."):
        return relative
    home = os.path.expanduser("~")
    return (
        value.replace(home + os.sep, "~/", 1)
        if value.startswith(home + os.sep)
        else value
    )


def bar(done, total, width=14):
    if total <= 0:
        return ""
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def question_text(question, kind):
    labels = {
        "source-review": "Review this contract?",
        "cloud-baselines": "Run CPU + CUDA baselines in the cloud?",
        "contract-repair": "Apply the proposed repair and retry?",
        "model-publication": "Publish this model to the Hub?",
        "fleet-update": "Update the shared fleet?",
    }
    return labels.get(kind, question)


def prompt(question, details, kind, context):
    width = max(40, min(shutil.get_terminal_size().columns - 6, 100))
    print("\n  " + color("? " + question_text(question, kind), "question"), flush=True)
    if kind == "contract-repair" and context.get("diff"):
        lines = [
            details.splitlines()[0],
            "Diff: " + short_path(context["diff"]["path"]),
        ]
    elif kind == "cloud-baselines":
        lines = [
            "Model: " + str(context.get("repository", "")),
            "Stages an input snapshot; model main is published after review.",
        ]
    else:
        lines = details.splitlines()[:3]
    for line in lines:
        for wrapped in textwrap.wrap(line, width=width)[:3] or [""]:
            print("    " + color(wrapped), flush=True)
    if len(details.splitlines()) > 3 or kind in ("contract-repair", "cloud-baselines"):
        print("    " + color("D: full details"), flush=True)


def details(text):
    if interactive() and sys.stdin.isatty():
        pydoc.pager(text)
    else:
        print(text, flush=True)


class Activity:
    """One live line while a local stage writes its verbose output to a log."""

    def __init__(self, prefix, label, enabled=True, status=None):
        self.prefix, self.label = prefix, label
        self.status = status
        self.enabled = enabled and interactive()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if not self.enabled:
            return
        started = time.monotonic()

        def animate():
            frame = 0
            frames = "|/-\\"
            while not self.stop_event.is_set():
                elapsed = time.monotonic() - started
                detail = self.status() if self.status else ""
                active = self.label + (" · " + detail if detail else "")
                active = textwrap.shorten(
                    active,
                    width=max(
                        30, shutil.get_terminal_size().columns - len(self.prefix) - 15
                    ),
                    placeholder="…",
                )
                print(
                    f"\r\033[K{self.prefix} {active} {frames[frame % 4]} {elapsed:.0f}s",
                    end="",
                    flush=True,
                )
                frame += 1
                self.stop_event.wait(0.15)

        self.thread = threading.Thread(target=animate, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            print("\r\033[K", end="", flush=True)


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


def phase(path, offset=0):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(offset, handle.tell() - 8192))
            lines = handle.read().decode(errors="replace").splitlines()
        for line in reversed(lines):
            if line.startswith("MIC_PROGRESS "):
                value = json.loads(line.removeprefix("MIC_PROGRESS "))
                return value["phase"] + (
                    " · " + value["detail"] if value.get("detail") else ""
                )
    except (OSError, ValueError, KeyError):
        pass
    return ""
