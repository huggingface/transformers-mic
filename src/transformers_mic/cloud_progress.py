"""Best-effort live progress authenticated with short-lived GitHub Actions OIDC."""

import json
from datetime import datetime, timezone
import os
import sys
import time
import shutil
import re
import textwrap
from urllib.parse import urlencode
from urllib.request import Request, build_opener

from .dashboard_auth import NoRedirect
from . import terminal


class Progress:
    def __init__(self, job_id, profile):
        self.job_id, self.profile = job_id, profile
        self.url = os.getenv("MIC_PROGRESS_URL", "")
        self.sequence = 0
        self.token, self.token_time = None, 0
        self.warned = False
        self.retry_after = 0

    def emit(self, stage, completed, total, detail=""):
        self.sequence += 1
        event = {
            "job_id": self.job_id,
            "profile": self.profile,
            "stage": stage,
            "completed": completed,
            "total": total,
            "detail": detail[:200],
            "sequence": self.sequence,
        }
        print(json.dumps({"progress": event}), flush=True)
        if not self.url or time.monotonic() < self.retry_after:
            return
        try:
            if not self.token or time.monotonic() - self.token_time > 240:
                separator = (
                    "&" if "?" in os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] else "?"
                )
                url = (
                    os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
                    + separator
                    + urlencode({"audience": "serge-mic-baselines"})
                )
                request = Request(
                    url,
                    headers={
                        "Authorization": "Bearer "
                        + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
                    },
                )
                with build_opener(NoRedirect()).open(request, timeout=5) as response:
                    self.token = json.loads(response.read(16384))["value"]
                self.token_time = time.monotonic()
            request = Request(
                self.url,
                data=json.dumps(event).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.token,
                },
            )
            with build_opener(NoRedirect()).open(request, timeout=5) as response:
                response.read(1024)
        except (OSError, ValueError, KeyError):
            self.retry_after = time.monotonic() + 60
            # Measurement validity never depends on live telemetry availability.
            if not self.warned:
                print(
                    "Live progress relay unavailable; GitHub workflow progress is still visible.",
                    file=sys.stderr,
                )
                self.warned = True


def display(job, previous=None):
    """Refresh one live block in a terminal; emit only changed rows in logs."""
    view = {
        "status": job["status"],
        "run_url": job.get("run_url"),
        "jobs": job.get("workflow_jobs", []),
        "progress": [
            {
                key: event[key]
                for key in ("profile", "stage", "completed", "total", "detail")
            }
            for _, event in sorted(job.get("progress", {}).items())
        ],
    }
    old = json.loads(previous) if previous else {}
    old_rows = old.pop("_rows", 0)
    old_tick = old.pop("_tick", None)
    live = (
        terminal.interactive(sys.stderr) or os.getenv("TMIC_LIVE_PROGRESS") == "1"
    ) and os.getenv("TERM") != "dumb"
    tick = int(time.monotonic() / 10) if live else None
    if view == old and (not live or tick == old_tick):
        return previous
    width = max(40, shutil.get_terminal_size().columns - 1)

    def job_line(item):
        name = item["name"]
        profile = re.search(r"\b(cpu|cuda)\b", name, re.I)
        if profile:
            name = profile.group(1).upper()
        elif name == "validate":
            name = "Preflight"
        elif name == "publish":
            name = "Hub upload"
        status = item["status"]
        detail = item.get("step") or {
            "queued": "Waiting for a runner",
            "in_progress": "Running",
            "completed": "Finished",
            "failed": "Failed",
        }.get(status, status)
        total = item.get("total_steps", 0)
        if status == "queued" and live and item.get("queued_at"):
            try:
                since = datetime.fromisoformat(item["queued_at"].replace("Z", "+00:00"))
                minutes = max(
                    0, int((datetime.now(timezone.utc) - since).total_seconds() // 60)
                )
                detail += f" · queued {minutes}m"
            except ValueError:
                pass
        counter = (
            terminal.bar(item.get("completed_steps", 0), total, 10) + " steps · "
            if total
            else ""
        )
        return f"    {name}: {counter}{detail}"

    def event_line(event):
        return f"    [{event['profile']}] {event['stage']} {event['completed']}/{event['total']} {terminal.bar(event['completed'], event['total'], 10)} {event['detail']}"

    if live:
        lines = [
            "  Cloud · " + view["status"] + " · checked " + time.strftime("%H:%M:%S")
        ]
        if view.get("run_url"):
            lines.append("    Run: " + view["run_url"])
        lines += [job_line(item) for item in view["jobs"]]
        lines += [event_line(event) for event in view["progress"]]
        rendered = []
        for line in lines:
            if line.startswith("    Run:"):
                rendered.extend(
                    textwrap.wrap(
                        line,
                        width=width,
                        subsequent_indent="      ",
                        drop_whitespace=False,
                    )
                )
            else:
                rendered.append(line[:width])
        prefix = f"\033[{old_rows}A\r\033[J" if old_rows else ""
        sys.stderr.write(prefix + "\n".join(rendered) + "\n")
        sys.stderr.flush()
        view["_rows"] = len(rendered)
        view["_tick"] = tick
    else:
        if view.get("run_url") and old.get("run_url") != view["run_url"]:
            print("  Run: " + view["run_url"], file=sys.stderr, flush=True)
        if old.get("status") != view["status"]:
            print("[cloud] " + view["status"], file=sys.stderr, flush=True)
        old_jobs = {item["name"]: item for item in old.get("jobs", [])}
        for item in view["jobs"]:
            if old_jobs.get(item["name"]) != item:
                print(job_line(item), file=sys.stderr, flush=True)
        old_events = {event["profile"]: event for event in old.get("progress", [])}
        for event in view["progress"]:
            if old_events.get(event["profile"]) != event:
                print(event_line(event), file=sys.stderr, flush=True)
    return json.dumps(view, sort_keys=True)
