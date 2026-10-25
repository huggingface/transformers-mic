"""Best-effort live progress authenticated with short-lived GitHub Actions OIDC."""

import json
import os
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, build_opener

from .dashboard_auth import NoRedirect


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
    """Human progress on stderr; emit only changes, without erasing CI log history."""
    view = {
        "status": job["status"],
        "jobs": job.get("workflow_jobs", []),
        "progress": [
            {
                key: event[key]
                for key in ("profile", "stage", "completed", "total", "detail")
            }
            for _, event in sorted(job.get("progress", {}).items())
        ],
    }
    signature = json.dumps(view, sort_keys=True)
    if signature == previous:
        return signature
    print("[cloud] " + job["status"], file=sys.stderr, flush=True)
    for item in view["jobs"]:
        if item["status"] != "completed":
            print(
                f"  {item['name']}: {item.get('step') or item['status']}",
                file=sys.stderr,
                flush=True,
            )
    for event in view["progress"]:
        print(
            f"  [{event['profile']}] {event['stage']} {event['completed']}/{event['total']} {event['detail']}",
            file=sys.stderr,
            flush=True,
        )
    return signature
