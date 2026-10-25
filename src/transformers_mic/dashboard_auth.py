"""Authenticate tmic to the dashboard without GitHub or Grafana credentials."""

import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import webbrowser

DEFAULT_DASHBOARD = "https://transformers-ci.huggingface.cool"


class DashboardError(ValueError):
    def __init__(self, error):
        self.error = error
        super().__init__(error)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DashboardError(
            "Dashboard redirected an authentication request; check --dashboard"
        )


def dashboard_url(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "--dashboard must be an HTTPS origin, e.g. " + DEFAULT_DASHBOARD
        )
    return value.rstrip("/")


def request(dashboard, path, *, body=None, token=None, binary=False):
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = Request(
        dashboard + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
    )
    try:
        with build_opener(NoRedirect()).open(req, timeout=20) as response:
            limit = 256_000_000 if binary else 65536
            payload = response.read(limit + 1)
            if len(payload) > limit:
                raise DashboardError("response_too_large")
            return payload if binary else json.loads(payload)
    except HTTPError as error:
        try:
            reason = json.loads(error.read(4096)).get("error", "request_failed")
        except (ValueError, AttributeError):
            reason = "request_failed"
        # Only print protocol error names, never a response carrying credentials.
        if not isinstance(reason, str) or not reason.replace("_", "").isalnum():
            reason = "request_failed"
        raise DashboardError(reason) from None
    except (URLError, TimeoutError):
        raise DashboardError(
            "Could not reach the dashboard; check the URL and connection"
        ) from None


def request_bytes(dashboard, path, *, token):
    return request(dashboard, path, token=token, binary=True)


def credential_path():
    base = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "transformers-mic" / "dashboard-auth.json"


def credentials():
    path = credential_path()
    if not path.exists():
        return {}
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError(f"Credential file must be private (chmod 600): {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Invalid dashboard credential file")
    return data


def save(data):
    path = credential_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".dashboard-auth-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def access_token(dashboard):
    record = credentials().get(dashboard, {})
    if not record or record.get("expires", 0) <= time.time():
        raise ValueError("Dashboard login required; run tmic auth login")
    return record["access_token"]


def run(args):
    dashboard = dashboard_url(args.dashboard)
    if args.auth_command == "login":
        grant = request(dashboard, "/auth/device", body={"client_id": "tmic"})
        # The selected HTTPS origin is the only place we open or send credentials.
        verification = grant["verification_uri_complete"]
        if (
            urlsplit(verification).scheme != "https"
            or urlsplit(verification).netloc != urlsplit(dashboard).netloc
        ):
            raise ValueError("Dashboard returned an unexpected authorization URL")
        print(f"Open {verification}\nConfirm code: {grant['user_code']}", flush=True)
        if not args.no_browser:
            webbrowser.open(verification)
        interval = max(5, int(grant["interval"]))
        deadline = time.monotonic() + min(int(grant["expires_in"]), 900)
        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                result = request(
                    dashboard,
                    "/auth/token",
                    body={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": grant["device_code"],
                    },
                )
            except DashboardError as error:
                if error.error == "authorization_pending":
                    continue
                if error.error == "slow_down":
                    interval += 5
                    continue
                raise
            token = result["access_token"]
            try:
                user = request(dashboard, "/auth/me", token=token)
                data = credentials()
                previous = data.get(dashboard, {}).get("access_token")
                data[dashboard] = {
                    "access_token": token,
                    "expires": time.time() + int(result["expires_in"]),
                    "login": user["login"],
                }
                save(data)
            except (OSError, ValueError):
                request(dashboard, "/auth/revoke", body={}, token=token)
                raise
            if previous:
                try:
                    request(dashboard, "/auth/revoke", body={}, token=previous)
                except DashboardError:
                    pass  # expired or already revoked; the new session is saved
            print(
                f"Signed in as @{user['login']} on {dashboard}. Access expires in eight hours."
            )
            return 0
        raise ValueError("Authorization expired; run tmic auth login again")
    if args.auth_command == "status":
        user = request(dashboard, "/auth/me", token=access_token(dashboard))
        print(
            json.dumps(
                {
                    "dashboard": dashboard,
                    "id": user["id"],
                    "login": user["login"],
                    "scope": user["scope"],
                },
                indent=2,
            )
        )
        return 0
    data = credentials()
    record = data.get(dashboard)
    if record:
        try:
            request(dashboard, "/auth/revoke", body={}, token=record["access_token"])
        except DashboardError as error:
            if error.error != "unauthorized":
                raise  # retain the credential so logout can retry server revocation
        data.pop(dashboard)
        save(data)
    print(f"Signed out of {dashboard}.")
    return 0
