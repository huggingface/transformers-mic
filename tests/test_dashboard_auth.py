import json
from types import SimpleNamespace

import pytest

from transformers_mic import dashboard_auth as auth
from transformers_mic.cli import main, parser


@pytest.fixture(autouse=True)
def private_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


def args(action="login", **extra):
    return SimpleNamespace(
        dashboard="https://dashboard.test",
        auth_command=action,
        no_browser=True,
        **extra,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://dashboard.test",
        "https://name:secret@dashboard.test",
        "https://dashboard.test/path",
        "https://dashboard.test?x=1",
        "//dashboard.test",
    ],
)
def test_credentials_require_an_https_origin(url):
    with pytest.raises(ValueError, match="HTTPS origin"):
        auth.dashboard_url(url)


def test_login_handles_pending_and_slow_down_saves_private_credential(
    monkeypatch, capsys
):
    calls, sleeps = [], []
    replies = iter(
        [
            auth.DashboardError("authorization_pending"),
            auth.DashboardError("slow_down"),
            {"access_token": "dashboard-secret", "expires_in": 28800},
        ]
    )

    def request(url, path, **kwargs):
        calls.append((url, path, kwargs))
        if path == "/auth/device":
            return {
                "verification_uri_complete": "https://dashboard.test/auth/cli?user_code=ABCDEF-123456",
                "user_code": "ABCDEF-123456",
                "device_code": "device-secret",
                "interval": 5,
                "expires_in": 900,
            }
        if path == "/auth/token":
            result = next(replies)
            if isinstance(result, Exception):
                raise result
            return result
        assert path == "/auth/me" and kwargs["token"] == "dashboard-secret"
        return {
            "id": 42,
            "login": "octocat",
            "scope": "profile:read mic:read mic:submit",
        }

    monkeypatch.setattr(auth, "request", request)
    monkeypatch.setattr(auth.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        auth.webbrowser,
        "open",
        lambda url: pytest.fail("--no-browser opened a browser"),
    )
    assert auth.run(args()) == 0
    assert sleeps == [5, 5, 10]
    path = auth.credential_path()
    assert path.stat().st_mode & 0o777 == 0o600
    assert auth.access_token("https://dashboard.test") == "dashboard-secret"
    assert "dashboard-secret" not in capsys.readouterr().out
    assert "device-secret" not in capsys.readouterr().out
    assert auth.run(args("status")) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["id"] == 42 and "access_token" not in status


def test_logout_revokes_only_the_selected_dashboard(monkeypatch):
    auth.save(
        {
            "https://dashboard.test": {"access_token": "secret"},
            "https://another.test": {"access_token": "other"},
        }
    )
    calls = []
    monkeypatch.setattr(auth, "request", lambda *a, **kw: calls.append((a, kw)))
    assert auth.run(args("logout")) == 0
    assert calls == [
        (("https://dashboard.test", "/auth/revoke"), {"body": {}, "token": "secret"})
    ]
    assert "https://dashboard.test" not in auth.credentials()
    assert "https://another.test" in auth.credentials()


def test_failed_revocation_retains_credential_to_retry(monkeypatch):
    auth.save({"https://dashboard.test": {"access_token": "secret"}})

    def fail(*a, **kw):
        raise auth.DashboardError("authentication_unavailable")

    monkeypatch.setattr(auth, "request", fail)
    with pytest.raises(auth.DashboardError):
        auth.run(args("logout"))
    assert "https://dashboard.test" in auth.credentials()


def test_expired_login_and_insecure_file_are_rejected():
    auth.save({"https://dashboard.test": {"access_token": "secret", "expires": 0}})
    with pytest.raises(ValueError, match="login required"):
        auth.access_token("https://dashboard.test")
    auth.credential_path().chmod(0o644)
    with pytest.raises(ValueError, match="must be private"):
        auth.credentials()


def test_credential_symlink_is_rejected(tmp_path):
    auth.save({})
    path = auth.credential_path()
    path.rename(path.with_suffix(".old"))
    path.symlink_to(path.with_suffix(".old"))
    with pytest.raises(ValueError, match="must be private"):
        auth.credentials()


def test_login_cannot_open_a_different_origin(monkeypatch):
    monkeypatch.setattr(
        auth,
        "request",
        lambda *a, **kw: {"verification_uri_complete": "https://evil.test"},
    )
    with pytest.raises(ValueError, match="unexpected authorization URL"):
        auth.run(args())


def test_cli_auth_does_not_open_a_mic_workspace(monkeypatch, capsys):
    parsed = parser().parse_args(
        ["auth", "login", "--dashboard", "https://dashboard.test", "--no-browser"]
    )
    assert parsed.no_browser and parsed.auth_command == "login"
    assert main(["auth", "status", "--dashboard", "https://dashboard.test"]) == 2
    assert "login required" in capsys.readouterr().err


def test_no_redirect_can_forward_a_credential():
    with pytest.raises(auth.DashboardError, match="redirected"):
        auth.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.test")
