"""Does this Python match the cloud runners before the local factory check runs?

`tmic run` checks model factories and every variant locally, then measures in the
cloud. The cloud runners have the `runtime` extra's packages (Transformers' own
`testing` extra installs ``transformers-mic[runtime]``) and a Transformers commit
from ``main``, resolved at dispatch. A local check in an environment without a
package, or on a Transformers release, fails or passes for reasons the cloud does
not share. This module names the difference and the pip command that removes it.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement

from . import resources

REPOSITORY = "https://github.com/huggingface/transformers"
ALLOW_DRIFT = "TMIC_ALLOW_ENVIRONMENT_DRIFT"


def runtime_requirements():
    """The `runtime` extra, from this checkout's pyproject.toml or the installed metadata.

    An editable install keeps the metadata of the day it was installed, so a source
    checkout reads its pyproject.toml: an extra added since is still checked.
    """
    pyproject = resources.CHECKOUT / "pyproject.toml"
    if pyproject.is_file():
        project = tomllib.loads(pyproject.read_text()).get("project", {})
        if project.get("name") == "transformers-mic":
            return [
                Requirement(text)
                for text in project["optional-dependencies"]["runtime"]
            ]
    requirements = [
        Requirement(text)
        for text in importlib.metadata.requires("transformers-mic") or []
    ]
    return [
        requirement
        for requirement in requirements
        if requirement.marker and requirement.marker.evaluate({"extra": "runtime"})
    ]


def missing(requirements, version=importlib.metadata.version):
    """Requirements that are not installed, or installed at a version they exclude."""
    result = []
    for requirement in requirements:
        if requirement.marker and not requirement.marker.evaluate({"extra": "runtime"}):
            continue
        try:
            installed = version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            result.append((requirement, None))
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            result.append((requirement, installed))
    return result


def transformers_source(distribution=None):
    """(version, commit, uncommitted source path) of the installed Transformers.

    The commit comes from pip's ``direct_url.json``: a ``git+`` install records it,
    and a local source tree (``pip install /path``, ``-e /path``) is asked with git.
    A release from PyPI has neither, so its commit is None; so is Transformers that
    is not installed at all (version None).
    """
    if distribution is None:
        try:
            distribution = importlib.metadata.distribution("transformers")
        except importlib.metadata.PackageNotFoundError:
            return None, None, None
    text = distribution.read_text("direct_url.json")
    if not text:
        return distribution.version, None, None
    origin = json.loads(text)
    commit = origin.get("vcs_info", {}).get("commit_id")
    if commit:
        return distribution.version, commit, None
    url = urlparse(origin.get("url", ""))
    if url.scheme != "file":
        return distribution.version, None, None
    path = Path(unquote(url.path))

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(path), *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD")
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return distribution.version, None, None
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return distribution.version, commit, path if dirty else None


def main_commit():
    """The commit Transformers main resolves to now, or None when it cannot be reached."""
    try:
        out = subprocess.run(
            ["git", "ls-remote", REPOSITORY, "refs/heads/main"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0] if out and re.fullmatch(r"[0-9a-f]{40}", out[0]) else None


def compare(requirements, source, head, version=importlib.metadata.version):
    """(problems, notes, install): what stops the check, what only warns, and the fix."""
    problems, notes, install = [], [], []
    framework = f"transformers @ git+{REPOSITORY}@{head or 'main'}"
    absent = [
        (requirement, installed)
        for requirement, installed in missing(requirements, version)
        if requirement.name != "transformers"
    ]
    if absent:
        problems.append(
            "The cloud runners install the transformers-mic `runtime` extra; this "
            "Python lacks "
            + ", ".join(
                str(requirement) + (f" (installed {installed})" if installed else "")
                for requirement, installed in absent
            )
            + "."
        )
        for requirement, _ in absent:
            requirement = Requirement(str(requirement))
            requirement.marker = None
            install.append(str(requirement))
    transformers, commit, dirty = source
    measured = f"huggingface/transformers main ({head or 'not resolved'})"
    if transformers is None:
        problems.append(
            f"Transformers is not installed; the cloud runners measure {measured}."
        )
        install.append(framework)
    elif commit is None:
        problems.append(
            f"Transformers {transformers} is a release, or an install with no commit "
            f"to trace; the cloud runners measure {measured}."
        )
        install.append(framework)
    elif head is None:
        notes.append(
            f"Could not resolve Transformers main; Transformers here is {commit}."
        )
    elif commit != head:
        notes.append(
            f"Transformers here is {commit[:12]}, main is {head[:12]}; the cloud "
            "measures main at dispatch. To match: python -m pip install "
            + shlex.quote(framework)
        )
    if dirty:
        notes.append(
            f"Transformers at {dirty} has uncommitted changes the cloud runners do not have."
        )
    return problems, notes, install


def require_cloud_match():
    """Stop before the local factory check when this Python differs from the cloud's."""
    problems, notes, install = compare(
        runtime_requirements(), transformers_source(), main_commit()
    )
    for note in notes:
        print(f"tmic: note: {note}", file=sys.stderr, flush=True)
    if not problems:
        return
    message = "\n".join(
        [
            *problems,
            f"Python: {sys.executable}",
            "Fix: python -m pip install " + " ".join(map(shlex.quote, install)),
        ]
    )
    if os.environ.get(ALLOW_DRIFT) == "1":
        print(
            f"tmic: warning ({ALLOW_DRIFT}=1, continuing): {message}",
            file=sys.stderr,
            flush=True,
        )
        return
    raise ValueError(message)
