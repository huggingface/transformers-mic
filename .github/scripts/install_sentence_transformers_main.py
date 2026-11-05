"""Resolve both upstream mains afresh, install them, and verify their commit IDs."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

PROJECTS = {
    "transformers": "https://github.com/huggingface/transformers.git",
    "sentence-transformers": "https://github.com/huggingface/sentence-transformers.git",
}


def resolve_heads():
    heads = {}
    for package, url in PROJECTS.items():
        result = subprocess.run(
            ["git", "ls-remote", url, "refs/heads/main"],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = result.stdout.split()
        if (
            len(fields) != 2
            or fields[1] != "refs/heads/main"
            or not re.fullmatch(r"[0-9a-f]{40}", fields[0])
        ):
            raise ValueError(f"Could not resolve {package} main to one commit")
        heads[package] = {"url": url, "commit": fields[0]}
    return heads


def verify_installed(heads):
    for package, expected in heads.items():
        distribution = importlib.metadata.distribution(package)
        provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
        if (
            provenance.get("url") != expected["url"]
            or provenance.get("vcs_info", {}).get("commit_id") != expected["commit"]
        ):
            raise ValueError(
                f"Installed {package} is not the resolved main commit {expected['commit']}"
            )


def install(out):
    heads = resolve_heads()
    out.mkdir(parents=True, exist_ok=True)
    (out / "upstream-commits.json").write_text(json.dumps(heads, indent=2) + "\n")
    requirements = [
        f"{package} @ git+{pin['url']}@{pin['commit']}"
        for package, pin in heads.items()
    ]
    # Explicit immutable refs avoid a cached/image copy of the same dev version.
    # Upgrade dependencies only when required; retain the image's torch stack.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "--upgrade-strategy",
            "only-if-needed",
            *requirements,
        ],
        check=True,
    )
    verify_installed(heads)
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (out / "pip-freeze.txt").write_text(frozen.stdout)
    print(json.dumps({"verified_upstream_commits": heads}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/sentence-transformers"))
    install(parser.parse_args().out)


if __name__ == "__main__":
    main()
