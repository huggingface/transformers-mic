---
name: release
description: Cut a transformers-mic (tmic) release to PyPI. Updates CHANGELOG.md and the single pyproject version, runs the tests and a packaged smoke test, pushes the vX.Y-release branch and, after confirmation, the vX.Y.Z tag that publishes through GitHub OIDC, then opens the next version on main. Use when asked to release tmic, cut a version, or ship X.Y.Z.
---

# Release transformers-mic

The procedure and its reasons are in `docs/release.md`; this is the checklist. CI publishes: pushing the `vX.Y.Z`
tag runs `.github/workflows/release.yml`, which builds, tests and uploads to PyPI through trusted publishing.

## Input

- `<version>`: `X.Y.Z`, or `X.Y.ZrcN` for a pre-release.

## Rules

- Start from a clean, up-to-date `main`. Never release uncommitted changes.
- Never run `twine upload`. Publishing is CI-only, from the tag.
- **Ask the user before pushing the tag.** It publishes, and a PyPI version can never be reused.
- `pyproject.toml` `[project].version` is the only version string. Do not add another, and do not bump
  `DEFAULT_BASE_VERSION` in `src/transformers_mic/_version.py`.
- Land changes on `main` through PRs, not direct pushes.

## Steps

1. **Preflight.** `git fetch && git status` on `main`; `<version>` must be above the latest tag (`git tag`).
   For a first release, confirm with the user that the PyPI pending publisher and the `pypi-release` GitHub
   environment exist (`gh api repos/huggingface/transformers-mic/environments`).
2. **Changelog.** Add `## [X.Y.Z] - YYYY-MM-DD` (today) above the previous release in `CHANGELOG.md`, from
   `git log vPREV..HEAD --oneline`. Use `### Added` / `### Changed` / `### Fixed`; one line per user-visible change,
   linking PRs where useful. Fold an existing `## [Unreleased]` section into it.
3. **Version.** Set `[project].version` in `pyproject.toml` to `<version>`.
4. **Test and build** in a Python 3.11 venv with `pip install -e ".[runtime,dev]"`:
   ```bash
   make test
   make build-release
   python -m twine check --strict dist/*
   ```
5. **Smoke-test the wheel** from outside the checkout:
   ```bash
   python3.11 -m venv /tmp/tmic-release && /tmp/tmic-release/bin/pip install dist/*.whl
   cd /tmp && /tmp/tmic-release/bin/tmic --version && /tmp/tmic-release/bin/tmic run --help > /dev/null
   ```
   `tmic --version` must print exactly `<version>`.
6. **Land the bump** on `main`: branch `release-X.Y.Z`, commit `Release X.Y.Z`, PR, squash-merge.
7. **Release branch.** New minor: `git switch main && git pull --ff-only && git switch -c vX.Y-release && git push
   -u origin vX.Y-release`. Patch: fast-forward the existing branch from `main` and push. Wait for the `Release`
   run on the branch to pass (`gh run list --repo huggingface/transformers-mic --workflow release.yml`).
8. **Tag**, after the user confirms, from `vX.Y-release`:
   ```bash
   git tag vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z
   gh run watch --repo huggingface/transformers-mic
   ```
   The publish job may wait for a reviewer on the `pypi-release` environment; say so instead of waiting silently.
9. **Verify** in a fresh venv: `pip install "transformers-mic==X.Y.Z"` then `tmic --version`.
10. **Next version.** PR on `main`: `[project].version` to `X.Y.(Z+1)` and an empty `## [Unreleased]` heading in
    `CHANGELOG.md`, commit `Starting version NEXT`. Not tagged, not merged into the release branch.
