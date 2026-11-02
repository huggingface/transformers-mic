# Releasing `transformers-mic`

Releases are tag-driven, as in `transformers-mlinter`. A branch named `vX.Y-release` runs the
[`Release`](../.github/workflows/release.yml) workflow's build and test job. A pushed tag `vX.Y.Z` (or `vX.Y.ZrcN`)
runs the same job, then publishes the built wheel and sdist to PyPI through GitHub OIDC trusted publishing. Nobody
uploads from a laptop, and no PyPI token is stored in GitHub.

No `.dev0` suffix is used: a release is one version bump, a validation pass, a release-branch push and a tag push,
followed by a post-release bump that puts `main` on the next plain version.

## Prerequisites

- PyPI maintainer access for `transformers-mic`.
- A PyPI trusted publisher for this repository, workflow `release.yml`, environment `pypi-release`.
- A GitHub environment named `pypi-release` on this repository, ideally with required reviewers.
- A clean checkout of `main` and a Python 3.11 venv with `pip install -e ".[runtime,dev]"`.

### First release

`transformers-mic` did not exist on PyPI before 0.1.0. Before pushing the first tag, add a **pending** trusted
publisher in your PyPI account's publishing settings (not a project page): owner `huggingface`, repository
`transformers-mic`, workflow `release.yml`, environment `pypi-release`. The first successful publish creates the
project and turns the pending publisher into a normal one.

## The version

`pyproject.toml` `[project].version` is the only place the version is written.
[`_version.py`](../src/transformers_mic/_version.py) reads it from the installed distribution, falling back to
`pyproject.toml`, and adds `+g<commit>` for a Git checkout or a Git install. Its `DEFAULT_BASE_VERSION` (`0.0.0`)
is a last-resort sentinel and is never bumped.

## 1. Update the changelog and version

Add a `## [X.Y.Z] - YYYY-MM-DD` section to [`CHANGELOG.md`](../CHANGELOG.md) from
`git log vPREV..HEAD --oneline`, then set `[project].version` in `pyproject.toml`.

## 2. Test, build and check

```bash
make test
make build-release
python -m twine check --strict dist/*
```

Then install the wheel in a fresh venv and check the CLI from outside the checkout:

```bash
python3.11 -m venv /tmp/tmic-release
/tmp/tmic-release/bin/pip install dist/*.whl
cd /tmp
/tmp/tmic-release/bin/tmic --version             # X.Y.Z, no +g suffix
/tmp/tmic-release/bin/python -m transformers_mic --version
/tmp/tmic-release/bin/tmic run --help > /dev/null
```

The workflow runs the same checks, and also asserts that the wheel bundles the runner (`_backend/run.py`) and the
contract catalog (`_catalog/hub`).

## 3. Commit, then push the release branch

Land the changelog and version bump on `main` through a PR, then create the release branch from it. For a new minor
line:

```bash
git switch main && git pull --ff-only
git switch -c vX.Y-release
git push -u origin vX.Y-release
```

For a patch on an existing line:

```bash
git switch vX.Y-release && git merge --ff-only main && git push origin vX.Y-release
```

The branch push runs the build and test job only.

## 4. Tag and publish

The tag push publishes to PyPI and cannot be undone: a version number can never be uploaded twice.

```bash
git tag vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
gh run watch --repo huggingface/transformers-mic
```

The `upload_package` job runs in the `pypi-release` environment, which may wait for a reviewer's approval. Then:

```bash
python -m pip install -U "transformers-mic==X.Y.Z"
tmic --version
```

## 5. Open the next version

Through a PR on `main`, set `[project].version` to the next patch `X.Y.(Z+1)` and add an empty `## [Unreleased]`
heading above the released section of `CHANGELOG.md`. Commit as `Starting version NEXT`. This commit is not
tagged and not merged into `vX.Y-release`.
