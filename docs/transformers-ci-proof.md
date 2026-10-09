# Plugging the fleet into Transformers CI: proof plan and to-do

Goal: show that the contracts run as a non-blocking job in Transformers' own PR CI, with selection, a pinned environment that matches the baselines, and a failure report a developer can act on. Then use the result to ask maintainers whether to adopt it.

Status on 2026-10-09: everything runs in this repository (shadow, nightly, backfill); contracts and fixtures are on the Hub, pinned by revision and hash; nothing runs inside Transformers CI yet. The broader list is in `docs/next-steps.md` (local, not in the repository).

## Facts the plan rests on (checked 2026-10-09)

- Transformers PR CI is [`pr-ci-caller.yml`](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml), which calls `huggingface/transformers-ci/.github/workflows/pr-ci_dynamic_caller_example.yml@main`: `tests_fetcher` picks test lists, then one reusable job per suite (`pr-ci_reusable_test_job.yml`).
- Torch test jobs run in the `huggingface/transformers-torch-light` image on runner group `aws-m8i-l-cache` and install the PR with `uv pip install -e .`.
- m8i instances are Intel; our `cpu-x86_64` baselines were recorded on AMD EPYC (7R32, matched by 7763). bfloat16 already differed on a CPU with native bfloat16 (EPYC 9V45). Intel float32 agreement is untested.
- This repository has access to `aws-g5-4xlarge-cache` only (infra, 2026-10-06), not `aws-m8i-l-cache`.
- The runner needs doc-builder's runnable-markdown collector, which is on doc-builder `main` only (PyPI `hf-doc-builder` is 0.5.0).
- The runner today: `example/run.py` + `select_contracts.py` + `patches.py` + `schemas/`, reading `example/transformers/tests/contracts/fleet.lock` and `expectations/`. `local_patches` currently holds three workarounds (#49385, #49403, #49456).

## Decisions to take first

- [ ] **D1. Where the proof runs.** Options: (a) a branch on `huggingface/transformers` with a separate `.github/workflows/contracts.yml` (same-repo branch, so self-hosted runners are available; check `pr-ci-security-gate.yml`), opened as a draft PR; (b) a branch of `huggingface/transformers-ci` adding a `tests_contracts` job to the dynamic caller; (c) a fork with GitHub-hosted runners only (no m8i, weaker proof). Recommended: (a) for the proof, (b) once accepted.
- [ ] **D2. Environment.** Install the fleet lock's pins in the job, or record baselines in the CI image. Decided by phase 1's measurements.
- [ ] **D3. Runner packaging.** Vendor into `utils/contracts/` for the proof; a published package (the design's "shared runner, to be named") later.
- [ ] **D4. Workarounds in Transformers.** `local_patches` patch the framework under test: fine for an outside observer, not for the framework's own CI. Proposed: no patches in Transformers; open bugs become `known_failures` entries (xfail with the issue). Check what that hides (e.g. #49456 would xfail all Whisper targets).
- [ ] **D5. doc-builder.** Ask for a release with the collector, or pin `git+https://github.com/huggingface/doc-builder@<sha>` in the job.

## Phase 1: validate the environment on the PR CI runner (this repository)

Question: do our `cpu-x86_64` fixture baselines hold on `aws-m8i-l-cache` in the CI image, with and without the lock's pins?

- [ ] Ask infra for this repository's access to `aws-m8i-l-cache` (or run phase 1 from the Transformers branch in D1a).
- [ ] New workflow `ci-image-probe.yml`: container `huggingface/transformers-torch-light`, runner `aws-m8i-l-cache`; install Transformers at the pinned build (`8625785e`), then run `run.py --fixtures-only --require-all` twice:
  - [ ] **as is**: image packages, `environment_differs` recorded;
  - [ ] **pinned**: `.github/scripts/install_pinned.sh` first.
- [ ] Record CPU model and flags (`avx512_bf16`, `amx_*`), package versions, timings (install, run).
- [ ] Decide: if *as is* passes, the job needs no pin install; if only *pinned* passes, the job installs pins (measure cost); if neither, record an Intel class (`cpu-x86_64` vs a new platform) and decide which runner records required baselines.
- [ ] Note the result in [evidence](evidence.md#stability-and-cost).

## Phase 2: package the fleet for Transformers

- [ ] Runner changes in this repository first, tested here:
  - [ ] `--lock PATH` (default stays `example/transformers/tests/contracts/fleet.lock`); expectations next to the lock.
  - [ ] Make the lock's `path` optional: published contracts and fixtures need no source folder (only `--local-*` and script contracts do).
  - [ ] `patches` import only when `local_patches` is non-empty.
  - [ ] Keep `schemas/` beside the runner.
- [ ] Transformers branch layout:
  - [ ] `tests/contracts/fleet.lock` (no `path`, no `local_patches`; `known_failures` per D4), `tests/contracts/expectations/` (x86 platform(s) from phase 1; drop `cpu-arm64` and `cuda` unless needed).
  - [ ] `utils/contracts/run.py`, `select_contracts.py`, `schemas/`, `report.py` (phase 3), with a header naming this repository as the source.
- [ ] Local check on the branch: `python utils/contracts/run.py --framework transformers --fixtures-only --lock tests/contracts/fleet.lock` passes on the branch's own build.

## Phase 3: the CI job (non-blocking)

- [ ] `.github/workflows/contracts.yml` on `pull_request` (per D1):
  - [ ] container and runner from phase 1; checkout with enough history for `git diff origin/main...HEAD`;
  - [ ] install: `uv pip install -e .`, runner deps (jsonschema, doc-builder per D5), pins per D2;
  - [ ] cache `~/.cache/huggingface/hub` keyed on the lock hash (fixtures and contracts are ~150 MB);
  - [ ] run: `git diff --name-only origin/main...HEAD | python utils/contracts/run.py --framework transformers --fixtures-only --changed-files - --require-all --output results.jsonl` (selection restricts the run: first real use of `select_contracts.py` without `--run-all`);
  - [ ] `continue-on-error: true`; upload `results.jsonl`.
- [ ] `utils/contracts/report.py`: results to a job summary (`$GITHUB_STEP_SUMMARY`):
  - [ ] per target: contract, fixture, status, first failure line, baseline path;
  - [ ] repro command: `python utils/contracts/run.py --framework transformers --fixtures-only --contract <C> --target <T>`;
  - [ ] selection reason (`selected_by`) and skipped contracts;
  - [ ] for `fail`/`error`: what to do (regression vs intended change; link to the baseline update steps).
- [ ] Budget: under 2 minutes end to end on a warm cache (runner time was ~13 s for 8 fixtures on a 2-core runner).

## Phase 4: demonstrate on real PRs (draft PRs against the D1 branch)

- [ ] **A. Clean:** an unrelated change (docs or a model outside the fleet) runs no contract or all-pass, with a readable summary.
- [ ] **B. Shared code:** a no-op change in `generation/utils.py` selects every contract; all pass; record time.
- [ ] **C. Model-scoped:** a change in `models/clip/` selects only CLIP.
- [ ] **D. Known regression:** reintroduce #49135 (revert `d7ba937671`, UMT5 non-causal decoder under SDPA): the T5 contract fails on `proto-umt5-small-slim` (`label_logprobs`, loss); the summary alone is enough to reproduce locally.
- [ ] **E. Live regression:** the #49456 Whisper failure on `main` shows up as `xfail` (per D4) or as a failure, whichever D4 decides; check the report makes the cause obvious.
- [ ] Record per PR: selection, run time, install time, report link. Add a "Transformers CI proof" section to [evidence](evidence.md).

## Phase 5: write-up and maintainer decision

- [ ] One-page summary for Transformers maintainers: what ran, cost, what it caught, what a developer sees, what upkeep it adds (baseline updates, fixtures, contract changes).
- [ ] Ask for: runtime and noise budget, owners per contract, whether to move the job into `transformers-ci` (D1b), and the policy for D4.

## After the proof (not needed to show it plugs in)

- [ ] **Intended changes:** a PR-triggered job that records candidate baselines on every required platform and posts them for review (`review_candidates.py` in CI), replacing today's three manual steps.
- [ ] **`run-slow`:** map `run-slow: <family>` to contracts and run their full checkpoints on GPU through `self-comment-ci.yml`.
- [ ] **After merge and nightly** inside Transformers; failure alerts (an issue or Slack message per failing run). The nightly here has no alert yet: add one regardless.
- [ ] **Contract and framework changing together:** candidate contract revision tested with the PR, then a lock bump.
- [ ] **Model updates:** test a new checkpoint revision, check it still shares its fixture, propose pin and baseline updates.
- [ ] **Coverage:** more contracts (one MoE or coupled-config model, one GPU-heavy), map each `run-slow` test a contract replaces.
- [ ] **Isolation and private artifacts** before contracts from outside Hugging Face.
- [ ] Choose runners from `resources` instead of fixed jobs (next action 8).

## Open items from the current session (finish first)

- [ ] `cpu-x86_64` and `cuda` Whisper baselines for `max_length_tokens`, `assisted_tokens`, `longform_tokens`: runs 37897926458 (`cpu-baselines.yml machine=gpu-host`) and 37897930698 (`nightly.yml mode=record`), queued on `aws-g5-4xlarge-cache` since 2026-10-09 ~09:00. Download candidates, `python example/review_candidates.py --platform cpu-x86_64` and `--platform cuda`, approve if unchanged + added only, push before the 02:41 UTC nightly.
- [ ] Re-run the five-month backfill after those baselines land (`shadow-backfill.yml since=2026-05-01 until=2026-10-08`) and record the second round's detections in [improving contracts](improving-contracts.md).
- [ ] Track #49456 (Whisper `generate()` since #48282): drop `config-field-generation-params` once fixed. #49419 fixed #49403: drop `whisper-decode-tensor-batch` at the next pinned-build bump, together with #49386 (Wav2Vec2) when it lands.
