# Plugging the fleet into Transformers CI: proof plan and to-do

Goal: run the contracts as a non-blocking job in Transformers' own PR CI, with selection, a pinned environment matching the baselines, and an actionable failure report; then ask maintainers whether to adopt it.

Status (2026-10-09): shadow, nightly, and backfill run in this repository; contracts and fixtures are on the Hub, pinned by revision and hash; nothing runs inside Transformers CI yet.

## Facts the plan rests on (checked 2026-10-09)

- PR CI is [`pr-ci-caller.yml`](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml), calling `huggingface/transformers-ci/.github/workflows/pr-ci_dynamic_caller_example.yml@main`: `tests_fetcher` picks test lists, then one reusable job per suite (`pr-ci_reusable_test_job.yml`).
- Torch test jobs run in `huggingface/transformers-torch-light` on runner group `aws-m8i-l-cache` and install the PR with `uv pip install -e .`.
- m8i is Intel; our `cpu-x86_64` baselines come from AMD EPYC (7R32, matched by 7763). bfloat16 already differs on native-bfloat16 AMD (9V45, hence `cpu-x86_64-bf16`). Intel agreement is untested, float32 included.
- This repository can use `aws-g5-4xlarge-cache` only, not `aws-m8i-l-cache`. Access was removed on 2026-10-09 ~11:40 UTC; infra (Guillaume Legendre) is re-adding it.
- The runner needs doc-builder's runnable-markdown collector, on doc-builder `main` only (PyPI `hf-doc-builder` is 0.5.0).
- Runner today: `example/run.py`, `select_contracts.py`, `patches.py`, `schemas/`, reading `example/transformers/tests/contracts/fleet.lock` and `expectations/`. `local_patches` holds three workarounds (#49385, #49403, #49456).

## Decisions to take first

- [ ] **D1. Where the proof runs.** (a) a same-repo branch on `huggingface/transformers` with its own `.github/workflows/contracts.yml` (self-hosted runners available; check `pr-ci-security-gate.yml`), as a draft PR; (b) a `huggingface/transformers-ci` branch adding a `tests_contracts` job to the dynamic caller; (c) a fork on GitHub-hosted runners (no m8i, weaker proof). Recommended: (a) now, (b) once accepted.
- [ ] **D2. Environment.** Install the lock's pins in the job, or record baselines in the CI image. Decided by phase 1.
- [ ] **D3. Runner packaging.** Vendor into `utils/contracts/` for the proof; a published package (the design's "shared runner") later.
- [ ] **D4. Workarounds.** `local_patches` patch the framework under test: fine for an outside observer, not for its own CI. Proposed: no patches in Transformers; open bugs become `known_failures` (xfail with the issue). Check what that hides (#49456 would xfail all Whisper targets).
- [ ] **D5. doc-builder.** A release with the collector, or pin `git+https://github.com/huggingface/doc-builder@<sha>`.

## Phase 1: validate the environment on the PR CI runner (this repository)

Question: do the `cpu-x86_64` fixture baselines hold on `aws-m8i-l-cache` in the CI image, with and without the lock's pins?

- [ ] Ask infra for `aws-m8i-l-cache` access (or run phase 1 from the D1a branch).
- [ ] `ci-image-probe.yml`: container `huggingface/transformers-torch-light`, runner `aws-m8i-l-cache`; install Transformers `8625785e`, then `run.py --fixtures-only --require-all` twice:
  - [ ] **as is**: image packages, `environment_differs` recorded;
  - [ ] **pinned**: `.github/scripts/install_pinned.sh` first.
- [ ] Record CPU model and flags (`avx512_bf16`, `amx_*`), package versions, install and run times.
- [ ] Decide: *as is* passes → no pin install; only *pinned* passes → install pins (measure cost); neither → an Intel platform class, and decide which runner records required baselines.
- [ ] Note the result in [evidence](evidence.md#stability-and-cost).

## Phase 2: package the fleet for Transformers

- [ ] Runner changes here first:
  - [ ] `--lock PATH` (default unchanged); expectations next to the lock.
  - [ ] Lock `path` optional: published contracts and fixtures need no source folder (only `--local-*` and script contracts do).
  - [ ] Import `patches` only when `local_patches` is non-empty.
  - [ ] Keep `schemas/` beside the runner.
- [ ] Transformers branch layout:
  - [ ] `tests/contracts/fleet.lock` (no `path`, no `local_patches`; `known_failures` per D4), `tests/contracts/expectations/` (x86 platform(s) from phase 1; drop `cpu-arm64` and `cuda` unless needed).
  - [ ] `utils/contracts/run.py`, `select_contracts.py`, `schemas/`, `report.py` (phase 3), with a header naming this repository as the source.
- [ ] On the branch, `python utils/contracts/run.py --framework transformers --fixtures-only --lock tests/contracts/fleet.lock` passes on its own build.

## Phase 3: the CI job (non-blocking)

- [ ] `.github/workflows/contracts.yml` on `pull_request` (per D1):
  - [ ] container and runner from phase 1; checkout with history for `git diff origin/main...HEAD`;
  - [ ] install `uv pip install -e .`, runner deps (jsonschema, doc-builder per D5), pins per D2;
  - [ ] cache `~/.cache/huggingface/hub` keyed on the lock hash (~150 MB of fixtures and contracts);
  - [ ] run `git diff --name-only origin/main...HEAD | python utils/contracts/run.py --framework transformers --fixtures-only --changed-files - --require-all --output results.jsonl` (first real use of selection without `--run-all`);
  - [ ] `continue-on-error: true`; upload `results.jsonl`.
- [ ] `utils/contracts/report.py` → `$GITHUB_STEP_SUMMARY`:
  - [ ] per target: contract, fixture, status, first failure line, baseline path;
  - [ ] repro: `python utils/contracts/run.py --framework transformers --fixtures-only --contract <C> --target <T>`;
  - [ ] `selected_by` and skipped contracts;
  - [ ] for `fail`/`error`: regression vs intended change, with a link to the baseline update steps.
- [ ] Budget: under 2 minutes on a warm cache (~13 s for 8 fixtures on a 2-core runner).

## Phase 4: demonstrate on real PRs (drafts against the D1 branch)

- [ ] **A. Clean:** an unrelated change runs no contract, or all pass, with a readable summary.
- [ ] **B. Shared code:** a no-op in `generation/utils.py` selects every contract; all pass; record time.
- [ ] **C. Model-scoped:** a change in `models/clip/` selects only CLIP.
- [ ] **D. Known regression:** revert `d7ba937671` (#49135, UMT5 non-causal under SDPA): T5 fails on `proto-umt5-small-slim` (`label_logprobs`, loss); the summary alone suffices to reproduce.
- [ ] **E. Live regression:** #49456 shows as `xfail` or failure per D4; the report makes the cause obvious.
- [ ] Per PR: selection, run and install time, report link. Add a "Transformers CI proof" section to [evidence](evidence.md).

## Phase 5: write-up and maintainer decision

- [ ] One page for maintainers: what ran, cost, what it caught, what a developer sees, added upkeep (baselines, fixtures, contract changes).
- [ ] Ask for: runtime and noise budget, owners per contract, moving the job into `transformers-ci` (D1b), the D4 policy.

## After the proof

- [ ] **Intended changes:** a PR-triggered job records candidate baselines on every required platform and posts them for review (`review_candidates.py` in CI), replacing three manual steps.
- [ ] **`run-slow`:** map `run-slow: <family>` to contracts; run full checkpoints on GPU via `self-comment-ci.yml`.
- [ ] **After merge and nightly** inside Transformers, with failure alerts (issue or Slack per failing run). The nightly here has none yet: add one regardless.
- [ ] **Contract and framework changing together:** candidate contract revision tested with the PR, then a lock bump.
- [ ] **Model updates:** test a new checkpoint revision, check its fixture key, propose pin and baseline updates.
- [ ] **Coverage:** one MoE or coupled-config model, one GPU-heavy; map each `run-slow` test a contract replaces.
- [ ] **Isolation and private artifacts** before outside contracts.
- [ ] Choose runners from [`resources`](design.md#minimum-hardware) instead of fixed jobs.

## Open items from the current session (finish first)

- [ ] **Restore GPU access.** Runs 37897926458 and 37897930698 and every pending GPU run were cancelled when this repository lost `aws-g5-4xlarge-cache` (2026-10-09 ~11:40 UTC). Once re-added: `nightly.yml --ref smollm-bf16-fixture -f mode=record -f transformers_ref=8625785e7e3e138a328ec2e678dc198e356e7c6c`, then `python example/review_candidates.py --platform cuda`; approve if only unchanged, added, or the expected SmolLM bfloat16 changes.
- [ ] **Merge `smollm-bf16-fixture` into `main` before the 02:41 UTC nightly.** Done on the branch: `cpu-x86_64` Whisper baselines for `max_length_tokens`, `assisted_tokens`, `longform_tokens` (07d0a6f); bfloat16 proto-SmolLM-slim baselines on `cpu-arm64`, `cpu-x86_64`, `cpu-x86_64-bf16`. Pending: `cuda` (above).
- [ ] Then re-run the backfill (`shadow-backfill.yml since=2026-05-01 until=2026-10-08`) and record the second round's detections in [improving contracts](improving-contracts.md).
- [ ] Track #49456 (Whisper `generate()` since #48282): drop `config-field-generation-params` once fixed. #49419 fixed #49403: drop `whisper-decode-tensor-batch` at the next pinned-build bump, with #49386 (Wav2Vec2) when it lands.
