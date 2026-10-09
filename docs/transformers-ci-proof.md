# Plugging the fleet into Transformers CI: proof plan and to-do

Goal: run the contracts as a non-blocking job in Transformers' own PR CI, with selection, a pinned environment matching the baselines, and an actionable failure report; then ask maintainers whether to adopt it.

Status (2026-10-09): shadow, nightly, and backfill run in this repository; contracts and fixtures are on the Hub, pinned by revision and hash; baselines cover `cpu-arm64`, `cpu-x86_64`, `cpu-x86_64-bf16`, and `cuda`. All five decisions taken. Phase 1 done: the fixture baselines hold on the PR CI runner and image. Nothing runs in Transformers' PR CI yet; a probe workflow runs on the `contracts-proof` branch of `huggingface/transformers`.

## Facts the plan rests on (checked 2026-10-09)

- PR CI is [`pr-ci-caller.yml`](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml), calling `huggingface/transformers-ci/.github/workflows/pr-ci_dynamic_caller_example.yml@main`: `tests_fetcher` picks test lists, then one reusable job per suite (`pr-ci_reusable_test_job.yml`).
- Torch test jobs run in `huggingface/transformers-torch-light` on runner group `aws-m8i-l-cache` and install the PR with `uv pip install -e .`.
- m8i is Intel; our `cpu-x86_64` baselines come from AMD EPYC (7R32, matched by 7763). bfloat16 already differs on native-bfloat16 AMD (9V45, hence `cpu-x86_64-bf16`). Intel agreement is untested, float32 included.
- This repository can use `aws-g5-4xlarge-cache` only, not `aws-m8i-l-cache`. It was dropped from the group's authorized repositories on 2026-10-09 (jobs queued from 07:14 UTC, no runner requested); infra (Guillaume Legendre) re-added it, and a `gpu-probe.yml` canary ran at 12:05 UTC. Jobs queued for more than ~30 minutes: ask infra whether the repository is still authorized.
- The runner needs doc-builder's runnable-markdown collector, on doc-builder `main` only. PyPI `hf-doc-builder` is 0.5.0 (2024-03-11). Transformers already installs doc-builder with `git+https://github.com/huggingface/doc-builder.git@main` (`build_pr_documentation.yml`) and pins doc-builder's reusable workflows by SHA.
- `pr-ci-security-gate.yml` (`pull_request_target`) gates fork PRs only; PRs from members, owners, and collaborators trigger PR CI directly. A same-repository PR runs the workflow files on its own branch. Branches named `ci-*` also trigger push CI.
- `known_failures` today is per target: one open bug makes the whole target `xfail`, every other check included.
- Runner today: `example/run.py`, `select_contracts.py`, `patches.py`, `schemas/`, reading `example/transformers/tests/contracts/fleet.lock` and `expectations/`. `local_patches` holds three workarounds (#49385, #49403, #49456).

## Decisions (taken 2026-10-09 unless noted)

- [x] **D1. Where the proof runs: a same-repository branch on `huggingface/transformers` with its own `.github/workflows/contracts.yml`, opened as a draft PR.** Real runner group (`aws-m8i-l-cache`) and image, no change to shared CI. Branch name must not start with `ci-`. Rejected for now: a `transformers-ci` branch (shared infra before evidence; the target once maintainers accept, D1b) and a fork on GitHub-hosted runners (no m8i or CI image).
- [x] **D2. Environment: the CI image as is, no pin install** (phase 1: 16/16 pass with the image's packages, which differ from the lock in Python, torch, numpy, and nine more). Results keep recording `environment_differs`; an image rebuild that breaks a baseline shows up as a failure on every PR, so the first move then is to rerun with the pins (16 s) to tell an environment change from a regression.
- [x] **D3. Runner packaging: vendored into `utils/contracts/`** (`run.py`, `select_contracts.py`, `schemas/`, `report.py`), with a header naming this repository as the source. A published package (the design's shared runner) after the proof.
- [x] **D4. Open framework bugs: no `local_patches` in Transformers; signature-scoped `known_failures`.** A patch changes the framework under test, acceptable for an outside observer, not for the framework's own CI. Each entry names the issue and either a failure signature (an error substring) or the outputs it skips; any other failure still fails, and a pass reports "fixed, remove the entry". Whole-target xfail rejected: while #49456 is open it would blind every Whisper check. Today's entries:
  - #49456: Whisper targets, signature `You have modified the pretrained model configuration`.
  - #49385: Wav2Vec2 targets, skip `training.loss` (depends on uninitialized `masked_spec_embed`) until #49386 lands.
  - #49403: none; fixed on `main` by #49419, which the PR builds include.
- [x] **D5. doc-builder: pinned by SHA**, `git+https://github.com/huggingface/doc-builder@<sha>`, as Transformers already does for doc-builder's workflows. No release needed for the proof.

## Phase 1: validate the environment on the PR CI runner (done 2026-10-09)

Question: do the `cpu-x86_64` fixture baselines hold on `aws-m8i-l-cache` in the CI image, with and without the lock's pins? **Yes, both** ([evidence](evidence.md#on-the-transformers-pr-ci-runner)).

- [x] Run from the D1 branch: `contracts-proof` on `huggingface/transformers`, `.github/workflows/contracts-probe.yml` on push to that branch only (no PR; only the secret scan also runs), runner vendored in `utils/contracts/` from `b303626`. No infra request needed.
- [x] Transformers `8625785e`, eight fixtures, default and eager: **as is** 16/16, **pinned** 16/16.
- [x] Host: Intel Xeon 6975P-C, 128 cores, `avx512_bf16`, `amx_bf16`; Python 3.11.17 in the image. Install over the image 14 s, pins 16 s; run 37–57 s.
- [x] Scope: 6 contracts (SmolLM, CLIP, BERT, T5 + UMT5, Whisper, Wav2Vec2), 7 architectures, 8 fixtures; 32 checks (2 variants × 2 package sets). Whole job 3 min 55 s, 9 s queued; the part a PR job needs (setup, install over the image, 16 checks with cold Hub downloads) ~1 min 45 s.
- Not yet a PR check: it installs `8625785e`, not the branch; runs on push to `contracts-proof` only; runs every contract (no selection); still applies `local_patches`; reports no repro. Phases 2 and 3 close these.
- [x] Decided D2: image as is.
- Learned on the way: Transformers' `.gitignore` has `*.lock`, so the vendored `fleet.lock` needs `git add -f` (or another name in phase 2). The shared `/mnt/cache` HF cache is read-only in PR jobs; the probe downloads into `/tmp` (~150 MB).

## Phase 2: package the fleet for Transformers (next)

- [ ] Runner changes here first:
  - [ ] Signature-scoped `known_failures` (D4): entry `{issue, match | skip_outputs}`; schema bump; a pass or a non-matching failure is reported as such. Keep the plain string form for this repository's lock.
  - [ ] `--lock PATH` (default unchanged); expectations next to the lock.
  - [ ] Lock `path` optional: published contracts and fixtures need no source folder (only `--local-*` and script contracts do).
  - [ ] Import `patches` only when `local_patches` is non-empty.
  - [ ] Keep `schemas/` beside the runner.
- [ ] Transformers branch layout:
  - [ ] `tests/contracts/fleet.lock` (`*.lock` is gitignored in Transformers: rename, e.g. `fleet-lock.json`, or `git add -f`; no `path`, no `local_patches`; `known_failures` per D4), `tests/contracts/expectations/` (x86 platform(s) from phase 1, `cpu-x86_64-bf16` if the m8i has `avx512_bf16`/`amx_bf16`; drop `cpu-arm64` and `cuda` unless needed).
  - [ ] `utils/contracts/run.py`, `select_contracts.py`, `schemas/`, `report.py` (phase 3), with a header naming this repository as the source.
- [ ] On the branch, `python utils/contracts/run.py --framework transformers --fixtures-only --lock tests/contracts/fleet.lock` passes on its own build.

## Phase 3: the CI job (non-blocking)

- [ ] `.github/workflows/contracts.yml` on `pull_request` (per D1):
  - [ ] container and runner from phase 1; checkout with history for `git diff origin/main...HEAD`;
  - [ ] install `uv pip install -e .`, runner deps (jsonschema, doc-builder at a pinned SHA per D5), pins per D2;
  - [ ] cache `~/.cache/huggingface/hub` keyed on the lock hash (~150 MB of fixtures and contracts);
  - [ ] run `git diff --name-only origin/main...HEAD | python utils/contracts/run.py --framework transformers --fixtures-only --changed-files - --require-all --output results.jsonl` (first real use of selection without `--run-all`);
  - [ ] `continue-on-error: true`; upload `results.jsonl`.
- [ ] `utils/contracts/report.py` → `$GITHUB_STEP_SUMMARY`:
  - [ ] per target: contract, fixture, status, first failure line, baseline path;
  - [ ] repro: `python utils/contracts/run.py --framework transformers --fixtures-only --contract <C> --target <T>`;
  - [ ] `selected_by` and skipped contracts;
  - [ ] for `fail`/`error`: regression vs intended change, with a link to the baseline update steps.
- [ ] Budget: under 2 minutes. On the m8i, 8 fixtures take 37–39 s in one process (vs ~13 s on a 2-core GitHub runner): try `OMP_NUM_THREADS` (PR jobs set it per suite).
- [ ] HF cache: `/mnt/cache` is read-only; use `actions/cache` keyed on the lock hash, or ask infra to pre-populate the shared cache with the pinned fixtures and contracts.

## Phase 4: demonstrate on real PRs (drafts against the D1 branch)

- [ ] **A. Clean:** an unrelated change runs no contract, or all pass, with a readable summary.
- [ ] **B. Shared code:** a no-op in `generation/utils.py` selects every contract; all pass; record time.
- [ ] **C. Model-scoped:** a change in `models/clip/` selects only CLIP.
- [ ] **D. Known regression:** revert `d7ba937671` (#49135, UMT5 non-causal under SDPA): T5 fails on `proto-umt5-small-slim` (`label_logprobs`, loss); the summary alone suffices to reproduce.
- [ ] **E. Live regression:** #49456 shows as `xfail` on the matched signature only; other Whisper checks still run; the report makes the cause obvious.
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

## Done in the 2026-10-09 session

- [x] **GPU access restored** (see facts above); the stuck recordings were cancelled and re-run.
- [x] **Fixtures load in their source checkpoint's dtype** ([evidence](evidence.md#bfloat16-on-cpu)): proto-SmolLM-slim rebuilt in bfloat16, overfit by reviewed choice (`overfit`, `target_loss` in the recipe), republished and repinned with the SmolLM contract.
- [x] **`cpu-x86_64-bf16` platform** for bfloat16 runs on x86 CPUs with native bfloat16; real SmolLM checkpoints differ there by up to 0.25 in logits.
- [x] **Baselines re-recorded and approved** on `cpu-arm64` (M5), `cpu-x86_64` (EPYC 7763), `cpu-x86_64-bf16` (EPYC 9V45), and `cuda` (A10G, run 37927847268): every float32 target unchanged; Whisper gains `max_length_tokens`, `assisted_tokens`, `longform_tokens` on every platform; only proto-SmolLM-slim changed. Merged to `main` (`f5b6b79`) ahead of the 02:41 UTC nightly.

## Open items

- [x] **Backfill third round** (run 37930845944; shadow check on `main` 37930841449 passed): a call added for #48000 caught #48981 at its introducing commit, a regression no triage listed; no new false alarm, including the bfloat16 SmolLM fixture on both x86 platforms ([evidence](evidence.md#third-round-second-round-calls-and-bfloat16-smollm)).
- [ ] Track #49456 (Whisper `generate()` since #48282): drop `config-field-generation-params` once fixed. #49419 fixed #49403: drop `whisper-decode-tensor-batch` at the next pinned-build bump, with #49386 (Wav2Vec2) when it lands.
