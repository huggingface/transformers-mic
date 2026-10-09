# Plugging the fleet into Transformers CI: proof plan and to-do

Goal: run the contracts as a non-blocking job in Transformers' own PR CI, with selection, a pinned environment matching the baselines, and an actionable failure report; then ask maintainers whether to adopt it.

Status (2026-10-09): shadow, nightly, and backfill run in this repository; contracts and fixtures are on the Hub, pinned by revision and hash; baselines cover `cpu-arm64`, `cpu-x86_64`, `cpu-x86_64-bf16`, and `cuda`. All five decisions taken. Phases 1–3 done: on the `contracts-proof` branch of `huggingface/transformers`, `.github/workflows/contracts.yml` runs the contracts on the branch's own code in the PR CI runner and image, selected from changed files, with narrowed known failures and a job summary (7 pass + Whisper `xfail` in 17 s). Phase 4 done: all five cases behave as intended ([evidence](evidence.md#transformers-ci-proof)). Phase 5 write-up done ([below](#for-transformers-maintainers-model-integration-contracts-in-pr-ci)); next: share it and collect the maintainers' answers. Nothing runs in Transformers' PR CI for other PRs.

## Facts the plan rests on (checked 2026-10-09)

- PR CI is [`pr-ci-caller.yml`](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml), calling `huggingface/transformers-ci/.github/workflows/pr-ci_dynamic_caller_example.yml@main`: `tests_fetcher` picks test lists, then one reusable job per suite (`pr-ci_reusable_test_job.yml`).
- Torch test jobs run in `huggingface/transformers-torch-light` on runner group `aws-m8i-l-cache` and install the PR with `uv pip install -e .`.
- m8i is Intel; our `cpu-x86_64` baselines come from AMD EPYC (7R32, matched by 7763). bfloat16 already differs on native-bfloat16 AMD (9V45, hence `cpu-x86_64-bf16`). Intel agreement is untested, float32 included.
- This repository can use `aws-g5-4xlarge-cache` only, not `aws-m8i-l-cache`. It was dropped from the group's authorized repositories on 2026-10-09 (jobs queued from 07:14 UTC, no runner requested); infra (Guillaume Legendre) re-added it, and a `gpu-probe.yml` canary ran at 12:05 UTC. Jobs queued for more than ~30 minutes: ask infra whether the repository is still authorized.
- The runner needs doc-builder's runnable-markdown collector, on doc-builder `main` only. PyPI `hf-doc-builder` is 0.5.0 (2024-03-11). Transformers already installs doc-builder with `git+https://github.com/huggingface/doc-builder.git@main` (`build_pr_documentation.yml`) and pins doc-builder's reusable workflows by SHA.
- `pr-ci-security-gate.yml` (`pull_request_target`) gates fork PRs only; PRs from members, owners, and collaborators trigger PR CI directly. A same-repository PR runs the workflow files on its own branch. Branches named `ci-*` also trigger push CI.
- `known_failures` was per target (one open bug made the whole target `xfail`); since 2026-10-09 an entry can narrow to a failure message or to skipped outputs. A contract that raises stops there, so a matched error still hides the checks after it; a skipped output hides only itself.
- Runner today: `example/run.py`, `select_contracts.py`, `patches.py`, `schemas/`, reading `example/transformers/tests/contracts/fleet.lock` and `expectations/`. `local_patches` holds three workarounds (#49385, #49403, #49456).

## Decisions (taken 2026-10-09 unless noted)

- [x] **D1. Where the proof runs: a same-repository branch on `huggingface/transformers` with its own `.github/workflows/contracts.yml`, opened as a draft PR.** Real runner group (`aws-m8i-l-cache`) and image, no change to shared CI. Branch name must not start with `ci-`. Rejected for now: a `transformers-ci` branch (shared infra before evidence; the target once maintainers accept, D1b) and a fork on GitHub-hosted runners (no m8i or CI image).
- [x] **D2. Environment: the CI image as is, no pin install** (phase 1: 16/16 pass with the image's packages, which differ from the lock in Python, torch, numpy, and nine more). Results keep recording `environment_differs`; an image rebuild that breaks a baseline shows up as a failure on every PR, so the first move then is to rerun with the pins (16 s) to tell an environment change from a regression.
- [x] **D3. Runner packaging: vendored into `utils/contracts/`** (`run.py`, `select_contracts.py`, `schemas/`, `report.py`), with a header naming this repository as the source. A published package (the design's shared runner) after the proof.
- [x] **D4. Open framework bugs: no `local_patches` in Transformers; signature-scoped `known_failures`.** A patch changes the framework under test, acceptable for an outside observer, not for the framework's own CI. Each entry names the issue and either a failure message (`match`) or the outputs it skips (`skip_outputs`); any other failure still fails, and a pass with `match` reports "fixed, remove the entry". Whole-target xfail rejected: it would hide a second, unrelated Whisper failure. Today's entries, checked on `main` (`33ab420a77`) without `local_patches` (6 fixtures pass, stable over two isolated runs):
  - #49456: `Whisper/proto-whisper-tiny-slim`, `match` `You have modified the pretrained model configuration` → `xfail`. The error comes from the first `generate()`, so Whisper's later checks don't run until the fix: a matched raise is narrower than whole-target xfail but no substitute for the fix.
  - #49385: `Wav2Vec2/proto-wav2vec2-base-960h-slim`, `skip_outputs` `training.loss` and `training.gradients` until #49386 lands → `pass`, inference outputs compared. The uninitialized `masked_spec_embed` made the gradients non-finite on the m8i runner (first run, 37934608391), not on the M5: what uninitialized memory holds depends on the machine. Wav2Vec2's training step is unchecked in Transformers CI until the fix.
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

## Phase 2: package the fleet for Transformers (done 2026-10-09)

- [x] Runner changes here first (2026-10-09):
  - [x] Scoped `known_failures` (D4): a string keeps the whole-target form; an object `{issue, match?, skip_outputs?}` narrows it; results carry `skipped_outputs`. Tested: partial match fails, match plus skip gives `xfail`, skip alone passes, `match` on a passing target asks for removal, an `issue`-only object is rejected by the schema.
  - [x] `--lock PATH` (default unchanged); `expectations/` next to the lock; baseline paths in results relative to the runner or the working directory.
  - [x] Lock `path` optional: published contracts and fixtures run from the Hub; a missing `path` is an error only for `--local-*`, unpublished, and script contracts.
  - [x] `patches` imported only when `local_patches` is non-empty (already the case).
  - [x] `schemas/` stays beside the runner.
- [x] Transformers branch layout (`contracts-proof`):
  - [x] `tests/contracts/fleet-lock.json` (renamed: Transformers ignores `*.lock`), no `path`, no `local_patches`, `known_failures` per D4; `tests/contracts/expectations/` with `cpu-x86_64` and `cpu-x86_64-bf16` only (the m8i has `amx_bf16`).
  - [x] `utils/contracts/run.py`, `select_contracts.py`, `report.py`, `schemas/`, and a README naming this repository and commit as the source.
- [x] On the branch, `python utils/contracts/run.py --framework transformers --fixtures-only --lock tests/contracts/fleet-lock.json` passes on its own build (`main` at `90ef040d40`).

## Phase 3: the CI job (non-blocking; done 2026-10-09)

Run [37934829379](https://github.com/huggingface/transformers/actions/runs/37934829379): 7 pass, Whisper `xfail` on #49456, 17.3 s for 8 fixtures with `OMP_NUM_THREADS=8` (37 s without). The first run (37934608391) failed Wav2Vec2's gradient check: see D4.

- [x] `.github/workflows/contracts.yml` on `pull_request` (per D1):
  - [x] container and runner from phase 1; checkout with history for `git diff origin/main...HEAD`;
  - [x] install `uv pip install -e .`, runner deps (jsonschema, doc-builder at a pinned SHA per D5), pins per D2;
  - [x] cache `~/.cache/huggingface/hub` keyed on the lock hash (~150 MB of fixtures and contracts);
  - [x] run with `--changed-files` from `git diff --name-only origin/<base>...HEAD` (first real use of selection without `--run-all`); pushes to `contracts-proof` run every contract, pushes to `proof-*` select from their diff against it;
  - [x] `continue-on-error: true`; upload `results.jsonl`.
- [x] `utils/contracts/report.py` → `$GITHUB_STEP_SUMMARY`:
  - [x] per target: contract, fixture, status, first failure line, baseline path;
  - [x] repro: `python utils/contracts/run.py --framework transformers --fixtures-only --lock tests/contracts/fleet-lock.json --contract <C> --target <T>`;
  - [x] `selected_by` and skipped contracts;
  - [x] for `fail`/`error`: regression vs intended change, with a link to the baseline update steps.
- [x] Budget: under 2 minutes. 8 fixtures in 17 s with `OMP_NUM_THREADS=8` (37–39 s with all 128 cores).
- [x] HF cache: `/mnt/cache` is read-only; `actions/cache` on `/tmp/hf_home/hub`, keyed on the lock hash. Later: ask infra to pre-populate the shared cache.
- The job fails (red, not required) when a contract fails; the run step itself continues so the summary and artifact are always written.
- Fixed 2026-10-09 (`46fa53214f`): from `22d49c28` (the `proof-*` trigger) to `fd5ecc0768`, pushes to `contracts-proof` failed before the runner started, because a log line read the changed-files list, which full runs don't write (runs 37935918629, 37938153499). `proof-*` runs always have the list and were unaffected. Full run after the fix: [37938424467](https://github.com/huggingface/transformers/actions/runs/37938424467), 7 pass + Whisper `xfail`, green.

## Phase 4: demonstrate (branch pushes against the D1 branch)

Branch pushes, not draft PRs (decided 2026-10-09): a PR, even against `contracts-proof`, also triggers Transformers' full PR CI (`pr-ci-caller.yml`), the docs build, and the PR bots. `proof-*` branches off `contracts-proof` trigger only `contracts.yml` (and the secret scan), selected from their diff as a PR would be; the job summary is the same. Branches: `proof-a-docs`, `proof-b-shared`, `proof-c-clip`, `proof-d-umt5-revert`.

Runs of 2026-10-09 (each run's summary has the table and repro commands; `contracts` artifact has the records):

- [x] **A. Clean:** `proof-a-docs` (a docs page), [run 37935955253](https://github.com/huggingface/transformers/actions/runs/37935955253): no contract selected, green; 11.5 s of runner time (Hub and selection only).
- [x] **B. Shared code:** `proof-b-shared` (no-op in `generation/utils.py`), [run 37935960229](https://github.com/huggingface/transformers/actions/runs/37935960229): all 6 contracts selected ("shared code"), 7 pass + Whisper `xfail`, green, 17.8 s.
- [x] **C. Model-scoped:** `proof-c-clip` (no-op in `models/clip/modeling_clip.py`), [run 37935964228](https://github.com/huggingface/transformers/actions/runs/37935964228): only CLIP selected ("models/clip"), pass, 11.5 s.
- [x] **D. Known regression:** `proof-d-umt5-revert` (reverts `d7ba937671`, #49135), [run 37935967981](https://github.com/huggingface/transformers/actions/runs/37935967981): only T5 selected ("models/umt5"); `proto-t5-small-slim` passes, `proto-umt5-small-slim` fails (`label_logprobs` off by 1.22, `training.loss` by 0.204, the same deltas as in the backfill), red, with the repro command in the summary. 13.4 s.
- [x] **E. Live regression:** #49456 is `xfail` on the matched message only in B and every full run; the summary names the issue and that Whisper's later checks did not run.
- [x] Per run: selection, runner time, run link, in [evidence](evidence.md#transformers-ci-proof) (install over the image ~15 s, same on every run).
- [x] **Mac repro** (`fd5ecc0768`): `cpu-arm64` baselines added to the branch, copied from here. On an M5 with the branch's build: 7 pass + Whisper `xfail`, as in CI; on `proof-d-umt5-revert`, `proto-umt5-small-slim` fails with CI's deltas (1.22, 0.204), `proto-t5-small-slim` passes.

## Phase 5: write-up and maintainer decision

- [x] One page for maintainers (below), 2026-10-09.
- [ ] Share it with the Transformers CI maintainers and collect answers to the asks; record them here.

### For Transformers maintainers: model integration contracts in PR CI

**What it is.** A contract is a model's documented usage page (load, preprocess, generate or predict, decode, one training step) written as runnable doc-builder blocks, plus a manifest of the checkpoints it covers. In CI it runs on a slim fixture: the real config shrunk, the real tokenizer and processor files copied byte for byte, published on the Hub and pinned by revision and hash. Outputs (tokens, text, selected logits, loss, named gradient checks) are compared with reviewed baselines in `tests/contracts/expectations/`. Design: [design](design.md); fixtures: [Small fixtures](fixtures.md).

**What ran.** Branch `contracts-proof` of `huggingface/transformers`, [`contracts.yml`](https://github.com/huggingface/transformers/blob/contracts-proof/.github/workflows/contracts.yml): the branch's own build, on the PR test jobs' runner group and image (`aws-m8i-l-cache`, `transformers-torch-light`), with the image's packages as they are. 6 contracts, 7 architectures, 8 fixtures (SmolLM/Llama, CLIP, BERT, T5 + UMT5, Whisper, Wav2Vec2). Contracts are selected from the changed files: a model directory selects the contracts whose fixtures use it; other library code selects all; files outside `src/` select none.

| Change (demo branch) | Selected | Result | Runner time |
| --- | --- | --- | ---: |
| Docs page | none | green | 11.5 s |
| No-op in `generation/utils.py` | all 6 | 7 pass, Whisper expected failure (#49456) | 17.8 s |
| No-op in `models/clip/` | CLIP | green | 11.5 s |
| Revert of the UMT5 fix [#49135](https://github.com/huggingface/transformers/pull/49135) | T5 | **red**: UMT5 `label_logprobs` off by 1.22, loss by 0.20 | 13.4 s |

**Cost.** ~15 s to install the PR and the runner over the image, then 11–18 s of runner time with `OMP_NUM_THREADS=8`, about 1 min 45 s of job time with a cold cache. Fixtures and contracts are ~150 MB, cached per lock. Over five months of `main`, selection ran nothing on 52% of commits, everything on 46%, some contracts on 1%. CPU only; no GPU in the PR path.

**What it caught** (on history unless noted; [evidence](evidence.md)):

- UMT5 decoder not causal under SDPA: from its introducing commit (#47014, 2026-07-30), two months before the fix #49135.
- Whisper assisted `generate()` raising for 41 commits after #48981, from its introducing commit, by a call added for a different bug: the one catch no triage aimed at.
- Whisper `generate()` raising on every original checkpoint after #48282: caught by the nightly the night it merged, after the PR's own CI passed; reported as [#49456](https://github.com/huggingface/transformers/issues/49456).
- Four processor and tokenizer regressions, once the contracts made the calls users make (post-hoc: those calls were added knowing the bugs).
- Live bugs found while writing contracts: #49385 (Wav2Vec2-family `masked_spec_embed` uninitialized; fix #49386 open), #49403 (fixed by #49419), #49372.
- False alarms: one in five months (a benign load-time RNG shift, #46259), one baseline update.

**What a developer sees.** A non-required check. The job summary lists the selected contracts and why, one row per fixture, known failures labelled with their issue, and for each failure the output that moved, by how much, and the command to reproduce it. The decision is theirs: a regression is fixed in the PR; an intended change re-records the baseline and gets it reviewed.

**Upkeep.**

- **Baselines:** 2 updates in five months across 8 fixtures (one intended: T5 moved to SDPA; one benign). Each is a reviewed JSON change under `tests/contracts/expectations/`, per platform (`cpu-x86_64`, plus `cpu-x86_64-bf16` for bfloat16 fixtures on native-bfloat16 CPUs, which the m8i is). Recording is manual today (run with `--record` on the platform); a CI recording job is the first follow-up.
- **Known failures:** an open framework bug becomes a lock entry naming the issue and either the error message (expected failure only on that message) or the outputs to skip; a fixed bug makes the entry fail until removed. Today: #49456 (Whisper stops at `generate()`, so its later checks don't run) and #49385 (Wav2Vec2 loss and gradients skipped).
- **Fixtures and contracts:** maintained in the contracts repository and on the Hub, bumped here by a lock change; not rebuilt in CI. Rebuilds depend on hardware, so fixtures are published, not regenerated.
- **Runner:** vendored in `utils/contracts/` for the proof; a package later.

**Limits.** Six families is a pilot, not coverage: the generic shrink rule reaches 314 of 559 usage paths, but each contract is hand-written and reviewed. Most catches are post-hoc; one is not. Fixtures run on CPU only; real checkpoints and GPUs stay in nightly and slow jobs. The branch also carries `cpu-arm64` baselines, so the repro command in a failing job's summary works unchanged on a Mac: the UMT5 revert fails there with the same deltas as in CI.

**Asks, with proposals:**

1. **Runtime and noise budget.** Proposed: non-required on every PR for four weeks, at most ~2 min of job time; promote a contract to required once it has had no unexplained failure in that window.
2. **Owners.** A maintainer per contract family (Llama/SmolLM, CLIP, BERT, T5/UMT5, Whisper, Wav2Vec2) to review baseline changes and known-failure entries.
3. **Where the job lives (D1b).** Proposed: a `tests_contracts` job in `transformers-ci`'s dynamic caller, after the four weeks; the runner as a package or kept vendored.
4. **Known-failure policy (D4).** Proposed: no workarounds patched into the framework under test; entries narrowed to a message or outputs, each linked to an open issue, removed when it closes (the job enforces it).
5. **Baseline recording.** Proposed: a CI job that records candidate baselines on the CI platform when a PR changes outputs on purpose, posted for review (replaces the manual `--record`).

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

- [ ] **Share the phase 5 write-up** with the Transformers CI maintainers; record their answers to the five asks here, then plan the follow-ups (CI baseline recording first). `cpu-arm64` baselines are on the branch (`fd5ecc0768`).
- [ ] Re-vendor `utils/contracts/` on `contracts-proof` whenever `example/run.py`, `report.py`, or the lock changes here (the branch README names the source commit). Keep `proof-*` branches rebased on `contracts-proof`.
- [ ] Cleanup after the proof: delete `proof-*` branches (and `contracts-proof` once moved to `transformers-ci`, D1b); the worktree lived in a session scratchpad (`git -C ~/Dev/transformers worktree prune`).

- [x] **Backfill third round** (run 37930845944; shadow check on `main` 37930841449 passed): a call added for #48000 caught #48981 at its introducing commit, a regression no triage listed; no new false alarm, including the bfloat16 SmolLM fixture on both x86 platforms ([evidence](evidence.md#third-round-second-round-calls-and-bfloat16-smollm)).
- [ ] Track #49456 (Whisper `generate()` since #48282): drop `config-field-generation-params` once fixed. #49419 fixed #49403: drop `whisper-decode-tensor-batch` at the next pinned-build bump, with #49386 (Wav2Vec2) when it lands.
