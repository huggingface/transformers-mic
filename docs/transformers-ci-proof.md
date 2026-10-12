# Transformers CI proof

## Current prototype proof (2026-10-12)

The existing `contracts-proof` branch now installs the **transformers-mic package**
at an immutable Git commit and runs two separate locks: the original six-contract,
eight-fixture historical control (default variant) and the **23-architecture
prototype fleet** (default, eager, bfloat16: 69 cases). Full proof runs explicitly
use `--run-all`; a coverage guard fails if any prototype case is missing.
Model-selection greens that ran zero cases are not model-validation evidence.
The proof is not adopted for unrelated Transformers PRs.

The prototype fleet is fixture-only and was generated without historical replay.
It adds GPT2, ModernBERT, ViT, Table Transformer, GOT-OCR2, Qwen2-Audio,
SpeechT5, Gemma3, Qwen2, Mistral3, Qwen2.5-VL, LLaVA, SmolVLM, InternVL,
GPT-OSS, DETR and Qwen2.5-Omni to the overlapping historical families.
Auto contracts use Auto classes for BERT/T5/ViT. The full requested catalog has
26 concrete architectures; CSM, Gemma3n and PI0 remain rejected. Their gate
failures are recorded in [the per-model report](../experiments/results/prototype-regeneration.json).
GPT-OSS coverage uses a dense slim and does not exercise quantized kernels.
DETR/SpeechT5 recipes explicitly exempt amplitude scaling canceled by input
normalization; required remaining probes and the .001 threshold stay fixed.
[Original failed probes](../experiments/results/prototype-normalization-probes.json)
are retained rather than relabeled as passes.

CPU jobs use four threads (matching native reference recording), pin
torch/torchvision/torchaudio 2.11/0.26/2.11 and scientific dependencies
including NumPy 1.26.4. Python comes from the runner image and is reported as an
environment difference (native recording used 3.11.17 versus the catalog's 3.10).
A rolling-image attempt genuinely executed 59 cases, with 23 pass and 36 fail:
missing native precision references and numeric deltas under torch 2.14/NumPy
2.3.5 exposed why GPU-worker CPU references alone were insufficient. Tolerances
were not loosened.

The [native CPU recording run](https://github.com/huggingface/transformers/actions/runs/38281747657)
on Intel m8i checked and exported **22 models / 66 variants**. Review accepted
only new or unchanged candidates; native BF16/FP16 profiles have their own files.
Imported outputs also compare within existing tolerances against the previous
proof references. [Recording/review evidence](../experiments/results/native-cpu-review.json)
and [import comparison](../experiments/results/native-cpu-import-comparison.json)
retain those checks. Recording and subsequent checks used independent processes;
the full fleet also checks both locks through the packaged CLI and verifies
case counts. Prototype variants currently launch a backend process per case.

No runtime framework patches enter the Transformers integration. Two narrow
exceptions preserve unresolved signals:

- **Prototype Whisper:** #49456's generation error is matched by its message.
  All three variants remain blocked, with no native CPU export or downstream
  checks claimed. The historical Whisper control also reproduces that signature
  against the explicitly selected source, so its message match is retained.
- **Wav2Vec2:** #49385 skips only `training.loss` and `training.gradients`.
  Reference recording initializes the missing SpecAugment parameter in a
  controlled run; final checks run unpatched and retain the explicit skips.

Some published GPU validation runs likewise used model-scoped patches. These
are controlled results, not evidence of unpatched upstream behavior. Immutable
Hub fixture/contract pins are checked after publication; build success alone
is not a published-pin check. No historical catch is attributed to this expanded
prototype fleet. Wider checkpoint validation, rejected fixtures, repeated native
CPU noise measurements and maintainer adoption remain follow-up work.

The [first full integration run](https://github.com/huggingface/transformers/actions/runs/38286964684)
executed all **77 cases** (69 prototypes plus eight historical controls):
**74 pass, three Whisper failures**. Whisper reached generation but then exceeded
its old F32 reference by 0.000112 and lacked a native BF16 reference. Those
failures were not approved or converted to expected numeric failures.
[First full-run evidence](../experiments/results/native-cpu-full-first.json).

The [focused Whisper recorder](https://github.com/huggingface/transformers/actions/runs/38293434434)
reproduced the generation error in all variants and verified the same source
module and configuration hash from both working directories.
[Provenance evidence](../experiments/results/native-whisper-provenance.json).
Both workflows now explicitly set checkout `src` on `PYTHONPATH`; full checks
also assert the framework import path. The source-verified
[full rerun](https://github.com/huggingface/transformers/actions/runs/38293434373)
executed all **77 cases**: **73 pass, three prototype Whisper xfails and one
historical Whisper error**. Every numeric comparison passed. The historical
Whisper error has the same #49456 signature, so its exact match was restored.
[Source-verified results](../experiments/results/native-cpu-source-verified.json).
The source-verified path changes the outcome from the first full run; that
first run is an environment observation, not validation of the reviewed source.
The verified suite took about **12 minutes** in the contract step, including
per-case subprocess startup; the eight historical controls take about 20 seconds.
These timings describe the expanded alpha CLI, not the small pilot's PR budget.
A subsequent [full run](https://github.com/huggingface/transformers/actions/runs/38296023340)
executed 77 cases with **72 pass, four Whisper xfails and one Table Transformer
BF16 numeric failure** (score delta 0.00195). It used eight threads, while native
references used four. [Thread/noise evidence](../experiments/results/native-cpu-thread-noise.json).
The proof now uses and asserts the four-thread reference profile. The
[focused native Table Transformer repeat](https://github.com/huggingface/transformers/actions/runs/38298142514)
passed fresh-process recording and all three checks; its checked outputs match
the original native references **exactly**. No replacement baseline was imported.
[Repeatability comparison](../experiments/results/native-table-repeatability.json).
The [aligned full rerun](https://github.com/huggingface/transformers/actions/runs/38298088861)
**passed: 77 executed cases, 73 pass and four exact Whisper xfails**. The coverage
guard verified all 69 prototype cases. Every non-Whisper comparison passed;
Wav2Vec2 retains only its stated loss/gradient skips. No runtime patches,
tolerance changes or changed reference approvals were made.
[Final full-run records](../experiments/results/native-cpu-final.json).

Final three published GPU pin checks:
[run 38286933326](https://github.com/huggingface/transformers-mic/actions/runs/38286933326),
**9/9 pass**, without runtime patches (DETR, SpeechT5, Qwen2.5-Omni).

## Historical pilot (2026-10-09)

The remaining plan, timings, issue states and results are the original pilot
snapshot. They describe the smaller vendored runner before the package and
prototype fleet updates above, and are retained as historical evidence.

## For Transformers maintainers: model integration contracts in PR CI

**What it is.** Runnable doc-builder usage (load, preprocess, generate/predict, decode, train) plus a checkpoint manifest. PR CI uses Hub-published fixtures: shrunk real configs, unchanged tokenizer/processor files, pinned revision and hash. Tokens, text, selected logits, loss, and named gradients are compared with reviewed `tests/contracts/expectations/` baselines. Details: [design](design.md), [fixtures](fixtures.md).

**What ran.** Branch `contracts-proof` of `huggingface/transformers`, [`contracts.yml`](https://github.com/huggingface/transformers/blob/contracts-proof/.github/workflows/contracts.yml): the branch's build on the PR test runner/image (`aws-m8i-l-cache`, `transformers-torch-light`), with unchanged image packages. Scope: 6 contracts, 7 architectures, 8 fixtures (SmolLM/Llama, CLIP, BERT, T5 + UMT5, Whisper, Wav2Vec2). Model-directory changes select affected contracts; shared library code and `setup.py`/`pyproject.toml` select all; other files select none.

Proof results and issue states below are the **2026-10-09 snapshot**. The branch is not adopted for ordinary PRs and still needs the newer MusicGen Melody and Whisper coverage vendored from this repository.

| Change (demo branch) | Selected | Result | Runner time |
| --- | --- | --- | ---: |
| Docs page | none | green | 11.5 s |
| No-op in `generation/utils.py` | all 6 | 7 pass, Whisper expected failure (#49456) | 17.8 s |
| No-op in `models/clip/` | CLIP | green | 11.5 s |
| Revert of the UMT5 fix [#49135](https://github.com/huggingface/transformers/pull/49135) | T5 | **red**: UMT5 `label_logprobs` off by 1.22, loss by 0.20 | 13.4 s |

**Cost.**

- About 15 s to install the PR and the runner over the image.
- Then 11–18 s of runner time with `OMP_NUM_THREADS=8`.
- About 1 min 45 s of job time with a cold cache.
- Fixtures and contracts total about 150 MB, cached per lock.
- Over five months of `main`, selection ran nothing on 52% of commits, everything on 46%, and some contracts on 1%.
- CPU only; no GPU in the PR path.

**What it caught** (on history unless noted; [evidence](evidence.md)):

- **UMT5 decoder not causal under SDPA:** caught from its introducing commit (#47014, 2026-07-30), two months before the fix, #49135.
- **Whisper assisted `generate()` raising after #48981:** caught for 41 commits, from the introducing commit, by a call added for a different bug. This is the one catch no triage aimed at.
- **Whisper `generate()` raising on every original checkpoint after #48282:** caught by the nightly the night it merged, after the PR's own CI had passed; reported as [#49456](https://github.com/huggingface/transformers/issues/49456).
- **Four processor and tokenizer regressions**, once the contracts made the calls users make. These are post-hoc: the calls were added knowing the bugs.
- **Live bugs found while writing contracts:**
  - #49385: Wav2Vec2-family `masked_spec_embed` left uninitialized; fix #49386 is open.
  - #49403: fixed by #49419.
  - #49372.
- **False alarms:** one in five months (a benign load-time RNG shift, #46259), which needed one baseline update.

**What a developer sees.** A non-required check. The job summary lists:

- the selected contracts and why;
- one row per fixture;
- known failures, each labelled with its issue;
- for each failure, the output that moved, by how much, and the command that reproduces it.

The repro command also works on a Mac, because the branch carries `cpu-arm64` baselines; the UMT5 revert fails there with the same deltas as in CI. The decision stays with the developer: a regression is fixed in the PR, and an intended change re-records the baseline, which then gets reviewed.

**Upkeep.**

- **Baselines:** 2 updates in five months across 8 fixtures (one intended: T5 moved to SDPA; one benign).
  - Each update is a reviewed JSON change under `tests/contracts/expectations/`, made per platform: `cpu-x86_64` in CI, plus `cpu-x86_64-bf16` for bfloat16 fixtures on CPUs with native bfloat16, which the m8i has.
  - Recording is manual today (`--record` on the platform). A CI recording job is the first follow-up.
- **Known failures:** an open framework bug becomes a lock entry that names the issue and gives either the error message (expected failure only on that message) or the outputs to skip.
  - A message-matched entry fails on an unexpected pass. Skipped outputs need manual review and removal after the fix; the runner does not detect their recovery.
  - Today's entries: #49456 (Whisper stops at `generate()`, so its later checks don't run) and #49385 (Wav2Vec2 loss and gradients skipped).
- **Fixtures and contracts:** maintained in the contracts repository and on the Hub, and bumped here by a lock change. CI never rebuilds them: rebuilds depend on the hardware, so fixtures are published, not regenerated.
- **Runner:** vendored in `utils/contracts/` for the proof; a package later.

**Limits.**

- Six families is a pilot, not coverage. A seventh, MusicGen Melody, runs in the contracts repository but is not on the branch yet.
- The generic shrink rule reaches 314 of 559 usage paths, but each contract is hand-written and reviewed.
- Most catches are post-hoc; one is not.
- Fixtures run on CPU only; real checkpoints and GPUs stay in nightly and slow jobs.

**Asks, with proposals:**

1. **Runtime and noise budget.** Proposed: non-required on every PR for four weeks, at most about 2 min of job time. A contract becomes required once it has had no unexplained failure in that window.
2. **Owners.** One maintainer per contract family (Llama/SmolLM, CLIP, BERT, T5/UMT5, Whisper, Wav2Vec2) to review baseline changes and known-failure entries.
3. **Where the job lives (D1b).** Proposed: a `tests_contracts` job in `transformers-ci`'s dynamic caller, after the four weeks, with the runner either packaged or kept vendored.
4. **Known-failure policy (D4).** Proposed: no framework patches; narrow each entry to a message or outputs, link its issue, and remove it after the tested build includes the fix. The job enforces unexpected passes for message matches; output skips require review.
5. **Baseline recording.** Proposed: a CI job that records candidate baselines on the CI platform when a PR changes outputs on purpose, and posts them for review. It replaces the manual `--record`.

## The proof plan

Goal: run the contracts as a non-blocking job in Transformers' own PR CI, with selection, an environment that matches the baselines, and an actionable failure report; then ask maintainers whether to adopt it.

**Status (2026-10-09):** all five decisions are taken and phases 1–5 are done. The write-up is [above](#for-transformers-maintainers-model-integration-contracts-in-pr-ci); sharing it with the CI maintainers is set aside for now. Nothing runs in Transformers' PR CI for other PRs.

## Facts the plan rests on (checked 2026-10-09)

- **PR CI structure:** [`pr-ci-caller.yml`](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml) calls `huggingface/transformers-ci/.github/workflows/pr-ci_dynamic_caller_example.yml@main`. `tests_fetcher` picks the test lists, then each suite runs as one reusable job (`pr-ci_reusable_test_job.yml`).
- **Torch test jobs** run in `huggingface/transformers-torch-light` on runner group `aws-m8i-l-cache` (Intel Xeon 6975P-C, `amx_bf16`) and install the PR with `uv pip install -e .`.
  - Our `cpu-x86_64` baselines come from AMD EPYC chips and hold on this Intel runner (phase 1).
  - Intel `amx_bf16` matched the AMD `cpu-x86_64-bf16` baseline on the bfloat16 fixture. Real bfloat16 checkpoints on Intel are untested.
- **Runner groups:** this repository can use `aws-g5-4xlarge-cache`, but not `aws-m8i-l-cache`.
  - It was dropped from the g5 group's authorized repositories on 2026-10-09, then re-added by infra.
  - If jobs sit queued for more than about 30 minutes, ask infra whether the repository is still authorized.
- **doc-builder:** the runner needs doc-builder's runnable-markdown collector, which exists only on doc-builder `main`; PyPI `hf-doc-builder` is 0.5.0 (2024-03-11). Transformers already installs doc-builder from `git+…@main` (`build_pr_documentation.yml`) and pins doc-builder's reusable workflows by SHA.
- **Who triggers PR CI:** `pr-ci-security-gate.yml` (`pull_request_target`) gates fork PRs only, so PRs from members, owners and collaborators trigger PR CI directly. A same-repository PR runs the workflow files from its own branch. Branches named `ci-*` also trigger push CI.
- **Known failures:** an entry narrows to a failure message or to skipped outputs. A contract that raises stops there, so a matched error still hides the checks after it; a skipped output hides only itself.

## Decisions (taken 2026-10-09)

- **D1. Where the proof runs: a same-repository branch on `huggingface/transformers` with its own `.github/workflows/contracts.yml`.**
  - It uses the real runner group and image, with no change to shared CI. The branch name must not start with `ci-`.
  - Rejected for now: a `transformers-ci` branch (shared infra before there is evidence; it is the target once maintainers accept, D1b), and a fork on GitHub-hosted runners (no m8i, no CI image).
- **D2. Environment: the CI image as is, no pin install.**
  - Phase 1: 16/16 pass with the image's packages, which differ from the lock in Python, torch, numpy, and nine more.
  - Results still record `environment_differs`. If an image rebuild breaks a baseline, it fails on every PR; rerunning with the pins (16 s) then tells an environment change from a regression.
- **D3. Runner packaging: vendored into `utils/contracts/`** (`run.py`, `select_contracts.py`, `schemas/`, `report.py`), with a README naming this repository and the source commit. A published package (the design's shared runner) comes after the proof.
- **D4. Open framework bugs: no `local_patches` in Transformers; known failures scoped by signature.**
  - A patch changes the framework under test. That is acceptable for an outside observer, not for the framework's own CI.
  - Each entry names the issue and either a failure message (`match`) or the outputs it skips (`skip_outputs`). Any other failure still fails, and a pass with `match` reports "fixed, remove the entry".
  - Whole-target xfail was rejected: it would hide a second, unrelated Whisper failure.
  - Current entries, checked on `main` (`33ab420a77`):
    - **#49456**, Whisper fixture, `match` "You have modified the pretrained model configuration" → `xfail`. The error comes from the first `generate()`, so Whisper's later checks don't run until the fix.
    - **#49385**, Wav2Vec2 fixture, `skip_outputs` `training.loss` and `training.gradients` until #49386 lands → `pass` on the inference outputs. The uninitialized `masked_spec_embed` made the gradients non-finite on the m8i runner (run 37934608391), but not on the M5: what uninitialized memory holds depends on the machine.
    - **#49403:** no entry needed; it is fixed on `main` by #49419, which the PR builds include.
- **D5. doc-builder: pinned by SHA**, `git+https://github.com/huggingface/doc-builder@<sha>`, as Transformers already does for doc-builder's workflows. No release is needed for the proof.

## Phases (all done 2026-10-09)

Evidence for every phase: [on the PR CI runner](evidence.md#on-the-transformers-pr-ci-runner) and [Transformers CI proof](evidence.md#transformers-ci-proof).

**1. Environment on the PR CI runner.** Question: do the `cpu-x86_64` fixture baselines hold on `aws-m8i-l-cache` in the CI image? **Yes**, with and without the lock's pins.

- Setup: probe workflow `contracts-probe.yml`, run on pushes to `contracts-proof`, on Transformers `8625785e`, with eight fixtures in the default and eager variants.
- Results: 16/16 with the image's packages, 16/16 with the pins.
- Timing: install over the image 14 s, pins 16 s, run 37–57 s. The part a PR job needs takes about 1 min 45 s.
- Learned:
  - Transformers' `.gitignore` excludes `*.lock`, hence the name `fleet-lock.json`.
  - `/mnt/cache` is read-only in PR jobs, so fixtures download into `/tmp` (about 150 MB).

**2. Packaging.**

- Runner changes made here first:
  - scoped `known_failures` (D4), whose results carry `skipped_outputs`;
  - `--lock PATH`, with `expectations/` next to the lock;
  - an optional lock `path` (published contracts and fixtures run from the Hub).
- On the branch:
  - `tests/contracts/fleet-lock.json`: no `path`, no `local_patches`, `known_failures` per D4;
  - `tests/contracts/expectations/` for `cpu-x86_64` and `cpu-x86_64-bf16`, plus `cpu-arm64` (`fd5ecc0768`) so failures reproduce on a Mac;
  - `utils/contracts/` with the runner and a README.

**3. The CI job.** `contracts.yml` runs on `pull_request`, plus pushes to `contracts-proof` (every contract) and to `proof-*` (selected from their diff against it).

- Steps: install `uv pip install -e .`, then the runner's dependencies (jsonschema, doc-builder at a pinned SHA), then run with `--changed-files` from `git diff origin/<base>...HEAD`.
- The HF cache is restored with `actions/cache` on `/tmp/hf_home/hub`, keyed on the lock hash.
- The run step continues on error, so the summary (`report.py` → `$GITHUB_STEP_SUMMARY`: per-target rows, repro command, `selected_by`, what to do next) and the `results.jsonl` artifact are always written. The job then turns red (not required) if a contract failed.
- Runs:
  - [37934829379](https://github.com/huggingface/transformers/actions/runs/37934829379): 7 pass + Whisper `xfail` in 17.3 s with `OMP_NUM_THREADS=8` (37 s with all 128 cores).
  - Fixed in `46fa53214f`: from `22d49c28` to `fd5ecc0768`, full runs failed before the runner started, because a log line read a changed-files list those runs don't write (37935918629, 37938153499). The `proof-*` runs were unaffected. Green after the fix: [37938424467](https://github.com/huggingface/transformers/actions/runs/37938424467).

**4. Demonstration.** The demos are pushes to `proof-*` branches off `contracts-proof`, not draft PRs: a PR would also trigger Transformers' full PR CI, the docs build, and the PR bots. Each run's summary has the table and repro commands, and its `contracts` artifact has the records.

| Case | Branch | Run | Outcome |
| --- | --- | --- | --- |
| A. Clean | `proof-a-docs` | [37935955253](https://github.com/huggingface/transformers/actions/runs/37935955253) | nothing selected, green, 11.5 s |
| B. Shared code | `proof-b-shared` | [37935960229](https://github.com/huggingface/transformers/actions/runs/37935960229) | all 6, 7 pass + Whisper `xfail`, green, 17.8 s |
| C. Model-scoped | `proof-c-clip` | [37935964228](https://github.com/huggingface/transformers/actions/runs/37935964228) | CLIP only, pass, 11.5 s |
| D. Known regression | `proof-d-umt5-revert` (reverts `d7ba937671`, #49135) | [37935967981](https://github.com/huggingface/transformers/actions/runs/37935967981) | T5 only; UMT5 fails with the backfill's deltas (1.22, 0.204), red, 13.4 s |
| E. Live regression | B and every full run | — | #49456 `xfail` on its message; the summary says Whisper's later checks did not run |

Mac repro: with the branch's build on an M5, the results are 7 pass + Whisper `xfail`, as in CI. On `proof-d-umt5-revert`, UMT5 fails with CI's deltas.

**5. Write-up.** [For Transformers maintainers](#for-transformers-maintainers-model-integration-contracts-in-pr-ci). Next: share it and record the answers to the five asks here.

## After the proof

- [ ] **Intended changes:** a PR-triggered job records candidate baselines on every required platform and posts them for review (`review_candidates.py` in CI), replacing three manual steps.
- [ ] **`run-slow`:** map `run-slow: <family>` to contracts, and run full checkpoints on GPU through `self-comment-ci.yml`.
- [ ] **After-merge and nightly runs** inside Transformers, with failure alerts (an issue or a Slack message per failing run). The nightly here has no alerts yet: add them regardless.
- [ ] **Contract and framework changing together:** test a candidate contract revision with the PR, then bump the lock.
- [ ] **Model updates:** test a new checkpoint revision, check its fixture key, and propose pin and baseline updates.
- [ ] **Coverage:** one MoE model and one GPU-heavy model (the coupled-config case now exists: MusicGen Melody); map each `run-slow` test a contract replaces.
- [ ] **Isolation and private-artifact rules** before accepting outside contracts.
- [ ] **Runner choice:** choose runners from [`resources`](design.md#minimum-hardware) instead of fixed jobs.

## Open items

- [ ] **Maintainers:** share the write-up and record the answers to the five asks (set aside for now). Then plan the follow-ups, CI baseline recording first.
- [ ] **Re-vendor `utils/contracts/`** on `contracts-proof` whenever `example/run.py`, `select_contracts.py`, `report.py`, or the lock changes here; the branch README names the source commit.
  - Not yet vendored: MusicGen Melody, selection by sub-config model type, and Whisper's controlled continuation (new pin and baselines).
  - Keep the `proof-*` branches rebased on `contracts-proof`.
- [ ] **Refresh fixed bugs:** #49456 is fixed by #49466 on newer `main`; remove its known failure when rebasing the proof onto that fix. At the next pinned-build bump here, drop `config-field-generation-params` and `whisper-decode-tensor-batch` (#49403, fixed by #49419). Drop Wav2Vec2's patch when the pinned build includes #49386.
- [ ] **Cleanup after the proof:** delete the `proof-*` branches, and `contracts-proof` once the job moves to `transformers-ci` (D1b). Prune the worktrees that lived in session scratchpads with `git -C ~/Dev/transformers worktree prune`.
