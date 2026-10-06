# MICs: next steps

Validate the [proposed standard](../README.md#the-proposed-standard) before expanding required CI or building Hub features. The decisions are whether slim fixtures catch real regressions, reduce total validation cost, and save maintainer time, and where overfitting earns its extra complexity.

## Status (2026-10-06)

Everything so far ran locally on CPU (ARM Mac, float32, Transformers `main` at `4d536e1f`). Results are in [Evidence](evidence.md) and [Small fixtures](fixtures.md).

| Step | State |
| --- | --- |
| Synthetic faults, random vs overfit | done: 42/42 on SmolLM and CLIP; 78/84 random vs 84/84 overfit on BERT, T5, Whisper, Wav2Vec2 (Whisper's random fixture ignores its audio) |
| 1. Historical regressions | done on CPU: 11 candidates, 8 reproducible, all 8 caught by slim fixtures through the affected public call with original processor files; 10 benign changes, no false alarms. GPU cases pending |
| 2. Stability and cost | CPU part done: no flags in 770 checks across torch 2.12–2.14, tokenizers, numpy 2.x, Pillow; per-check cost measured. CI platform, GPU, downloads, and setup time pending |
| 3. Contract runner | not started |
| 4. Run alongside CI | not started |
| 5. Hub surfaces | not started |

Side findings already acted on:

- [#49372](https://github.com/huggingface/transformers/issues/49372) (filed): `set_attn_implementation()` after loading silently leaves T5-family encoder/decoder on eager. Contracts and the experiment scripts choose attention at load time.
- The [agent walkthrough](agent-walkthrough.md) showed broken card snippets, no reference outputs, stale/contradictory metadata, and test fixtures leaking into Hub search; these are now cited in the proposal, evidence, fixtures, and design docs.

## What the replays changed

- **Generic checks are not enough.** Forward/generate on frozen inputs caught only the model-level UMT5 regression, and only under SDPA. The slim fixtures caught the processor and tokenizer bugs once the contract made the affected public call.
- **Copy processor files byte for byte.** Re-saving them hid three loading regressions ([#47628](https://github.com/huggingface/transformers/pull/47628), [#47848](https://github.com/huggingface/transformers/pull/47848), [#47988](https://github.com/huggingface/transformers/pull/47988)); with the original files copied, [#47628](https://github.com/huggingface/transformers/pull/47628) was caught on three families.
- **Run the default runtime configuration.** The UMT5 causal bug ([#49135](https://github.com/huggingface/transformers/pull/49135)) only exists under SDPA.
- **Pin the dependency environment, not just model revisions.** Builds before 2026-08-17 need `tokenizers` 0.22.x, and builds before mid-June an older `kernels`.
- **Organize around usage paths.** In-family regressions are rare in a five-month window; most bugs that matter sit in shared processor, tokenizer, and attention code.
- **Process start dominates check cost** (about 2.4 s of imports vs 0.1–0.9 s of work), so the runner must batch contracts per process. Savings come from transfer and model size, not local time on small checkpoints.

## Immediate next actions

1. **GPU runners.** Use the transformers ARC runner group `aws-g5-4xlarge-cache` (the one serge's GPU verify and run-slow use) from this repo's own workflows, with the `huggingface/transformers-all-latest-gpu` image and the `/mnt/cache` HF cache mount, as in transformers-ci's `serge-verify-slow.yml`. Infra is enabling this repo on that group; until then a job stays queued. `.github/workflows/gpu-probe.yml` (`gh workflow run gpu-probe.yml`) checks access and prints the GPU environment. Then: repeat the frozen checks and the stability matrix on one GPU (dtype and attention backend as users get them), calibrate tolerances on clean runs, and find at least one GPU-specific historical regression to replay. Until then, GPU results are missing, not passing.
2. **Finish step 1 gaps:** bugs found by nightly/slow tests (needs CI failure history), a larger benign set, and a real Whisper-like regression to test the overfit fallback.
3. **Step 3, runner prototype**, starting with the schema and one-command execution for the six families already covered.

## How to resume

Repository: https://github.com/huggingface/model-integration-contracts (private). Push over HTTPS; SSH is rejected for the org.

Environment: `~/Dev/transformers/.venv/bin/python` (Python 3.10, torch 2.11.0, tokenizers 0.23.1, numpy 1.26.4) with Transformers as an editable install of `~/Dev/transformers` at `4d536e1f`. Pinned checkpoints are in the local HF cache; `experiments/experiment.py` (`SOURCES`, `CLIP_REFERENCE`) and `experiments/usage_repros.py` (`EXTRA_SOURCES`) list them with revisions.

Scratch artifacts (frozen fixtures, extracted builds, dependency overlays, check outputs) lived in a session scratchpad and are gone. Regenerate them under `$REPLAY_WORKDIR` (default: `<tmp>/model-integration-replay`):

```sh
export REPLAY_WORKDIR=/tmp/model-integration-replay PY=~/Dev/transformers/.venv/bin/python
# 1. Freeze fixtures and real baselines on the known-good build (Transformers HEAD).
$PY experiments/replay.py freeze --families text clip bert t5 whisper wav2vec2 --real \
    --out $REPLAY_WORKDIR/frozen --build $(git -C ~/Dev/transformers rev-parse HEAD)
$PY experiments/replay.py freeze --families umt5 --real --out $REPLAY_WORKDIR/frozen_umt5 --build HEAD
$PY experiments/replay.py freeze --families olmo --out $REPLAY_WORKDIR/frozen_olmo --build HEAD
$PY experiments/usage_repros.py prepare          # byte-copied fixtures under $REPLAY_WORKDIR/copied
# 2. Replay one case (extracts <fix> and <fix>^ with git archive, never checks out).
ATTN=sdpa $PY experiments/replay_case.py <fix-sha>
FROZEN=$REPLAY_WORKDIR/frozen_umt5 ATTN=sdpa $PY experiments/replay_case.py d7ba937671
# 3. Usage-path repro on one build: real | slim (re-saved) | copied.
PYTHONPATH=$REPLAY_WORKDIR/builds/<sha>/src $PY experiments/usage_repros.py <case> real
# 4. Cost per check.
$PY experiments/cost.py --fixtures $REPLAY_WORKDIR/frozen --output experiments/results/cost.json
```

Every case, with fix/bad commits and verdicts, is in [cases.json](../experiments/cases.json).

Pitfalls seen on this machine:

- **Old builds:** pass an overlay on `PYTHONPATH` after the build's `src` (`OVERLAY=` for `replay_case.py`). Builds before 2026-08-17 need `tokenizers==0.22.2` (0.23.0 final was never published); builds before mid-June also need `kernels<0.13`. Install with `uv pip install --target <dir> --no-deps ...`.
- **torch overlays:** torchaudio stops at 2.11 on PyPI, so install only torch and torchvision. `uv` refuses the torch 2.14.1 macOS wheel (macOS 14 tag); unzip the wheel into the overlay directory instead.
- **Shell:** zsh does not word-split `$VAR` holding a command or a pair of versions; use functions or explicit arguments. Commands started with a plain `&` die with the tool's shell; use a tracked background task.
- **`rtk` hook:** it rewrites `curl`, `grep`, and `git` output; use `rtk proxy <cmd>` when exact output matters (JSON from `curl`, `git log -S`).
- **Attention:** never call `set_attn_implementation()` after loading for checks ([#49372](https://github.com/huggingface/transformers/issues/49372)); load with `attn_implementation=` instead.
- **Partial HF cache:** `snapshot()` falls back to the cached folder when only some files of a revision were downloaded.

## 1. Reproduce historical regressions

- [x] Select a small, documented set: 11 regressions fixed since May 2026 and 10 benign changes, covering tokenization, processors, attention masks, loading, and decoding. No GPU-specific case yet.
- [x] Record each case's PR/issue, fix, bad and introducing commits, call path, and verdicts ([cases.json](../experiments/cases.json)).
- [x] Reproduce each failure on the real checkpoint first; three candidates did not reproduce on standard checkpoints and are reported as such.
- [x] Freeze paired random/overfit fixtures, inputs, assertions, and tolerances on the known-good build before testing bad builds.
- [ ] Include a real regression on a fixture that fails the input-sensitivity gate (Whisper or another encoder-decoder audio/vision model), so the overfit fallback is tested on a real bug.
- [x] Record detections by assertion, misses, false alarms, and unsupported cases.
- [ ] Add GPU-specific regressions and bugs found by nightly/slow tests.

**Decision:** identify the paths slim CI actually covers. Keep GPU-only, full-scale, or otherwise missed cases in their existing jobs. Use overfitting where it adds detection or demonstrably improves diagnosis.

## 2. Measure stability and total cost

- [x] Run the frozen fixtures against benign Transformers changes and dependency upgrades (torch 2.12.1/2.13.0/2.14.1, tokenizers 0.23.2, numpy 2.2.6, Pillow 12.3.0; eager and SDPA): no flags in 770 checks ([stability.json](../experiments/results/stability.json)).
- [ ] Repeat on the actual CI CPU platform, then one supported GPU with its dtype and attention backend. Calibrate tolerances on clean runs and freeze them; keep hardware-specific baselines where needed. Real-checkpoint drift already reaches 8.4e-5 against a 1e-4 tolerance on CPU.
- [ ] Measure cold and warm execution including environment setup, Hub downloads, and a purged OS file cache. Process start, loading, inference, training probe, comparison, memory, and artifact size are measured locally ([cost.json](../experiments/results/cost.json)).
- [ ] Compare against current CI in wall time, CPU/GPU time, and transfer. So far only transfer: the slow tests for these eight families load about 144 GB of weights. Track fixture creation, repair, review, and triage time separately.

**Deliverable:** cost and stability report with sample counts, failures, baseline churn, and manual interventions. Agree on acceptable runtime and noise with maintainers before enabling a required check.

## 3. Build the smallest usable contract runner

Start from the six experiment families. Add one coupled-config or MoE model and one GPU-heavy or costly-to-maintain case; roles may overlap within the 3–5-model pilot. Assign maintainers before onboarding.

- [ ] Define schemas for the manifest, fleet lock, required output fields, expected shapes, baseline identity, and result record.
- [ ] Implement one-command execution with pinned model, processor, artifact, contract, fixture, baseline, and dependency inputs; run many contracts per process. Reject missing/non-finite outputs and unsupported variants explicitly.
- [ ] Add the Python adapter, then one Transformers.js case with a pinned ONNX artifact and conversion provenance. Keep training checks optional for inference-only frameworks.
- [ ] Build fixtures by copying processor, tokenizer, and preprocessing files unchanged; shrink only weights and model config. Mark published fixtures as test artifacts and keep them out of Hub search.
- [ ] Record baselines under the default attention implementation and dtype, chosen at load time; keep eager as an explicit variant.
- [ ] Implement the input-sensitivity gate at fixture build time, falling back to overfitting; report non-convergence as a build failure.
- [ ] Implement baseline creation and PR review without automatically accepting new outputs. Separate model-update discovery from the pinned inputs used by required framework checks.
- [ ] Define how changed framework files select relevant contracts, with a conservative fallback for shared code. Apply execution isolation and private-artifact rules.
- [ ] Reproduce the published usage entry point, including pipeline dispatch where the snippet uses it. Exercise private build → upstream PR → main once.

**Deliverable:** reviewed local commands and CI artifacts for each pilot contract, including a failure maintainers can reproduce from its recorded inputs.

## 4. Run alongside existing CI, then decide

- [ ] Begin with non-blocking PR runs, pinned real-checkpoint runs after merges and nightly, and a separate model-update job. Keep slim and real results distinct.
- [ ] Observe several weeks of framework and model changes, including at least one dependency update. Record regressions caught before nightly, misses, false alarms, time to fix, baseline updates, fixture repairs, and maintainer effort.
- [ ] Promote only stable contracts to required PR checks, with a named owner and a reported process for temporarily disabling a broken fixture.
- [ ] Map every proposed test migration to retained inputs, assertions, variants, hardware, and run frequency.

**Decision:** expand only with evidence of useful real-bug signal, acceptable CI cost and noise, and manageable upkeep.

## 5. Surface proven results on the Hub

After CI is independently useful, scope the Tests-tab prototype, companion-repository links, contract/result metadata and APIs, and validated "Use this model"/Colab snippets. Expose model/framework/build revisions, real-versus-slim coverage, and unvalidated results. Re-run the [agent walkthrough](agent-walkthrough.md) prompt to check whether agents find and run the tested example with less adaptation.
