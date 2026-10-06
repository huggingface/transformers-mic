# Model integration contracts: next steps

Validate the [proposal](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts-md) before expanding
required CI or building Hub features. The decisions are whether slim
fixtures catch real regressions, reduce total validation cost, and save
maintainer time—and where overfitting earns its extra complexity.

## Starting point

The [local experiment](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_experiment-py) compared
SmolLM and CLIP with three initialization seeds. Frozen random and overfit
fixtures both detected 42/42 synthetic fault/seed cases using logits,
loss, and targeted training checks; 60 controls produced no false alarms.
[Results](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_results-json) include provenance.

A second run (2026-10-06) added BERT, T5, Whisper, and Wav2Vec2 from
[PR #49103](https://github.com/huggingface/transformers/pull/49103)'s
list: 84 fault/seed cases per policy, 120 controls, no false alarms.
Random fixtures detected 78/84 and overfit 84/84. All six misses are
Whisper (reversed audio, shuffled labels, every seed): the tiny random
decoder barely depends on its audio, so logits moved below tolerance.
Overfit Whisper training took 23–51 s; one Wav2Vec2 seed did not
converge in 500 steps. [Results](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_results_more-json)
use the same script, commit, and tolerances.

This supports starting with frozen random fixtures **gated by input
sensitivity**, and overfitting fixtures that fail the gate. It has not
measured historical bug coverage, dependency-upgrade stability, GPU
behavior, or ongoing maintenance. Keep existing tests throughout the
pilot.

## Done

- [x] Random vs overfit on SmolLM and CLIP: 42/42 each.
- [x] Extend to BERT, T5, Whisper, Wav2Vec2: random 78/84, overfit 84/84;
  random misses isolated to Whisper input/label faults.
- [x] Add the input-sensitivity gate and "overfit on failure" rule to the
  proposal's fixture plan and decision.
- [x] First historical replay (2026-10-06): 7 regressions fixed since May
  2026 and 10 benign changes, frozen on `4d536e1f`, checked on bad and
  fixed builds with [freeze/check](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_replay-py),
  the [per-case driver](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_replay_case-py), and
  [usage repros](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_usage_repros-py). Results are
  in the technical design's "Historical regression replay" section.

## What the first replay changed

- Generic forward/generate fixtures caught only the model-level UMT5
  regression, and only under SDPA; the same slim fixtures called through
  the affected public entry point caught the processor/tokenizer bugs
  that did not depend on processor packaging.
- Fixture builders must copy processor/tokenizer files byte for byte:
  re-saving hid two tokenizer-loading regressions ([#47848](https://github.com/huggingface/transformers/pull/47848), [#47988](https://github.com/huggingface/transformers/pull/47988)).
- Contracts must run the default runtime configuration: the UMT5 causal
  bug ([#49135](https://github.com/huggingface/transformers/pull/49135)) only shows under SDPA. Choose attention at load time;
  `set_attn_implementation()` after loading silently skips the T5-family
  encoder/decoder on current `main` (not yet reported upstream).
- Builds before 2026-08-17 need `tokenizers` 0.22.x (and older `kernels`
  before mid-June); fleet locks must pin the dependency environment.
- In-family regressions are rare in a five-month window; most relevant
  bugs sit in shared processor, tokenizer, and attention code.

## 1. Reproduce historical regressions

- [ ] Select a small, documented set, initially targeting 10 regressions
  and 10 benign changes. Cover preprocessing/tokenization, attention or
  position handling, generation/cache behavior, loss or gradients, and
  published usage entry points. Include at least one GPU-specific case.
- [ ] Record each issue/PR, failing and fixed commits, checkpoint and
  processor revisions, environment, input, expected behavior, and the
  existing test that establishes the regression.
- [ ] Reproduce failure on the real checkpoint before evaluating slim
  detection. Prefer the original commits. If an old environment cannot
  be reconstructed, report that blocker; a selective reversion on current
  code is a separate reconstructed case, not an exact historical replay.
- [ ] Freeze paired random/overfit fixtures, inputs, assertion policies,
  and tolerances on the known-good build before testing the bad build.
  Use the same numerical and training assertions for both weight types.
- [ ] Include Whisper or another encoder-decoder audio/vision model whose
  random fixture fails the input-sensitivity gate, so a real regression
  tests the overfit fallback rather than only synthetic faults.
- [ ] Record detections by assertion, misses, false alarms, and cases
  unsupported by the fixture. Any tuning after seeing a miss creates a
  follow-up fixture version; preserve the original result.

**Deliverable:** a case manifest and a real-versus-slim result matrix.
Count distinct regressions separately from repeated seeds. Do not drop
unreproducible cases from the report or count them as successful checks.

Remaining for this step: GPU-specific cases, bugs found by nightly/slow
tests, a larger benign set, and the four candidates not yet replayed
([#47628](https://github.com/huggingface/transformers/pull/47628) config helpers, [#47440](https://github.com/huggingface/transformers/pull/47440) MMS `set_target_lang`, [#46406](https://github.com/huggingface/transformers/pull/46406) CLIP
conversion mapping, [#48785](https://github.com/huggingface/transformers/pull/48785) BERT-variant CLM masks).

**Decision:** identify the paths slim CI actually covers. Keep GPU-only,
full-scale, or otherwise missed cases in their appropriate existing jobs.
Use overfitting where it adds detection or demonstrably improves diagnosis.

## 2. Measure stability and total cost

- [ ] Run the frozen fixtures against documented benign changes and
  supported dependency upgrades. Verify benign behavior independently
  with the full checkpoint or existing tests; a changed snapshot alone
  does not establish a regression.
- [ ] Repeat on the actual CI CPU platform, then one supported GPU with
  its relevant dtype and attention backend. Declare unsupported variants
  explicitly. Calibrate tolerances on clean runs, then freeze them before
  evaluating faults; retain hardware-specific baselines where needed.
- [ ] Measure both cold and warm execution: environment setup, downloads,
  processor/model loading, inference, training checks, comparison, and
  reporting. Include fixture generation/export time and artifact size.
- [ ] Compare the same coverage against current CI using wall time,
  CPU/GPU time, and transfer/storage volume. Track fixture creation,
  repair, expectation review, and failure-triage time separately.

**Deliverable:** cost and stability report with sample counts, failures,
baseline churn, and manual interventions. Agree on acceptable job runtime
and noise with maintainers before enabling a required check. Local warm
latency and zero failures in a small sample are insufficient rollout gates.

## 3. Build the smallest usable contract runner

Start from the six existing experiment families. Add one coupled-config or
MoE model and one GPU-heavy or costly-to-maintain case; roles may overlap
within the proposed 3–5-model pilot. Assign maintainers before onboarding.

- [ ] Define schemas for the manifest, fleet lock, required output fields,
  expected shapes, baseline identity, and result record.
- [ ] Implement one-command execution with pinned model, processor,
  artifact, contract, fixture, baseline, and dependency inputs. Reject
  missing/non-finite outputs and unsupported variants explicitly.
- [ ] Add the Python adapter, then one Transformers.js case with a pinned
  ONNX artifact and conversion provenance. Keep training checks optional
  for inference-only frameworks.
- [ ] Copy processor, tokenizer, and preprocessing files from the real
  model unchanged; shrink only weights and model config.
- [ ] Record baselines under the default attention implementation and
  dtype; keep eager as an explicit variant.
- [ ] Implement the input-sensitivity gate at fixture build time: swap
  or perturb each input, require recorded logits to move well beyond
  tolerance, and fall back to overfitting (recording steps and
  convergence) when they do not. Report non-convergence as a build
  failure, not a skipped check.
- [ ] Implement baseline creation and PR review without automatically
  accepting new outputs. Separate model-update discovery from the pinned
  inputs used by required framework checks.
- [ ] Define how changed framework files select relevant contracts; use
  a conservative fallback for shared code changes. Apply the proposal's
  execution isolation and private-artifact access rules.
- [ ] Reproduce the published usage entry point, including pipeline
  dispatch where the snippet uses it. Exercise private build → upstream
  PR → main once; do not label private-build success as public support.

**Deliverable:** reviewed local commands and CI artifacts for each pilot
contract, including a failure that maintainers can reproduce from its
recorded inputs. An unknown baseline must remain unvalidated.

## 4. Run alongside existing CI, then decide

- [ ] Begin with non-blocking PR runs, pinned real-checkpoint runs after
  merges/nightly, and a separate model-update job. Send failures through
  existing maintainer reports. Keep slim and real results distinct.
- [ ] Observe several weeks of framework and model changes. Record
  regressions caught before nightly, misses, false alarms, time to fix,
  baseline updates, fixture repairs, and maintainer effort. Include at
  least one dependency update; elapsed time alone is not evidence.
- [ ] Promote only stable contracts to required PR checks. Keep a named
  owner and an explicit, reported process for temporarily disabling a
  broken fixture; avoid silent skips.
- [ ] Map every proposed test migration to retained inputs, assertions,
  variants, hardware, and run frequency. Preserve framework-behavior
  tests and any coverage the contract does not reproduce. Attach the
  scanner, source commit, and classifications behind migration estimates.

**Decision:** expand only with evidence of useful real-bug signal,
acceptable CI cost/noise, and manageable upkeep. Otherwise narrow the
supported fleet, improve assertions, or retain the current tests. Compare
random and overfit policies on those same outcomes.

## 5. Surface proven results on the Hub

After CI is independently useful, scope the Tests-tab prototype,
companion-repository links, contract/result metadata and APIs, and
validated “Use this model”/Colab snippets. Expose model/framework/build
revisions, real-versus-slim coverage, and unvalidated or unavailable
results. Evaluate whether users and agents can discover and run the
tested example with less adaptation. Co-owned hosting and remote-code
models remain later extensions.

## Immediate next action

Build the historical-case manifest and reproduce the first text and
processor-heavy regressions on their real checkpoints. That provides
the next meaningful evidence before more runner or Hub implementation.
