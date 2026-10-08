# Problems and opened questions

**A slim pass does not mean the full model passes.** Contracts replace model-specific `run-slow` tests only by running on full checkpoints; other slow tests stay ([CI integration](slim-models-in-transformers-ci.md)).

The experiments support a CI pilot. Removing tests or making contracts required needs more evidence. Six problems stand between the two:

| # | Problem | Kind |
| --- | --- | --- |
| 1 | [Contracts only catch what they call](#1-contracts-only-catch-what-they-call) | Limit of the approach |
| 2 | [A fixture does not cover every path of its checkpoint](#2-a-fixture-does-not-cover-every-path-of-its-checkpoint) | Limit of the approach |
| 3 | [The sensitivity gate checks the wrong thing](#3-the-sensitivity-gate-checks-the-wrong-thing) | Prototype gap |
| 4 | [Baselines multiply and need review rules](#4-baselines-multiply-and-need-review-rules) | Open cost |
| 5 | [Cross-repository coordination and total cost](#5-cross-repository-coordination-and-total-cost) | Open cost |
| 6 | [Runner validation gaps](#6-runner-validation-gaps) | Prototype gap, closed |

## 1. Contracts only catch what they call

**Problem.** Fixtures reproduce real regressions only when the contract makes the affected call. The detection rate on unseen bugs is unknown.

**What we know.**

- [Replay](evidence.md#historical-regression-replay): all eight reproducible regressions caught once the affected call was made; generic forward/generate checks missed most processor and tokenizer bugs.
- [Shadow backfill](evidence.md#shadow-backfill), 1,468 `main` commits: frozen contracts caught 1 of 5 regressions in their families (UMT5), missed 4 they didn't call, flagged 1 benign change.
- With the 4 calls added: 5 of 5, no new false alarm. **Post-hoc:** written knowing the bugs.

**Next.** Freeze contracts, inputs, fixtures, and tolerances; measure on regressions not used to design them (the shadow trial does this). Report detections, misses, unsupported, and false alarms separately, frozen-suite catches apart from post-hoc ones ([improvement loop](improving-contracts.md)).

**Open questions.**

- How are representative inputs and API options chosen before a bug is known?
- What must a contract demonstrate to replace a given fast test?

## 2. A fixture does not cover every path of its checkpoint

**Problem.** Same API, different code paths: smaller dimensions, short inputs, CPU, and a different default dtype each skip paths the checkpoint takes.

**What we know.**

- The fixture key guarantees identical processor files and retained config, not behavior depending on shrunk dimensions.
- CPU excludes GPU paths; short inputs leave long context untested.
- [SmolLM fixtures](../example/README.md#run-it) run float32; the 135M and 360M checkpoints default to bfloat16.

This is the expected tradeoff of a fast check.

**Next.** Per contract, record what the fixture covers, its limits, and which full-model or hardware job covers the rest. Choose PR variants explicitly, for cost versus signal.

**Open questions.**

- Which cheap dtype, attention, or input variants add signal?
- When should a PR also trigger a full-model `run-slow` before merge?

## 3. The sensitivity gate checks the wrong thing

**Problem.** The [gate](../example/perturb.py) shows a perturbed input moves *some* logit, not that a broken mask, detached module, or bad preprocessing moves the outputs the contract *compares*.

**What we know.**

- It measures max movement over all logits; contracts may compare only selected logits or label log-probs.
- The [builder](../example/build_fixtures.py)'s threshold uses absolute tolerances only, not relative ones or platform overrides.
- Overfitting helps on synthetic Whisper faults, but random fixtures caught all eight historical regressions too ([weights](fixtures.md#choosing-the-weights)).

**Next.** Measure perturbations through the contract's own outputs and rules, with a margin above normal variation. Keep a few injected faults to check sensitivity turns into detection.

**Open questions.**

- What margin suffices for given outputs and tolerances?
- Which faults beyond perturbations should be tested, and when is overfitting worth it?

## 4. Baselines multiply and need review rules

**Problem.** Baselines grow with targets × frameworks × hardware × dtype × attention. Re-recording is easy, so without review criteria a regression can be approved as a new answer.

**What we know.**

- Whisper already needs per-CPU-architecture baselines.
- [A10G results](../example/README.md#on-gpu) hold for that GPU and stack only; "per-platform baselines remove looser tolerances" is scoped to what was measured.
- Five months of `main`: 8 `cpu-x86_64` fixture baselines needed 2 updates (benign RNG shift, intended T5 SDPA), across three EPYC models with and without native bfloat16.
- bfloat16 *real* checkpoints differed on a native-bfloat16 CPU; may need its own environment class.
- Fresh-process reproduction proves repeatability, not correctness.

**Next.** Define environment classes, calibrate on representative runners, track churn and review time. Budget the slim PR matrix apart from the full-model one. Each baseline update explains the change with evidence beyond repeatability ([review process](design.md#revisions-and-baselines)).

**Open questions.**

- How narrowly must hardware and dependency environments be grouped?
- What evidence approves an output change, and how much upkeep is acceptable?

## 5. Cross-repository coordination and total cost

**Problem.** Contracts on the Hub, expectations in framework repos: one behavior change can need a contract revision, a fleet-lock bump, and baselines across repos, tested together and reproducible for old revisions.

**What we know.**

- [Cost measurements](evidence.md#stability-and-cost) show much smaller fixtures and why batching matters; fleet-scale transfer isn't measured end to end.
- Human cost (authoring, shrink rules, fixture repair, baseline review, coordination) isn't measured. Moving work doesn't reduce it.

**Next.**

- Run one full contract + framework update, with review and rollback.
- Track maintainer time next to execution, setup, and download cost.
- Before retiring a test (smoke-only ones included), map its inputs, assertions, variants, hardware, and frequency to the replacement.
- A full-model test moved into a contract keeps its full-model run.

**Open questions.**

- How are contract revisions and framework changes tested and reviewed together?
- Does the pilot reduce total maintainer effort, and which tests can go?

## 6. Runner validation gaps

**Closed 2026-10-08.** The [runner](../example/run.py) dropped checks silently in five ways; each was reproduced before the fix and fails after.

| Gap | Before | Now |
| --- | --- | --- |
| Missing output | Skipped if absent from result and baseline. | Every manifest output required (training ones only where the framework declares `training`); `--record` refuses incomplete candidates. |
| Empty gradient checks | `{}` passed `all_true`. | Non-empty dict, names matching the baseline, each `true`. |
| Changed shape | Flattened: (1, 4) matched (2, 2). | `allclose` outputs must match the baseline's shape, ragged lists included. |
| NaN gradients | NaN counted as nonzero. | Six training contracts also require `torch.isfinite(grad).all()`. |
| Nothing ran | All-`unsupported` runs exited 0. | Skips record why (`declared`, `hardware`, `variant`). No target run or no `--contract`/`--target` match fails. `--require-all` (shadow, backfill, nightly) fails on undeclared skips. |

Required outputs come from the manifest, shapes from the reviewed baseline; declared skips are reviewed with the contract.

**Still open.**

- GitHub runners (7 GB) skip SmolLM-1.7B when recording, so only `machine=gpu-host` enforces `REQUIRE_ALL`; choosing runners from `resources` (next action 8) fixes it.
- No per-framework exception list for non-training outputs; add one when needed.
