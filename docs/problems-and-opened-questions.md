# Problems and opened questions

Slim contracts are a fast regression signal. **A slim pass does not mean the full model passes.** Contracts can replace model-specific integration tests in `run-slow` only by running against full checkpoints; the other slow tests stay ([CI integration](slim-models-in-transformers-ci.md)).

The experiments support a CI pilot. Removing existing tests or making contracts required needs more evidence. This page lists the six problems that stand between the two, and what would settle each.

| # | Problem | Kind |
| --- | --- | --- |
| 1 | [Contracts only catch what they call](#1-contracts-only-catch-what-they-call) | Limit of the approach |
| 2 | [A fixture does not cover every path of its checkpoint](#2-a-fixture-does-not-cover-every-path-of-its-checkpoint) | Limit of the approach |
| 3 | [The sensitivity gate checks the wrong thing](#3-the-sensitivity-gate-checks-the-wrong-thing) | Prototype gap |
| 4 | [Baselines multiply and need review rules](#4-baselines-multiply-and-need-review-rules) | Open cost |
| 5 | [Cross-repository coordination and total cost](#5-cross-repository-coordination-and-total-cost) | Open cost |
| 6 | [Runner validation gaps](#6-runner-validation-gaps) | Prototype gap, closed |

## 1. Contracts only catch what they call

**Problem.** Small fixtures can reproduce real regressions, but only when the contract makes the affected call. We do not know the detection rate on bugs nobody has seen yet.

**What we know.**

- The [replay](evidence.md#historical-regression-replay) caught all eight reproducible regressions once the check made the affected call. Generic forward/generate checks missed most processor and tokenizer bugs.
- The [shadow backfill](evidence.md#shadow-backfill) ran the frozen contracts on 1,468 `main` commits (five months). Of five regressions in their families, it caught one (UMT5) and missed four whose calls no contract made. One benign change was flagged.
- With the four calls added, it catches all five and no new false alarm. **This is post-hoc:** the checks were written knowing the bugs.

**Next.**

- Freeze contracts, inputs, fixtures, and tolerances, then measure on regressions not used to design them. The shadow trial does this going forward.
- Report detections, misses, unsupported cases, and false alarms separately. Count catches by the frozen suite apart from catches after adding a call ([improvement loop](improving-contracts.md)).

**Open questions.**

- How do we choose representative inputs and API options before a bug is known?
- What must a contract demonstrate before it replaces a given fast test?

## 2. A fixture does not cover every path of its checkpoint

**Problem.** Same public API, different code paths. Smaller dimensions, shorter inputs, CPU, and a different default dtype can each skip paths the real checkpoint takes.

**What we know.**

- The fixture key guarantees identical processor files and retained config fields. It says nothing about behavior that depends on the shrunk dimensions.
- CPU runs exclude GPU paths. Short inputs leave long context untested.
- Dtypes can differ: the [SmolLM fixtures](../example/README.md#run-it) run in float32, the 135M and 360M checkpoints default to bfloat16.

This is the expected tradeoff of a fast check, not a defect.

**Next.** For each contract, record what the fixture covers, its known limits, and which full-model or hardware job covers the rest. Keep the required PR variants explicit and chosen for cost versus signal.

**Open questions.**

- Which cheap dtype, attention, or input variants add useful signal?
- When should a PR also trigger a full-model `run-slow` check before merge?

## 3. The sensitivity gate checks the wrong thing

**Problem.** The [gate](../example/perturb.py) shows that perturbing an input moves *some* logit. It does not show that a broken mask, detached module, or wrong preprocessing step moves the outputs the contract *compares*.

**What we know.**

- The gate measures max movement over all logits; contracts may compare only selected logits or label log-probabilities.
- The [builder](../example/build_fixtures.py) derives its threshold from absolute tolerances only, ignoring relative tolerances and platform overrides.
- Overfitting helps on synthetic Whisper faults, but random fixtures also caught all eight historical regressions ([weights comparison](fixtures.md#choosing-the-weights)).

**Next.** Measure perturbations through the contract's own outputs and comparison rules, with a margin above normal variation. Keep a small set of injected faults to check that sensitivity turns into detection.

**Open questions.**

- What margin is enough for a given set of outputs and tolerances?
- Which faults beyond input perturbations should be tested, and when is overfitting worth its cost?

## 4. Baselines multiply and need review rules

**Problem.** Baselines grow with targets × frameworks × hardware × dtype × attention. And re-recording is easy: without review criteria, a regression can be approved as a new answer.

**What we know.**

- Whisper already needs separate CPU-architecture baselines.
- The [A10G results](../example/README.md#on-gpu) are stable on that GPU and stack only. "Per-platform baselines remove the need for looser tolerances" holds for the configurations measured, no further.
- Five months of `main`: the eight `cpu-x86_64` fixture baselines needed two updates, one benign (load-time RNG shift), one intended (T5 SDPA). This held across three EPYC models, with and without native bfloat16.
- bfloat16 *real* checkpoints did differ on a CPU with native bfloat16; that may need its own environment class.
- A candidate reproducing in a fresh process proves repeatability, not correctness.

**Next.**

- Define environment classes, calibrate them on representative runners, and track churn and review time during the pilot.
- Budget the slim PR matrix separately from the full-model matrix.
- Require each baseline update to explain the behavior change, with evidence beyond repeatability ([review process](design.md#revisions-and-baselines)).

**Open questions.**

- How narrowly must hardware and dependency environments be grouped?
- What evidence approves an output change, and how much upkeep is acceptable?

## 5. Cross-repository coordination and total cost

**Problem.** Contracts live on the Hub, expectations in framework repos. One behavior change can need a contract revision, a fleet-lock bump, and baseline updates across repos, tested together and still reproducible for old revisions.

**What we know.**

- [Cost measurements](evidence.md#stability-and-cost) show much smaller fixtures and why batching matters. Fleet-scale transfer savings are not measured end to end.
- Human costs are not measured: authoring contracts, fixing shrink rules, repairing fixtures, reviewing baselines, coordinating updates. Moving this work does not reduce it.

**Next.**

- Run one complete contract + framework update, including review and rollback.
- Track maintainer time next to execution, setup, and download cost.
- Before retiring any test, map what it covers (inputs, assertions, variants, hardware, frequency) to what replaces it. Smoke-only tests need this before being paused.
- A full-model integration test moved into a contract keeps its full-model run.

**Open questions.**

- How are a contract revision and a framework change tested and reviewed together?
- Does the pilot reduce total maintainer effort, and which tests can go without losing coverage?

## 6. Runner validation gaps

**Closed 2026-10-08.** The [runner](../example/run.py) used to drop checks silently in five ways. Each was reproduced before the fix and fails after it.

| Gap | Before | Now |
| --- | --- | --- |
| Missing output | Skipped if absent from both result and baseline. | Every manifest output is required (training outputs only where the framework declares `training`). `--record` refuses incomplete candidates. |
| Empty gradient checks | `{}` passed `all_true`. | Must be a non-empty dict, names matching the baseline, each `true`. |
| Changed shape | Flattened: (1, 4) matched (2, 2). | `allclose` outputs must match the baseline's shape first, ragged lists included. |
| NaN gradients | NaN counts as nonzero. | The six training contracts also require `torch.isfinite(grad).all()`. |
| Nothing ran | All-`unsupported` runs exited 0. | Each skip records why: `declared`, `hardware`, or `variant`. No target run, or no `--contract`/`--target` match, fails. `--require-all` fails on any undeclared skip; shadow, backfill, and nightly use it. |

The manifest's `outputs` define what is required, shapes come from the reviewed baseline, and declared skips are reviewed with the contract.

**Still open.**

- GitHub runners (7 GB) skip SmolLM-1.7B when recording baselines, so only `machine=gpu-host` enforces `REQUIRE_ALL`. Choosing runners from `resources` (next action 8) removes the exception.
- No per-framework exception list for outputs other than training. Add one when a case appears.
