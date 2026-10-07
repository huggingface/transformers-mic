# Problems and opened questions

**Slim models provide a fast regression check. A passing slim contract does not mean that the full model passes.** Contracts can replace model-specific integration tests in `run-slow` by executing against full checkpoints. The [CI integration](slim-models-in-transformers-ci.md) keeps the slim and full-model results separate and retains the other slow framework tests.

The [MIC proposal](../README.md) needs to establish how much useful early signal these fast checks provide and what that signal costs to maintain. The experiments support a CI pilot, but changing existing test coverage or making contracts required needs stronger evidence and stricter validation. These six issues distinguish limits of the approach from concrete gaps in the current prototype.

## Regression coverage beyond the replay cases

The [historical replay](evidence.md#historical-regression-replay) caught eight reproducible regressions using slim fixtures once the affected public calls, inputs, processor files, and runtime were included. That establishes that small fixtures can reproduce those failures. It does not establish a future detection rate for the contracts as currently written: generic checks missed most processor and tokenizer bugs, and the [current T5 contract](../example/README.md#runner-prototype) still misses the batch-decode regression because it never makes the affected call.

Choosing representative usage paths remains substantial test-authoring work. A short documentation example can be useful without covering the options and input combinations that existing integration tests exercise.

Freeze the contract code, inputs, fixtures, assertions, and tolerances before evaluating a set of regressions that was not used to design them. Continue the same measurement on future changes during the shadow trial. The [shadow backfill](evidence.md#shadow-backfill) does this on past history: frozen contracts on all 1,468 `main` commits from 2026-05-02 to 2026-10-07. Of the five regressions in the contracts' families during that time, the frozen suite caught one (UMT5, at its introducing commit) and missed four whose affected calls it does not make; it also flagged one benign change. That is the detection-rate problem stated above, measured: the misses are usage paths, not fixtures. Report detections, misses, unsupported cases, and false alarms separately; distinguish failures caught by the frozen suite from failures caught only after adding a reproduction.

Questions to resolve:

- How are representative inputs and API options chosen before a bug is known?
- What early detection must a contract demonstrate before it can replace a particular existing fast test?

## Boundaries of slim coverage

Calling the same public API does not ensure that a fixture and its checkpoint exercise the same implementation paths. Smaller dimensions and depth can change which conditions are reached; CPU execution excludes GPU paths. A fixture's default dtype can also differ from its checkpoint's: the [SmolLM example](../example/README.md#run-it) runs its fixtures in float32 while the 135M and 360M checkpoints default to bfloat16. The fast check therefore does not cover every path used by those checkpoints.

The fixture key establishes agreement on copied processor files and retained configuration. It does not establish equivalence for behavior affected by the dimensions that were replaced. Short inputs also leave long-context behavior untested, even when the fixture retains the relevant configuration.

These differences are an expected tradeoff of a fast check. The fixture does not need to reproduce all full-model behavior to be useful. For each contract, record the paths covered by the fixture, its known limits, and the full-model or hardware-specific checks that retain the remaining coverage. Choose PR variants for their early detection value and cost, and keep the required variants explicit.

Questions to resolve:

- Which affordable dtype, attention, or input variants add useful signal to the fast check?
- When should a change also trigger a full-model `run-slow` check before merge?

## What the input sensitivity gate establishes

The [gate](../example/perturb.py) establishes that perturbing an input changes some numerical output. It does not establish that a broken attention mask, detached module, or wrong preprocessing step will change the outputs the contract actually compares.

There is also a mismatch between the gate and the assertions. The gate measures the maximum absolute movement anywhere in the logits, while contracts may compare selected logits or label log-probabilities. A movement outside those selected outputs can pass the gate without improving the contract's detection. The [builder](../example/build_fixtures.py) derives its default threshold from the manifest's absolute tolerances; it does not incorporate relative tolerances or platform overrides into that calculation.

Evaluate perturbations through the contract's recorded outputs and comparison rules, with a margin above normal variation in the supported runtimes. Retain a small set of representative injected faults to check whether input sensitivity translates into regression detection. Overfitting remains a useful fallback to evaluate: its measured detection gain is currently on synthetic Whisper faults, while random fixtures caught all eight reproducible historical regressions too ([weight comparison](fixtures.md#choosing-the-weights)).

Questions to resolve:

- What sensitivity margin is sufficient for the outputs and tolerances a contract uses?
- Which faults should be tested beyond input perturbations, and when does overfitting add enough detection to justify its cost?

## Baseline reliability and maintenance

The baseline set grows with targets, frameworks, hardware classes, dtypes, and attention variants. CPU architecture differences have already required separate expectations for Whisper. The [A10G results](../example/README.md#on-gpu) establish stability on that tested GPU and environment, but do not establish that one `cuda` class will work across other GPUs and software stacks. The claim that platform-specific baselines eliminate the need for looser tolerances should remain scoped to the configurations measured so far.

Reproducing a candidate in a fresh process establishes repeatability. Approving a changed answer still requires evidence that the new behavior is acceptable. The [baseline review process](design.md#revisions-and-baselines) needs practical criteria for distinguishing an intended change, harmless numerical variation, and a regression; repeatedly recording new answers could otherwise hide failures.

Define the supported environment classes, calibrate them across representative runners, and measure baseline churn and review time during the pilot. So far: over five months of `main`, the eight `cpu-x86_64` fixture baselines needed one update for a benign change (a load-time RNG shift) and one for an intended one (T5-family SDPA), on GitHub runners with three EPYC models (7763, 9V45, 9V74), with and without native bfloat16; bfloat16 real checkpoints, by contrast, differed on a CPU with native bfloat16 support, which may need its own environment class. Budget the slim PR matrix separately from the full-model validation matrix; every full-model variant need not become a fast check. Each intentional baseline update should explain the behavior change and provide supporting evidence beyond the candidate's repeatability.

Questions to resolve:

- How narrowly must hardware and dependency environments be grouped for stable comparisons?
- What evidence is required to approve an output change, and what level of baseline upkeep is acceptable?

## Coordination and total maintenance cost

Keeping contracts on the Hub and expectations in framework repositories creates a review boundary. A change to both the calling code and framework behavior can require a contract revision, fleet-lock updates, and baseline changes across repositories. That arrangement needs a way to test the combined candidate before either part becomes the default, while preserving reproducibility for older framework revisions.

The [cost measurements](evidence.md#stability-and-cost) show much smaller fixture artifacts and why batching matters; reduced transfer at fleet scale remains to be measured end to end. The measurements provide less evidence about the cost of authoring contracts, fixing shrink rules, repairing fixtures, reviewing baselines, and coordinating updates. Moving this work does not by itself reduce it.

Exercise one complete update involving both a contract and framework code, including review and rollback. Track maintainer time alongside execution, setup, and download costs. Keep existing coverage until each proposed migration has a mapping of retained inputs, assertions, variants, hardware, and run frequency. The proposal to pause or retire smoke-only tests needs that comparison first. Moving a full-model integration test into a contract must retain its full-model execution; a slim pass cannot replace it.

Questions to resolve:

- How are candidate contract revisions and framework changes tested and reviewed together?
- Does the pilot reduce total maintainer effort, and which existing tests can be removed without losing coverage?

## Validation gaps in the prototype

The [runner](../example/run.py) does not yet enforce all the guarantees described in the proposal. The following cases can lose checks without producing a failing result:

| Gap | Current behavior | Required behavior before required CI |
| --- | --- | --- |
| Missing output fields | A declared numeric field missing from both the baseline and actual output is skipped. | Declare required outputs per framework and reject incomplete candidates and results. |
| Missing named gradient checks | An empty gradient dictionary passes `all_true`, even when the baseline contains named checks. | Require the expected module names and explicit successful assertions for each. |
| Changed output shapes | Numeric comparison flattens arrays, accepting different shapes with identical flattened values. | Validate output shapes as well as values. |
| Non-finite gradients | Contracts check that selected gradients exist and are nonzero, without checking finiteness. | Check gradient finiteness explicitly on the build under test. |
| Missing execution coverage | A run containing only `unsupported` results can exit successfully. | Enforce the expected set of targets and variants, with explicit permitted skips. |

The first three cases were reproduced directly against the comparator. The gradient gap is visible in the [Whisper contract](../example/hub/Whisper-integration/integrations/transformers/usage.md) and the other training contracts. The fixture builder's numeric-health gate checks finiteness when a fixture is built, but cannot detect a gradient regression introduced by a later framework build.

Define output requirements independently of whatever a candidate happens to emit. An inference-only framework can legitimately omit training outputs; a framework that declares training support must supply the required training checks. Likewise, insufficient hardware and an unsupported usage path should remain visible outcomes, while the CI policy decides whether either is allowed for a required target.

Questions to resolve:

- Where are required fields, shapes, and named assertions declared for each framework?
- Which targets and variants must run for a job to count as successful, and how are intentional exceptions reviewed?
