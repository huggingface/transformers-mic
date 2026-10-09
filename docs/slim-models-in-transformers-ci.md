# Slim models in Transformers development and CI

Slim fixtures give fast PR checks; the same contracts run on full checkpoints in `run-slow` and scheduled jobs. **A slim pass does not mean the full model passes.**

- **Developers gain** an automatic check and a reproducible failure report.
- **Contract authors and maintainers pay** for fixtures, baseline review, and cross-repo coordination.

MICs run today only as a separate prototype with a shadow job, outside Transformers' PR jobs and `run-slow`. The rest is proposed.

## What contracts replace in `run-slow`

Model-specific integration tests (load a checkpoint, call a public path, check outputs) become contracts: `run-slow` runs them on the full checkpoint, fast PR CI on a slim fixture, each against its own baseline.

| Part of the flow | Effect |
| --- | --- |
| Fast PR checks | Gain slim runs of selected contracts. |
| Model-specific tests in `run-slow` | Become full-checkpoint contracts once coverage is verified equivalent. |
| `run-slow` trigger and orchestration | Unchanged. |
| Other slow tests (cache, compilation, batching, offloading, multi-GPU) | Stay in Transformers. |

The pilot runs tests and contracts side by side. A test is removed only once its contract keeps its inputs, assertions, variants, hardware, and run frequency; a slim-only check never justifies removal.

## Impact on development work

| Developer task | Today | Added by MICs |
| --- | --- | --- |
| Fix code, behavior unchanged | Change code and tests; request slow tests if needed. | Read slim results. No rebuild, no Hub edit. |
| Intentionally change behavior | Update expectations, explain in the PR. | Review candidate baselines in the same PR, with separate full-model evidence. |
| Add a model or usage path | Add support and fast/slow tests. | Add or extend a Hub contract; reuse or build fixtures; add fleet-lock entries and reviewed baselines. |
| Change a migrated test's inputs, calls, or variants | Edit the test. | Edit the Hub contract, with matching fleet-lock and baseline updates. |
| Change a checkpoint or rebuild a fixture | Update references and expectations. | Validate the candidate, then explicitly update pin and baselines; required checks keep old pins until reviewed. |

Ownership ([details](design.md#ownership)): framework maintainers own the fleet lock and expected results; Hugging Face owns integration and fixture repos; creators keep their repos and run no framework CI. Cross-repo changes need an agreed review flow.

## An ordinary framework PR

1. The developer changes code and tests.
2. Fast CI selects the relevant MICs, fetches each pinned fixture once, runs them on the PR build, and compares with committed baselines, never rebuilding or training fixtures.
3. The report names contract, fixture, variant, failure, and repro command.
4. For full-model validation, maintainers request `run-slow`: migrated tests run as contracts on real checkpoints, reported apart from slim results.
5. Merge follows the repo's requirements. A slim pass never satisfies a pending full-model requirement.

The prototype [selector](../example/select_contracts.py) maps model directories to contracts (a UMT5 change selects the T5 contract) and selects all of them for shared code or dependency changes. The shadow job runs everything and records the selection, measuring missed dependencies first.

## Handling a contract failure

Reproduce on the reported target: pinned fixture for slim, checkpoint and baseline for full. A fixture failure is a signal on its own and can justify a full-model run.

- **Unintended:** fix the framework or the contract.
- **Intended:** record a candidate, explain why the new output is acceptable, review it in the PR. Approving a fixture answer never approves a full-model one.

`--record` writes only candidates that reproduce in a fresh process; required CI compares with reviewed baselines, never the build under test. Missing artifacts, missing baselines, insufficient resources, and unsupported variants are reported apart from regressions ([validation gaps](problems-and-opened-questions.md#6-runner-validation-gaps)).

## Adding or changing a contract

1. Write the usage example and manifest in the integration repo: checkpoints, variants, outputs, resources.
2. Reuse a fixture with the same key, or build, validate, and publish one at a pinned revision (one-time).
3. Record fixture and checkpoint candidates on a known-good build, every supported runtime; review before approval.
4. Add pins, baselines, and selection info to the framework's fleet. Run alongside existing tests during the pilot.

**Contract and framework changing together.** Proposed: publish an immutable candidate contract revision, test it with the framework PR and candidate baselines, then bump the fleet lock through review. Old commits keep their old lock. Untested end to end: the contracts are published and pinned, but no framework PR has gone through this flow yet.

**Creator updates a model.** This must not silently change every required check's input. A flow (not built) should test the revision, check it still shares the fixture, and propose pin and baseline updates for review.

## Fast PR checks and full model checks

Transformers separates [PR CI](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml) from slow checks: `@slow` tests run only with `RUN_SLOW` ([testing utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py)). A maintainer comments:

```text
run-slow: llama, clip
```

and the [comment workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-comment-ci.yml) runs those families' slow tests, with full checkpoints, on the PR merge commit before merge (e.g. [Llama](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py)).

| When | Contract target | Purpose |
| --- | --- | --- |
| Fast PR checks | Pinned slim fixtures, CPU | Cheap early feedback. |
| `run-slow` on a PR | Selected full checkpoints, suitable hardware | Validate real models before merge. |
| After merge | Affected full checkpoints | Check the merged build. |
| Nightly | Full-checkpoint fleet, explicit exceptions | Wider than one PR selects. |
| Release candidate | Full checkpoints, required variants | The release's tested compatibility. |

The [scheduled Nvidia workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-scheduled-caller.yml) already runs broader model CI. Fleet, selection, and release integration are pilot work ([design](design.md#what-runs-in-ci)).

## One contract with separate targets and expectations

One usage path, several model files, one baseline per target. The [SmolLM contract](../example/hub/SmolLM-integration/integration.json) covers three checkpoints with two fixtures:

- **Fast:** `proto-SmolLM-slim` for the shared 135M/360M path; `proto-SmolLM-1.7B-slim` for 1.7B's distinct config and tokenizer.
- **Full:** SmolLM-135M, -360M, -1.7B, each with its own outputs. A fixture pass marks no checkpoint as passing.

Fixtures and checkpoints are pinned independently. Baselines are per target, platform, and variant (fixture float32 on CPU, checkpoint bfloat16 on GPU); results record the runtime used. A fixture change moves pin and baseline together. Framework changes never generate their own passing answers.

## What each result means

| Result | Meaning |
| --- | --- |
| Slim passes | The build reproduces the fixture's checked behavior on fixed inputs and runtime. Fast signal only. |
| Slim fails | Investigate; a full-model run shows real impact. |
| Full passes | That checkpoint passes for the recorded revision, inputs, build, and runtime. |
| Full fails | The checkpoint fails the path or differs from reviewed expectations. |
| Full not run | No full-model result for this build, whatever slim says. |

In PR summaries and any Hub status, a slim pass never turns an untested checkpoint green, cancels a required full check, or overrides a `run-slow` failure.

## Current prototype and rollout

Prototype jobs (this repo, evidence only):

- **[Shadow](../.github/workflows/shadow.yml):** daily on `main`, or dispatched on a commit/PR. Runs every contract's fixtures (`--fixtures-only --run-all`) and records what selection would have chosen.
- **[Backfill](../.github/workflows/shadow-backfill.yml):** the same on each `main` commit of a past window ([results](evidence.md#shadow-backfill)).
- **[GPU checks](../.github/workflows/gpu-checks.yml):** contracts and targets on an A10G.
- **[Nightly](../.github/workflows/nightly.yml):** real checkpoints and fixtures on that runner's CPU and GPU, standing in for full-checkpoint paths.

Rollout:

1. Non-blocking slim results in fast PR CI.
2. Full-checkpoint targets in the slow paths, existing tests kept.
3. Stable contracts become required fast checks.
4. Once full-model coverage is equivalent, remove duplicated tests; `run-slow` calls contracts.

Open problems: [problems and opened questions](problems-and-opened-questions.md). Decisions for CI maintainers:

- **Invocation and reporting:** pytest adapter or separate job; runner install; what a failure shows.
- **Selection and scheduling:** `run-slow` families → contract targets; post-merge full checks; scheduled and release coverage.
- **Merge requirements:** when a slim check becomes required; which changes need a full run; missing coverage or broken fixtures.
- **Environment and capacity:** installing the build with pinned deps; caching artifacts; hardware for full targets.
- **Coordination:** who reviews contract and baseline changes; cross-repo candidate testing; rollback keeping earlier pins.

Measure per PR cycle: time to first actionable failure, fast-job duration with setup and downloads, extra full runs requested, repro time, review work, and total CI cost while both suites run. Fast inference alone doesn't make the flow faster.

## Trying the two targets locally

With the [example environment](../example/README.md#run-it):

```sh
# Fast checks on the SmolLM fixtures.
python example/run.py --framework transformers --contract SmolLM --fixtures-only

# A separate check on the actual SmolLM-135M checkpoint.
python example/run.py --framework transformers --contract SmolLM --target SmolLM-135M
```

The runner picks targets by argument; `RUN_SLOW` only affects pytest. More commands: [end-to-end tests](end-to-end-tests.md).
