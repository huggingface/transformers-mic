# Slim models in Transformers development and CI

The proposal uses slim fixtures for fast PR checks and the same contracts on full checkpoints in `run-slow` and scheduled jobs. **A slim pass does not mean the full model passes.** Developers gain a reproducible failure report; maintainers pay for fixtures, baseline review, and cross-repo coordination. Today the [prototype](#current-prototype-and-rollout) runs here, plus a branch-only proof in Transformers CI; wider rollout is proposed.

## What contracts replace in `run-slow`

Model-specific integration tests (load a checkpoint, call a public path, check outputs) become contracts, run on the full checkpoint in `run-slow` and on a slim fixture in PR CI, each against its own baseline. The `run-slow` trigger and orchestration are unchanged; other slow tests (cache, compilation, batching, offloading, multi-GPU) stay. A test is removed only once its contract keeps its inputs, assertions, variants, hardware, and run frequency; a slim-only check never justifies removal.

## Impact on development work

| Developer task | Today | Added by MICs |
| --- | --- | --- |
| Fix code, behavior unchanged | Change code and tests; request slow tests if needed. | Read slim results. |
| Intentionally change behavior | Update expectations, explain in the PR. | Review candidate baselines in the same PR, with separate full-model evidence. |
| Add a model or usage path | Add support and fast/slow tests. | Add or extend a Hub contract and fixture; add fleet-lock entries and reviewed baselines. |
| Change a migrated test's inputs, calls, or variants | Edit the test. | Edit the Hub contract, with matching fleet-lock and baseline updates. |
| Change a checkpoint or rebuild a fixture | Update references and expectations. | Validate, then update pin and baselines; required checks keep old pins until reviewed. |

Ownership is as in the [design](design.md#ownership); cross-repo changes need an agreed review flow.

## An ordinary framework PR

1. Fast CI selects the relevant MICs, fetches each pinned fixture once, runs them on the PR build, and compares with committed baselines; it never rebuilds or trains fixtures.
2. The report names contract, fixture, variant, failure, and repro command.
3. For full-model validation, maintainers request `run-slow`: migrated tests run as contracts on real checkpoints, reported apart from slim results. A slim pass never satisfies a pending full-model requirement.

The prototype [selector](../example/select_contracts.py) maps model directories to contracts, sub-models included (a UMT5 change selects T5; an EnCodec change selects MusicGen Melody), and selects all for shared code or dependency changes. The shadow job runs everything and records the selection, to measure missed dependencies first.

## Handling a contract failure

Reproduce on the reported target and baseline; a fixture failure can justify a full-model run. Fix unintended changes. For intended changes, `--record` creates a candidate reproduced in a fresh process; explain and review it in the PR. Fixture approval never approves a full-model answer. Artifact/baseline errors and hardware/variant exclusions are reported separately ([validation](problems-and-opened-questions.md#6-runner-validation-gaps)).

## Adding or changing a contract

1. Write the usage example and manifest (checkpoints, variants, outputs, resources) in the integration repo.
2. Reuse a fixture with the same key, or build, validate, and publish one at a pinned revision.
3. Record and review fixture and checkpoint candidates on a known-good build, every supported runtime.
4. Add pins, baselines, and selection info to the framework's fleet lock.

**Contract and framework changing together** (proposed, untested end to end): publish an immutable candidate contract revision, test it with the framework PR and candidate baselines, then bump the fleet lock through review. Old commits keep their old lock.

**Creator updates a model.** This must not silently change required checks' inputs. A flow (not built) should test the revision, check it still shares the fixture, and propose pin and baseline updates.

## Fast PR checks and full model checks

Transformers separates [PR CI](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml) from `@slow` tests, which run only with `RUN_SLOW` ([testing utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py)). A maintainer comment `run-slow: llama, clip` makes the [comment workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-comment-ci.yml) run those families' slow tests on full checkpoints, on the PR merge commit (e.g. [Llama](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py)); the [scheduled Nvidia workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-scheduled-caller.yml) runs broader model CI. Contracts slot in per trigger as in the [design](design.md#what-runs-in-ci): fixtures on PRs, selected full checkpoints in `run-slow`, the full fleet after merge, nightly, and on release candidates.

## One contract with separate targets and expectations

The [SmolLM contract](../example/hub/SmolLM-integration/integration.json) covers three checkpoints with two fixtures: `proto-SmolLM-slim` for the shared 135M/360M path, `proto-SmolLM-1.7B-slim` for 1.7B's distinct config and tokenizer. Each target, fixture or checkpoint, is pinned independently and has its own baselines per platform and variant, in its source's dtype; a fixture pass marks no checkpoint as passing. A fixture change moves pin and baseline together.

## What each result means

| Result | Meaning |
| --- | --- |
| Slim passes | The build reproduces the fixture's checked behavior on fixed inputs and runtime; fast signal only. |
| Slim fails | Investigate; a full-model run shows real impact. |
| Full passes | That checkpoint passes for the recorded revision, inputs, build, and runtime. |
| Full fails | The checkpoint fails the path or differs from reviewed expectations. |
| Full not run | No full-model result, whatever slim says. |

PR summaries and Hub status keep these scopes separate: slim success cannot cancel required full checks or override `run-slow` failures.

## Current prototype and rollout

In this repo: [shadow](../.github/workflows/shadow.yml) runs every fixture (`--fixtures-only --run-all`) daily on `main` or on a dispatched commit/PR, recording what selection would have chosen; [backfill](../.github/workflows/shadow-backfill.yml) does the same on each `main` commit of a past window ([results](evidence.md#shadow-backfill)); [nightly](../.github/workflows/nightly.yml) runs real checkpoints and fixtures on an A10G runner's CPU and GPU, standing in for full-checkpoint paths; [GPU checks](../.github/workflows/gpu-checks.yml) are one-off experiments. The checkpoint-only WhisperSpeech contract covers #48108 in nightly; fixture-only shadow runs report its explicit exclusion because both slim weight kinds miss the bug.

In `huggingface/transformers`, branch `contracts-proof`: [`contracts.yml`](https://github.com/huggingface/transformers/blob/contracts-proof/.github/workflows/contracts.yml) installs the branch, selects contracts from its changed files, and runs them on `aws-m8i-l-cache` in `transformers-torch-light`, with the runner vendored in `utils/contracts/` and the lock and baselines (x86, plus `cpu-arm64` for Mac repro) in `tests/contracts/`: six contracts, 8 fixtures in 17 s. All proof phases are done, including demos on `proof-*` branches and a maintainer write-up ([proof plan](transformers-ci-proof.md)).

Rollout ([proof plan](transformers-ci-proof.md)):

1. Non-blocking slim results in fast PR CI.
2. Full-checkpoint targets in the slow paths, existing tests kept.
3. Stable contracts become required fast checks.
4. Once full-model coverage is equivalent, remove duplicated tests; `run-slow` calls contracts.

Decisions for CI maintainers (open problems: [problems and opened questions](problems-and-opened-questions.md)):

- **Invocation and reporting:** pytest adapter or separate job; runner install; what a failure shows.
- **Selection and scheduling:** `run-slow` families → contract targets; post-merge full checks; scheduled and release coverage.
- **Merge requirements:** when a slim check becomes required; which changes need a full run; missing coverage or broken fixtures.
- **Environment and capacity:** installing the build with pinned deps; caching artifacts; hardware for full targets.
- **Coordination:** who reviews contract and baseline changes; cross-repo candidate testing; rollback keeping earlier pins.

Measure per PR cycle: time to first actionable failure, fast-job duration with setup and downloads, extra full runs requested, repro time, review work, and total CI cost while both suites run.

## Trying the two targets locally

With the [example environment](../example/README.md#run-it):

```sh
python example/run.py --framework transformers --contract SmolLM --fixtures-only          # fast checks on the SmolLM fixtures
python example/run.py --framework transformers --contract SmolLM --target SmolLM-135M     # the real SmolLM-135M checkpoint
```

`RUN_SLOW` only affects pytest. More commands: [end-to-end tests](end-to-end-tests.md).
