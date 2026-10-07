# Slim models in Transformers development and CI

**Slim models provide fast checks on pull requests. They do not establish that the full model passes.** In the proposed integration, the same contracts run against full checkpoints in `run-slow` and broader scheduled validation. The purpose of the [MIC proposal](../README.md) is to catch useful regressions earlier while preserving full-model validation.

The intended integration adds slim contract checks to fast PR CI and runs contracts against real checkpoints in the slower paths. For an ordinary framework fix, the developer gains an automatic check and a reproducible failure report. Contract authors and maintainers take on fixture preparation, baseline review, and coordination across repositories when the tested usage changes.

The MICs in this repository currently run as a separate prototype, including a shadow job; they are not yet wired into Transformers' fast PR jobs or `run-slow`. The development flows below describe the proposed integration and the decisions needed to implement it.

## What contracts replace in `run-slow`

Contracts can replace the model-specific integration tests currently executed by `run-slow`: tests that load a particular checkpoint, call a public usage path, and check its outputs. After migration, `run-slow` executes the contract against that full checkpoint and compares its reviewed full-model baseline. Fast PR CI executes the same contract against a slim fixture and its separate baseline.

| Part of the flow | Effect of adopting contracts |
| --- | --- |
| Fast PR checks | Gain slim-fixture executions of the selected contracts. |
| Model-specific integration tests in `run-slow` | Can move to contracts executed against full checkpoints, once equivalent coverage is verified. |
| The `run-slow` trigger and CI orchestration | Continue to select and schedule full-model checks on the requested PR build. |
| Other slow framework tests | Remain in Transformers, including framework-level cache, compilation, batching, offloading, and multi-GPU tests. |

During the pilot, existing integration tests and their candidate contracts run alongside each other. Once the contract retains a test's inputs, assertions, variants, hardware, and run frequency, the duplicated test can be removed. Keeping full-model validation does not require permanently maintaining two copies of the same integration test. A slim-only check cannot justify that removal.

## Impact on development work

| Developer task | Existing work | Work added by MICs |
| --- | --- | --- |
| Fix framework code without changing expected behavior | Change code and tests, inspect PR CI, request slow tests when needed. | Inspect the selected slim contract results. Reuse the existing fixtures and baselines; no fixture rebuild or Hub edit is normally needed. |
| Intentionally change model behavior | Update affected test expectations and explain the change in the PR. | Review candidate baselines for affected contract targets and variants in the same framework PR, with separate evidence from full-model runs. |
| Add a model or usage path to the tested fleet | Add framework support and appropriate fast and slow tests. | Add or extend a Hub contract, reuse or build its fixtures, and add fleet-lock entries and reviewed fixture and full-checkpoint baselines. |
| Change a migrated integration test's inputs, API calls, or variants | Update the test in Transformers. | Edit its Hub contract and coordinate the reviewed revision with framework fleet-lock and baseline updates. |
| Change a checkpoint or rebuild a fixture | Update the test artifact references and expectations that depend on it. | Validate the candidate revision separately, then update its pin and affected baselines explicitly. Required framework checks keep using the previous pins until that update is reviewed. |

Framework maintainers own the fleet lock and expected results in the framework repository. Hugging Face owns the integration and fixture repositories. Model creators keep their existing repositories and do not need to maintain framework CI. The [ownership split](design.md#ownership) reduces dependence on creator access, while requiring an agreed review flow for changes spanning the contract and the framework.

## An ordinary framework PR

1. The developer changes Transformers code and adds or updates the relevant framework tests.
2. Fast PR CI runs its existing checks and selects the relevant MICs. It fetches each pinned fixture once, executes the contracts against the PR build, and compares with the fixture baselines committed in the framework repository. It does not rebuild fixtures or train new ones.
3. The PR report identifies each contract, fixture, variant, comparison failure, and reproduction command. A pass reports only that the fast fixture check passed. The developer investigates a failure in the same way as another regression test.
4. Maintainers request `run-slow` when full-model validation is needed before merge. That path uses real checkpoints: migrated integration tests execute as contracts, while the remaining slow tests continue in Transformers. It reports full-model results separately from slim results.
5. The PR is merged under the repository's applicable check requirements. A slim pass does not satisfy an outstanding requirement to run or pass full-model tests. After merge, the broader full-model validation continues.

The prototype's [selector](../example/select_contracts.py) maps model directories to their contracts and selects all contracts for shared library code or dependency declarations. For example, a UMT5 modeling change selects the T5 contract; a change to shared generation code selects every contract. The shadow job currently runs all contracts while recording that selection so missed dependencies can be measured before relying on it in PR CI.

## Handling a contract failure

Start with the reported target and runtime. Reproduce a slim failure with the same pinned fixture; reproduce a full-model failure with that checkpoint and its baseline. A fixture failure can justify requesting a full-model run to assess its impact, but it is already a useful early signal even when that investigation happens later.

An unintended behavior change should be fixed in the framework or the contract, according to its cause. An intentional behavior change needs a candidate baseline, an explanation of why the new output is acceptable, and review in the framework PR. Fixture and full-model candidates remain separate; accepting a new fixture answer cannot approve a changed full-model answer.

The prototype's `--record` option writes candidates and checks that they reproduce in a fresh process. Required CI must continue comparing with reviewed baselines rather than recording the build under test as its own expected answer. Missing artifacts, missing baselines, insufficient resources, and unsupported variants must remain identifiable so maintainers can distinguish lost coverage from an output regression. The [validation gaps](problems-and-opened-questions.md#validation-gaps-in-the-prototype) need closing before these results become required checks.

## Adding or changing a contract

Onboarding a usage path is less frequent than running it on a PR, but requires more preparation:

1. Author the public usage example and manifest in the HF-owned integration repository, listing the real checkpoints, supported variants, outputs, and resources.
2. Reuse a fixture with the same fixture key, or build and validate a new fixture. Publish its frozen files at a pinned revision. This is preparation work, not part of every PR run.
3. Record candidate expectations for the fixtures and real checkpoints on a known-good framework build and the supported runtimes. Review the outputs before approving the baselines.
4. Add the contract and artifact pins, reviewed baselines, and contract-selection information to the framework's fleet. Run both the fast and full-model paths alongside existing tests during the pilot.

When a framework change also requires new contract code, review needs to cover the combined candidate. The proposed flow is to prepare an immutable candidate contract revision, test it with the framework PR and its candidate baselines, then update the framework's fleet lock through review. Older framework commits retain their earlier lock and expectations. The exact staging and publication mechanics still need an end-to-end trial; the prototype's integration repositories are currently local and unpublished.

A creator updating a model must not silently change the input to every required framework check. A separate model-update flow should test the new checkpoint revision, determine whether it can share the existing fixture, and propose reviewed pin and baseline updates. That flow remains to be implemented.

## Fast PR checks and full model checks

Transformers already separates its regular [PR CI](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml) from explicitly requested slow checks. Tests marked `@slow` are skipped by default and enabled with `RUN_SLOW`, as defined in [the testing utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py).

A maintainer can request slow model tests on a PR with a comment such as:

```text
run-slow: llama, clip
```

The [comment workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-comment-ci.yml) selects the requested model families and runs their slow tests on the PR's merge commit. Those integration tests use full published checkpoints: for example, the [Llama slow tests](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py) load real Llama checkpoints. `run-slow` can therefore validate full models before a PR is merged; it is not limited to checks after merge. The slow suite can also contain other expensive framework tests.

MICs fit into these paths as follows. This is the intended contract integration, with full-model coverage retained throughout the rollout:

| When | Contract target | Purpose |
| --- | --- | --- |
| Fast checks on a PR | Relevant pinned slim fixtures, normally on CPU | Catch regressions cheaply and return early feedback. |
| `run-slow` requested on a PR | Selected full checkpoints on suitable hardware | Validate the actual models and runtime variants before merge. |
| After merge to `main` | Affected full checkpoints, according to the validation policy | Check the merged build against real models. |
| Scheduled or nightly validation | The registered full-checkpoint fleet, with explicit resource or access exceptions | Provide broader real-model coverage beyond the checks selected for one PR. |
| Release candidate validation | Full checkpoints and required runtime variants | Establish the release candidate's tested compatibility. |

The [existing scheduled Nvidia workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-scheduled-caller.yml) already runs broader model CI. The contract fleet, selection policy, and release integration remain pilot work, as described in the [technical design](design.md#what-runs-in-ci).

## One contract with separate targets and expectations

The contract calls the same public usage path with different model files. The runner selects the target and compares its output with that target's reviewed baseline.

For example, the [SmolLM contract](../example/hub/SmolLM-integration/integration.json) covers three checkpoints with two fixtures:

- Fast PR checks run `proto-SmolLM-slim` once for the shared 135M and 360M usage path, and `proto-SmolLM-1.7B-slim` for the distinct 1.7B configuration and tokenizer.
- Full-model checks load SmolLM-135M, SmolLM-360M, and SmolLM-1.7B separately. Each has its own expected outputs; passing the shared fixture does not mark either covered checkpoint as passing.

The fleet lock pins fixtures and real checkpoints independently. Baselines are also separate by target, platform, and variant. A fixture may use float32 on CPU while a full checkpoint uses bfloat16 on GPU; each result describes the runtime actually checked.

Fixtures are built and validated ahead of time, then published and pinned. Fast CI downloads the frozen artifacts rather than shrinking or training a model on each PR. When a fixture changes, its pin and reviewed baseline change together. Framework changes are checked against existing expectations; they do not automatically generate their own passing answers.

## What each result means

| Result | Meaning |
| --- | --- |
| Slim fixture passes | The tested build reproduces the fixture's checked behavior on its fixed inputs and runtime. This is a fast regression signal. |
| Slim fixture fails | The fast check found a mismatch or execution failure to investigate. A full-model run can establish its impact on a real checkpoint. |
| Full checkpoint passes | That checkpoint passes the checks for the recorded revision, inputs, framework build, and runtime. |
| Full checkpoint fails | The actual checkpoint fails that tested usage path or differs from its reviewed expectations. |
| Full checkpoint has not run | There is no full-model result for this build, even if every slim check passes. |

Keep these outcomes distinct in PR summaries and any later Hub status. A slim pass must not turn an untested real checkpoint green, cancel a required full-model check, or override a failure from `run-slow`. Even a full-model pass remains scoped to the inputs and runtime tested.

Ordinary framework tests also continue to cover behavior such as caching, compilation, batching, and multi-GPU execution. Only the model-specific integration checks selected for migration move into contracts; adding a fast contract does not transfer all framework coverage into its small usage example.

## Current prototype and rollout

The [shadow workflow](../.github/workflows/shadow.yml) runs in this repository, independently of Transformers CI. It runs daily against Transformers `main` and can be dispatched against a commit or PR. It checks the slim fixtures with `--fixtures-only`, runs every contract with `--run-all`, and records which ones the changed-file selection would have chosen. That lets the pilot measure missed selections without skipping those checks. The [backfill workflow](../.github/workflows/shadow-backfill.yml) runs the same checks on every `main` commit in a past date window, so weeks of history are measured in one run: over 2026-08-17 to 2026-10-07 (563 commits), the fixtures failed only on the UMT5 regression, until its fix, and the selection skipped every contract on half the commits and selected all of them on nearly all others ([evidence](evidence.md#shadow-backfill)).

The [GPU workflow](../.github/workflows/gpu-checks.yml) separately exercises the contracts and their targets on an A10G, and the [nightly workflow](../.github/workflows/nightly.yml) runs every contract's real checkpoints and fixtures on Transformers `main` on that runner's CPU and GPU, standing in for the full-checkpoint paths. These prototype runs provide evidence for the future integration; they do not make a MIC a required Transformers PR check or automatically add it to `run-slow`.

The rollout is to add non-blocking slim contract results to fast PR CI, connect full-checkpoint targets to the slow validation paths, and keep existing tests running alongside them during evaluation. Only stable contracts should later become required fast checks. Once equivalent full-model coverage is verified, remove the duplicated integration tests and let `run-slow` invoke their contracts. Any migration must retain the tests' inputs, assertions, runtime coverage, and full-model execution frequency. The [problems and opened questions](problems-and-opened-questions.md) describe what remains to establish before changing coverage.

CI maintainers still need to settle these integration decisions:

- **Invocation and reporting:** whether contracts run through a pytest adapter or a separate job, how developers install the runner locally, and which commands and artifacts appear on a failure.
- **Selection and scheduling:** how model families requested by `run-slow` map to contract targets; which full-model checks run after each merge; and how scheduled and release jobs cover the rest of the fleet.
- **Merge requirements:** when a stable slim check becomes required, which changes also require a full-model run, and how missing coverage or a temporarily broken fixture is handled.
- **Environment and capacity:** how the framework build under test is installed with the pinned dependency environment, how fixtures and checkpoints are cached, and how full-model targets are assigned suitable hardware.
- **Coordination and upkeep:** who reviews contract and baseline changes, how candidate revisions are tested across repositories, and how a bad update is rolled back without losing the earlier pins.

Measure the impact on a developer's PR cycle: time to the first actionable failure, fast-job duration including setup and downloads, additional full-model runs requested, time spent reproducing failures, and baseline or fixture review work. Also measure the total CI cost while both suites run. Fast inference alone does not establish that the overall development flow becomes faster or cheaper.

## Trying the two targets locally

With the [example environment](../example/README.md#run-it) installed, the prototype can demonstrate both paths from this repository:

```sh
# Fast checks on the SmolLM fixtures.
python example/run.py --framework transformers --contract SmolLM --fixtures-only

# A separate check on the actual SmolLM-135M checkpoint.
python example/run.py --framework transformers --contract SmolLM --target SmolLM-135M
```

The second command loads the full checkpoint and compares its own baseline. These commands illustrate target selection in the prototype; `RUN_SLOW` controls Transformers' pytest tests, while the MIC runner selects targets through its command-line arguments. The [end-to-end test guide](end-to-end-tests.md) lists the remaining prototype jobs and commands.
