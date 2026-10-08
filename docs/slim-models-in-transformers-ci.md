# Slim models in Transformers development and CI

Slim fixtures give fast checks on PRs. **A slim pass does not mean the full model passes.** The same contracts run against full checkpoints in `run-slow` and scheduled validation. The goal of the [MIC proposal](../README.md) is to catch regressions earlier while keeping full-model validation.

Who gains and who pays:

- **Developers** get an automatic check and a reproducible failure report on ordinary framework fixes.
- **Contract authors and maintainers** take on fixture preparation, baseline review, and cross-repository coordination when tested usage changes.

Today the MICs run as a separate prototype with a shadow job. They are not wired into Transformers' fast PR jobs or `run-slow`. The flows below describe the proposed integration.

## What contracts replace in `run-slow`

Contracts can replace the model-specific integration tests in `run-slow`: tests that load a checkpoint, call a public usage path, and check the outputs. After migration, `run-slow` runs the contract against the full checkpoint and its full-model baseline. Fast PR CI runs the same contract against a slim fixture and its own baseline.

| Part of the flow | Effect of adopting contracts |
| --- | --- |
| Fast PR checks | Gain slim-fixture runs of the selected contracts. |
| Model-specific integration tests in `run-slow` | Move to contracts run against full checkpoints, once equivalent coverage is verified. |
| The `run-slow` trigger and CI orchestration | Unchanged: still select and schedule full-model checks on the PR build. |
| Other slow framework tests | Stay in Transformers: cache, compilation, batching, offloading, multi-GPU. |

During the pilot, existing tests and their candidate contracts run side by side. A test can be removed once its contract keeps its inputs, assertions, variants, hardware, and run frequency. No permanent duplication is needed, but a slim-only check never justifies removal.

## Impact on development work

| Developer task | Existing work | Work added by MICs |
| --- | --- | --- |
| Fix framework code without changing expected behavior | Change code and tests, inspect PR CI, request slow tests when needed. | Inspect the selected slim contract results. Fixtures and baselines are reused; no rebuild or Hub edit. |
| Intentionally change model behavior | Update test expectations and explain the change in the PR. | Review candidate baselines for affected targets and variants in the same PR, with separate full-model evidence. |
| Add a model or usage path to the fleet | Add framework support and fast and slow tests. | Add or extend a Hub contract, reuse or build fixtures, add fleet-lock entries and reviewed fixture and checkpoint baselines. |
| Change a migrated test's inputs, calls, or variants | Update the test in Transformers. | Edit the Hub contract; coordinate the revision with fleet-lock and baseline updates. |
| Change a checkpoint or rebuild a fixture | Update artifact references and dependent expectations. | Validate the candidate revision, then update its pin and baselines explicitly. Required checks keep the old pins until reviewed. |

Ownership ([details](design.md#ownership)):

- Framework maintainers own the fleet lock and expected results.
- Hugging Face owns the integration and fixture repositories.
- Model creators keep their repositories and maintain no framework CI.

This removes the dependence on creator access, but changes spanning contract and framework need an agreed review flow.

## An ordinary framework PR

1. The developer changes Transformers code and its tests.
2. Fast PR CI selects the relevant MICs, fetches each pinned fixture once, runs the contracts on the PR build, and compares with the committed fixture baselines. It never rebuilds or trains fixtures.
3. The PR report names each contract, fixture, variant, failure, and reproduction command. A pass means only that the fast fixture check passed. Failures are investigated like any regression test.
4. When full-model validation is needed before merge, maintainers request `run-slow`. Migrated tests run as contracts on real checkpoints; other slow tests run as before. Full-model results are reported apart from slim ones.
5. The PR merges under the repository's check requirements. A slim pass never satisfies a pending full-model requirement. Broader full-model validation continues after merge.

The prototype's [selector](../example/select_contracts.py) maps model directories to contracts, and selects every contract for shared library code or dependency changes. A UMT5 modeling change selects the T5 contract; a shared generation change selects all. The shadow job runs everything anyway and records the selection, so missed dependencies can be measured before PR CI relies on it.

## Handling a contract failure

Reproduce on the reported target and runtime: a slim failure with the same pinned fixture, a full-model failure with that checkpoint and baseline. A fixture failure is a useful early signal on its own; it can also justify a full-model run to assess impact.

- **Unintended change:** fix the framework or the contract, depending on the cause.
- **Intended change:** record a candidate baseline, explain why the new output is acceptable, and review it in the framework PR. Fixture and full-model candidates stay separate: approving a new fixture answer never approves a full-model one.

`--record` writes candidates and checks they reproduce in a fresh process. Required CI must still compare with reviewed baselines, never with the build under test. Missing artifacts or baselines, insufficient resources, and unsupported variants stay distinguishable from output regressions ([validation gaps](problems-and-opened-questions.md#6-runner-validation-gaps)).

## Adding or changing a contract

Onboarding a usage path is rarer than running it, but needs more preparation:

1. Write the usage example and manifest in the HF-owned integration repository: real checkpoints, variants, outputs, resources.
2. Reuse a fixture with the same fixture key, or build and validate a new one. Publish it at a pinned revision. This is one-time preparation, not per-PR work.
3. Record candidate expectations for fixtures and checkpoints on a known-good build and every supported runtime. Review them before approval.
4. Add the contract and artifact pins, baselines, and selection information to the framework's fleet. Run fast and full-model paths alongside existing tests during the pilot.

**Contract and framework changing together.** Review must cover the combination. Proposed flow: publish an immutable candidate contract revision, test it with the framework PR and its candidate baselines, then bump the fleet lock through review. Older framework commits keep their old lock and expectations. This still needs an end-to-end trial; the prototype's integration repositories are local and unpublished.

**Creator updates a model.** That must not silently change the input to every required check. A separate flow (not yet built) should test the new revision, decide whether it still shares the fixture, and propose reviewed pin and baseline updates.

## Fast PR checks and full model checks

Transformers already separates regular [PR CI](https://github.com/huggingface/transformers/blob/main/.github/workflows/pr-ci-caller.yml) from requested slow checks. `@slow` tests are skipped unless `RUN_SLOW` is set ([testing utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/testing_utils.py)). A maintainer requests them on a PR with a comment:

```text
run-slow: llama, clip
```

The [comment workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-comment-ci.yml) runs those families' slow tests on the PR's merge commit, with full checkpoints (e.g. the [Llama slow tests](https://github.com/huggingface/transformers/blob/main/tests/models/llama/test_modeling_llama.py)). So `run-slow` validates full models *before* merge. The slow suite also holds other expensive framework tests.

Where MICs fit, with full-model coverage kept throughout:

| When | Contract target | Purpose |
| --- | --- | --- |
| Fast checks on a PR | Relevant pinned slim fixtures, normally on CPU | Cheap, early regression feedback. |
| `run-slow` on a PR | Selected full checkpoints on suitable hardware | Validate real models and runtime variants before merge. |
| After merge to `main` | Affected full checkpoints, per validation policy | Check the merged build against real models. |
| Scheduled or nightly | The registered full-checkpoint fleet, with explicit resource or access exceptions | Broader coverage than one PR selects. |
| Release candidate | Full checkpoints and required variants | Establish the release's tested compatibility. |

The [scheduled Nvidia workflow](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-scheduled-caller.yml) already runs broader model CI. Fleet, selection policy, and release integration are pilot work ([technical design](design.md#what-runs-in-ci)).

## One contract with separate targets and expectations

One contract, one public usage path, different model files. The runner picks a target and compares with that target's baseline. The [SmolLM contract](../example/hub/SmolLM-integration/integration.json) covers three checkpoints with two fixtures:

- **Fast PR checks** run `proto-SmolLM-slim` once for the shared 135M/360M path, and `proto-SmolLM-1.7B-slim` for 1.7B's distinct config and tokenizer.
- **Full-model checks** load each of SmolLM-135M, -360M, and -1.7B, each with its own expected outputs. Passing the shared fixture marks no checkpoint as passing.

The fleet lock pins fixtures and checkpoints independently. Baselines are separate per target, platform, and variant: a fixture may run float32 on CPU while its checkpoint runs bfloat16 on GPU. Each result records the runtime actually used.

Fixtures are built, validated, published, and pinned ahead of time; CI downloads them, never shrinks or trains on a PR. A fixture change moves its pin and baseline together. Framework changes are checked against existing expectations and never generate their own passing answers.

## What each result means

| Result | Meaning |
| --- | --- |
| Slim fixture passes | The build reproduces the fixture's checked behavior on its fixed inputs and runtime. A fast regression signal. |
| Slim fixture fails | A mismatch or execution failure to investigate. A full-model run can show the real impact. |
| Full checkpoint passes | That checkpoint passes for the recorded revision, inputs, build, and runtime. |
| Full checkpoint fails | The checkpoint fails the usage path or differs from its reviewed expectations. |
| Full checkpoint has not run | No full-model result for this build, whatever the slim checks say. |

Keep these distinct in PR summaries and any later Hub status. A slim pass must never turn an untested checkpoint green, cancel a required full-model check, or override a `run-slow` failure. A full-model pass is still scoped to its inputs and runtime.

Framework tests keep covering caching, compilation, batching, and multi-GPU. Only the model-specific integration checks chosen for migration move into contracts.

## Current prototype and rollout

All prototype jobs run in this repository, outside Transformers CI. They provide evidence; they do not make a MIC a required PR check or add it to `run-slow`.

- **[Shadow](../.github/workflows/shadow.yml):** daily on Transformers `main`, or dispatched on a commit or PR. Checks slim fixtures (`--fixtures-only`), runs every contract (`--run-all`), and records what changed-file selection would have chosen, so missed selections are measured without skipping checks.
- **[Backfill](../.github/workflows/shadow-backfill.yml):** the same checks on every `main` commit in a past window. Over five months it caught the covered regression, flagged one benign change, missed four uncovered ones, and selection never skipped a contract whose result changed ([evidence](evidence.md#shadow-backfill)).
- **[GPU checks](../.github/workflows/gpu-checks.yml):** contracts and targets on an A10G.
- **[Nightly](../.github/workflows/nightly.yml):** every contract's real checkpoints and fixtures on `main`, on that runner's CPU and GPU, standing in for the full-checkpoint paths.

Rollout:

1. Add non-blocking slim results to fast PR CI.
2. Connect full-checkpoint targets to the slow paths. Keep existing tests running alongside.
3. Make only stable contracts required fast checks.
4. Once full-model coverage is verified equivalent, remove the duplicated tests and let `run-slow` call their contracts. Migration keeps inputs, assertions, runtime coverage, and full-model run frequency.

What remains to establish is in [problems and opened questions](problems-and-opened-questions.md).

Decisions for CI maintainers:

- **Invocation and reporting:** pytest adapter or separate job; local runner install; commands and artifacts shown on failure.
- **Selection and scheduling:** mapping `run-slow` families to contract targets; which full-model checks run after each merge; how scheduled and release jobs cover the rest.
- **Merge requirements:** when a stable slim check becomes required; which changes also need a full-model run; handling missing coverage or a broken fixture.
- **Environment and capacity:** installing the build under test with pinned dependencies; caching fixtures and checkpoints; assigning hardware to full-model targets.
- **Coordination and upkeep:** who reviews contract and baseline changes; testing candidate revisions across repos; rolling back a bad update without losing earlier pins.

Measure the effect on a PR cycle: time to first actionable failure, fast-job duration including setup and downloads, extra full-model runs requested, time reproducing failures, and review work. Also measure total CI cost while both suites run. Fast inference alone does not make the development flow faster or cheaper.

## Trying the two targets locally

With the [example environment](../example/README.md#run-it) installed:

```sh
# Fast checks on the SmolLM fixtures.
python example/run.py --framework transformers --contract SmolLM --fixtures-only

# A separate check on the actual SmolLM-135M checkpoint.
python example/run.py --framework transformers --contract SmolLM --target SmolLM-135M
```

The second command loads the full checkpoint and compares with its own baseline. The MIC runner picks targets by argument; `RUN_SLOW` only affects Transformers' pytest tests. Other jobs and commands: [end-to-end tests](end-to-end-tests.md).
