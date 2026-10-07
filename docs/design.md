# MICs: technical design

Companion to the [proposed standard](../README.md#the-proposed-standard). This document has the details: contract layout, CI triggers, where code lives, what a contract checks, baselines, the runner, the pilot plan, and open questions. Experimental results are in [Evidence](evidence.md); small-fixture details are in [Small fixtures](fixtures.md).

## The standard: contract layout

A MIC covers one usage path (an architecture and a task) and the checkpoints that share it, not one checkpoint. The [full example](../example/README.md) is a runnable MIC for SmolLM text generation covering three checkpoints with two fixtures, in Transformers and Transformers.js.

The pieces live in three places:

```text
hf-internal-testing/SmolLM-integration       # Hub, owned by Hugging Face: what to run
  integration.json                           # manifest
  integrations/
    transformers/usage.md                    # runnable markdown: documentation and test
    transformers.js/integration.js
hf-internal-testing/SmolLM-slim              # Hub, owned by Hugging Face: one fixture per fixture key
  model.safetensors, config.json, tokenizer files copied unchanged, build.json
transformers/tests/contracts/                # framework repo, owned by its maintainers: what to expect
  fleet.lock                                 # pinned contract, fixtures, checkpoints, artifacts, dependencies
  expectations/SmolLM/cpu/SmolLM-slim.json   # one reviewed baseline per target, platform, and variant
  expectations/SmolLM/cpu/SmolLM-135M.json
  expectations/SmolLM/cpu/SmolLM-135M.eager.json
```

The contract module should look like ordinary user code. It loads a model, runs the public API we actually want users to call, and emits a small structured result. The runner, not the contract, compares that result with the framework's baseline.

The manifest describes:

- the usage path and the checkpoints it covers;
- which fixture covers each checkpoint;
- per framework: entry point, format, artifacts (for example ONNX conversions), supported variants, and unsupported targets with a reason;
- whether training checks apply;
- resource requirements;
- the comparison rule for each output.

The runner supplies pinned model and dependency revisions and executes the contract in CI.

This should stay intentionally boring. The useful abstraction is the contract and runner, not a new testing DSL.

### Authoring: runnable markdown

Where a framework's docs support it, the contract is written as [doc-builder runnable code blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) ([example](../example/hub/SmolLM-integration/integrations/transformers/usage.md)). The same file is then:

- a usage example that doc-builder renders, with the runner plumbing removed by `# doc-builder: hide`;
- a test that doc-builder's pytest plugin runs as written against a real checkpoint;
- the MIC contract, which the runner executes against every fixture and checkpoint in the manifest. The runner collects the blocks with doc-builder's own collector, so it runs exactly the code the docs show and test.

What stays outside the snippet is the expected answer. Exact-output asserts in the code (the usual runnable-block style) cannot hold across a slim fixture and a real checkpoint, or across CPU, GPU, and dtypes. Expected results therefore live in reviewed baselines in each framework repository ([why](evidence.md#on-gpu)).

The hidden lines carry three things: the checkpoint or fixture the runner selects, the variant's load arguments, and the result record. Variants currently need a second, hidden load call. A doc-builder hook for parameterizing a block without changing its rendered form would remove it.

### Grouping checkpoints

One contract covers every checkpoint that shares a usage path. Slim fixtures are deduplicated by a fixture key: the processor files byte for byte, plus the config fields that survive shrinking (attention layout, RoPE settings, tied embeddings, vocabulary, special tokens). Width and depth are replaced by shrinking, so checkpoints that differ only in size share one fixture.

In the example, SmolLM-135M and -360M share a key and a fixture. SmolLM-1.7B ships a different `tokenizer.json` and `tokenizer_config.json` and uses full multi-head attention instead of grouped key/value heads, so it gets its own fixture. The fixture build fails if a checkpoint assigned to a fixture has a different key, rather than silently testing it through the wrong one.

Real-checkpoint baselines stay per checkpoint, since outputs differ. Framework PRs run one check per fixture; merges, nightlies, and releases run the listed checkpoints, all of them or a rotating sample.

## Ownership

Hugging Face should own the contracts and fixtures. Each framework should own its expected results. Model authors continue to own their model repositories.

The reason for separating these is operational. We need to be able to:

- update a test when framework behavior changes;
- regenerate a fixture;
- change an expectation after review;
- retire obsolete coverage;
- fix a broken contract.

Those operations should not depend on getting write access to a creator repository or waiting for the creator to maintain framework CI. Expected results change mostly when the framework changes, and they differ by framework and platform anyway, so they live where those changes are reviewed.

| Piece | Default location | Example | Owner |
| --- | --- | --- | --- |
| Real model, config, processor | `org/model` | `HuggingFaceTB/SmolLM-135M` | Creator |
| Contract and manifest, per usage path | HF-owned integration repo | `hf-internal-testing/SmolLM-integration` | Hugging Face |
| Small fixture, per fixture key | HF-owned fixture repo | `hf-internal-testing/SmolLM-slim` | Hugging Face |
| Converted artifacts | HF-owned companion repo | `onnx-community/SmolLM-135M-ONNX` | Hugging Face |
| Fleet lock and expected results | framework repo | `transformers/tests/contracts/` | Framework maintainers |

The exact repository names do not matter yet.

Creators could opt into hosting or co-owning contracts later. That should not be a requirement for the initial system.

## What runs in CI

We should use small fixtures for PR-level coverage and keep real checkpoints for the slower validation paths.

Roughly:

| Trigger | Contract | Weights | Expectations used |
| --- | --- | --- | --- |
| Framework PR | integration repo, pinned | each fixture, pinned | fixture baselines |
| Merge to `main` | integration repo, pinned | listed checkpoints (or HF conversions), pinned | per-checkpoint baselines |
| Nightly | same, full fleet | creator repos, pinned | per-checkpoint baselines |
| Release candidate | same, full fleet | creator repos, pinned | per-checkpoint baselines |
| Model update | integration repo | new creator revision | candidate baseline, reviewed before approval |

The contract code is the same for every trigger; only the weights change. A framework therefore keeps one baseline per fixture and one per checkpoint (and per variant and platform), since a fixture's outputs are unrelated to the real model's. Frameworks that need a converted artifact (for example Transformers.js) load the HF-owned conversion, such as `onnx-community/<model>-ONNX`, instead of the creator repo.

This gives us both directions of compatibility testing:

1. Does a framework change break existing models?
2. Does a model update break supported framework versions?

The important part is that PR checks are cheap enough to be required.

## Where the code lives

The contract code that runs is not in the Transformers repository. Transformers CI decides when to run and which revisions to use, and keeps the expected results; the contract and fixtures are downloaded from the Hub at pinned revisions.

| Piece | Lives in | Example |
| --- | --- | --- |
| Contract code and manifest | HF integration repo on the Hub | `hf-internal-testing/SmolLM-integration` |
| Fixture weights, shrunk config, copied processor files | HF fixture repo on the Hub | `hf-internal-testing/SmolLM-slim` |
| Shared runner (fetch, isolate, run, compare) | one shared package used by every framework | name to be decided |
| Fleet lock, expected results, Transformers adapter, CI workflow | Transformers repo | `tests/contracts/fleet.lock`, `tests/contracts/expectations/`, `.github/workflows/contracts.yml` |
| Framework-behavior tests (compile, cache, batching, generic pipelines) | Transformers repo, `tests/` | unchanged |
| Existing per-model integration tests | Transformers repo, `tests/models/` | unchanged during the pilot |

On a Transformers PR, the workflow reads the fleet lock, fetches each listed integration and fixture repo at its pinned revision, runs the contract against the PR's build, and compares with the baselines in `tests/contracts/expectations/`. A failure points to a contract, a target, and a baseline file, and the same command should reproduce it locally from a Transformers checkout.

When a Transformers PR intentionally changes behavior, it updates the affected baselines in the same PR, so reviewers see the expectation diff next to the code change. Changing the contract itself (new inputs, a new checkpoint, a new variant) is a Hub PR to the integration repo, reviewed by Hugging Face, followed by a revision bump in each framework's fleet lock.

Model-specific integration tests in `tests/models/` move to contracts only after the migration mapping described below; until then both run.

## What we should check

A successful forward pass is too weak.

At minimum, where applicable, a contract should record or check:

- generated tokens or task outputs;
- selected logits or other numerical outputs;
- finite loss;
- gradients on the modules expected to receive them;
- parameter updates after an optimizer step.

Training checks are only relevant to frameworks and models that support training.

The exact tolerances need to be calibrated per fixture and environment. Exact token equality should only be used where it is actually stable.

Baselines are per platform class and dtype; the fixture is shared. On one A10G GPU, baselines recorded there held at `1e-4`, but CPU-recorded logits for convolutional models drifted up to 3e-2, and bfloat16 or float16 runs failed every float32 baseline, sometimes with different generated outputs ([evidence](evidence.md#on-gpu)). A baseline therefore records the device, dtype, and attention implementation it was recorded with, and the runner should refuse to compare across them; the [example runner](../example/run.py) fails a run whose runtime differs from its baseline.

The contract should run the configuration users get by default. If users get SDPA attention, the contract records and checks SDPA; eager can be an extra variant, not a replacement chosen for determinism. In the [historical regression replay](evidence.md), a UMT5 regression that made decoder self-attention non-causal under SDPA was invisible to eager-pinned checks on both the slim fixtures and the real checkpoint.

Inputs need to be deliberate. A contract that only decodes "Paris" and "Berlin" cannot see a decoding regression that only affects punctuation spacing.

## Public API coverage

The integration should exercise the public entry point we expect users to use.

This matters because some regressions occur above `AutoModel`.

For example, the recent `image-to-text` issue was visible through generated Hub usage but would not necessarily have been caught by a direct low-level model call.

That does not mean all pipeline behavior moves out of Transformers. Generic framework behavior should remain framework tests.

The contract is for the model-specific end-to-end usage path.

The [historical regression replay](evidence.md) makes this concrete: generic forward/generate fixtures missed every processor and tokenizer regression we replayed, while the same small fixtures called through the affected public entry point (`processor(..., text_kwargs=...)`, `batch_decode(..., decode_with_timestamps=True)`, batch `decode`) caught them.

## Revisions and baselines

Everything used by required CI needs to be pinned.

A framework repository should keep a fleet lock containing the exact revisions of:

- each checkpoint (model and processor);
- converted artifacts;
- contract;
- fixtures;
- dependency environment.

The expected results sit next to the lock in the same repository, so a framework commit pins them too. Re-running an old framework commit therefore uses its old compatibility fleet and its old expectations.

Expected outputs should never update automatically.

The basic flow should be:

1. A new run produces candidate expectations.
2. A maintainer reviews them against existing tests or known-good behavior.
3. CI compares future runs against that approved baseline.
4. If framework behavior intentionally changes, the same PR updates the baselines in the framework repository, so it shows the expectation diff.
5. If a model changes, the new model revision is tested before its new baseline is approved.

A new framework build must not be allowed to define the expected answer that it is then tested against.

## What should move out of Transformers

The long-term goal is not to move every integration test into contracts.

Tests that mostly say “this checkpoint, through this public usage path, should produce this output” are good candidates.

Framework behavior should stay in Transformers. Examples include:

- compile/export behavior;
- cache behavior;
- batching;
- offloading;
- multi-GPU behavior;
- speculative decoding;
- long-context behavior;
- generic pipeline behavior;
- framework-level attention implementations.

Model-specific variants can live in the contract when they represent supported ways to run that model.

An initial heuristic scan suggests that a large fraction of the current integration suite may fit the contract model, but those numbers are provisional. We should not use them as a migration target until the scan is reproducible and the classifications have been reviewed.

In particular, one usage example cannot replace several tests if those tests exercise meaningfully different behavior.

## Runner

Each framework needs a small adapter that installs the build under test and invokes the contract once per target: each fixture, then each listed checkpoint. The [example runner](../example/run.py) shows the shape.

The contract receives its target through the environment:

```text
MIC_MODEL            # local path: the runner resolves the pinned fixture or checkpoint revision
MIC_MODEL_REVISION   # for remote artifacts (e.g. an ONNX conversion)
MIC_MODEL_KWARGS     # the variant's load arguments, as JSON
```

Resolving revisions in the runner and passing a local path keeps the contract code identical to what users write: `from_pretrained(model_id)`, with no revision plumbing in the visible snippet.

The final stdout line contains structured JSON, including the runtime actually used (device, dtype, attention implementation). The runner picks the baseline from the target, the variant, and that runtime's platform.

Missing fields, invalid numerical values, failed assertions, or expectation mismatches fail the contract.

The manifest should have a schema and must be inspectable without executing the integration code.

## Converted artifacts

A framework does not have to consume the same repository as another framework.

For example, Transformers may use the creator's PyTorch checkpoint while Transformers.js uses an HF-managed ONNX conversion.

For converted artifacts, we need to record separately:

- source model revision;
- converted artifact revision;
- exporter version;
- exporter options;
- conversion validation.

We should not pretend these are the same artifact or revision.

## Pilot

Before designing Hub product around this, run a small CI pilot.

Pick 3–5 usage paths, each covering the checkpoints that share it, including:

- a standard text model;
- a processor-heavy or multimodal model;
- a model with meaningful GPU requirements;
- a model that is currently painful to maintain;
- at least one Transformers.js integration.

For those usage paths:

1. build the contracts;
2. generate small fixtures, shared by fixture key;
3. implement the shared runner;
4. add PR, nightly, and model-update CI paths;
5. keep the existing tests running in parallel;
6. exercise at least one contribution using a private framework build before upstream merge.

The pilot should run long enough that we see actual maintenance rather than only initial setup.

## What we need to measure

The pilot needs to answer a few concrete questions.

### Bug coverage

Does a small contract catch real regressions that matter?

We should replay documented historical regressions and verify each one against:

1. the real checkpoint;
2. the small fixture;
3. the proposed checks.

A synthetic fault suite is not enough. The [first replay](evidence.md) is a start; the pilot should extend it with GPU-specific regressions and with bugs found by the existing nightly and slow tests.

### Cost

How much cheaper is PR validation at fleet scale?

Measure actual CI cost, including loading, setup, fixture storage, and runner overhead.

### Maintenance

How much work does this create?

Track:

- shrink-rule work;
- broken fixtures;
- baseline reviews;
- dependency-related expectation changes;
- special cases;
- fixture regeneration;
- manual overrides.

This is probably the biggest unknown.

### Signal quality

How often do we get actionable failures versus noise?

Track false positives, time to diagnose, and time to fix.

### Reuse

Does the same integration module work well enough to back docs, Hub snippets, Colabs, or agent usage without creating a second maintenance path?

That is useful upside, but it should not be required for the CI pilot to succeed.

## Hub follow-up

If the CI system works, the Hub can expose its results.

Possible follow-ups:

- revision-specific compatibility status;
- a Tests tab;
- API access to results;
- links to companion contract/fixture repositories;
- per-checkpoint results that framework CI publishes back (the baselines themselves stay in framework repositories);
- tested “Use this model” snippets;
- Colabs backed by the same integration module.

A compatibility result should be an attestation for a specific set of revisions and environment, not a permanent “supported” badge.

The [Claude walkthrough](agent-walkthrough/claude-opus.md) shows what this would replace: a card saying "install `transformers` from `main`" long after release, `transformersInfo` claiming native support for a repo that only works with (broken) remote code, and no link from that repo to the maintained native port in another organization.

The record should include at least:

- model and processor revision;
- converted artifact revision where applicable;
- contract and fixture revision;
- framework build;
- dependencies;
- hardware/environment class;
- variant;
- result;
- timestamp.

None of this is necessary to start the CI work.

## Execution safety

Integration modules are arbitrary code.

The runner therefore needs isolation, resource limits, pinned revisions, secret protection, and explicit handling of private models and private framework builds.

Reading a contract manifest must never execute integration code.

Remote-code models should probably come later rather than complicating the first version.

## Decision for the first version

Start with:

- HF-owned contracts, one per usage path, covering the checkpoints that share it;
- expected results kept in each framework's repository;
- frozen small fixtures, one per fixture key, shrunk by layer pattern, overfit by default, and gated on input sensitivity and numeric health in every dtype and platform they run on, with the real processor files copied unchanged ([why](fixtures.md#choosing-the-weights));
- contracts that call the public entry points users call, with the default runtime configuration;
- recorded logits and other useful numerical outputs;
- targeted training checks where training is supported;
- pinned baselines;
- required fixture-based PR CI;
- real-checkpoint validation after merge, nightly, and before release.

Do not require:

- creators to modify their repositories;
- creators to maintain the contracts;
- pruned real-weight fixtures, except as a fallback where overfitting does not converge and the size is acceptable;
- new Hub UI;
- migration of every existing integration test.

## Open questions

The pilot should resolve:

1. How many architectures can we shrink reliably?
2. Which failures require the real checkpoint or real hardware?
3. What is the right unit for selecting affected contracts on a PR?
4. How expensive is baseline review in practice?
5. Which expectations remain stable across supported hardware? On one GPU so far: generated outputs in float32 yes; logits only for models without convolutions ([evidence](evidence.md#on-gpu)).
6. How do we handle licensed, gated, or private models in fleet validation?
7. When should a model/framework combination be considered unsupported versus simply unvalidated?
8. What amount of contract maintenance is acceptable before this becomes another test suite we have to carry?
9. Can doc-builder runnable blocks carry the runner's parameters (checkpoint, variant) without hidden reload lines, and should doc pages include contracts directly from the Hub?
10. How do published fixtures stay out of model discovery (tags, card text, search exclusion, or an unlisted organization), so they don't show up as candidate models for users and agents?

If the pilot shows good real-regression coverage at substantially lower PR cost, without creating large ongoing maintenance overhead, we can expand the fleet and then build the Hub-facing pieces.
