# Model integration contracts — technical design

Companion to the [proposal](../README.md). This document has the details: contract layout, CI triggers, where code lives, what a contract checks, baselines, the runner, the pilot plan, and open questions. Experimental results are in [Evidence](evidence.md); small-fixture details are in [Small fixtures](fixtures.md).

## Proposed model

Each supported model/framework pair gets an integration module plus a manifest.

For example:

```text
integration.json
integrations/
  transformers/
    integration.py
  transformers.js/
    integration.js
expectations/
  ...
```

The integration module should look like ordinary user code. It loads a model, runs the public API we actually want users to call, checks the result, and emits a small structured result.

The manifest describes:

- framework and runtime;
- artifact repository;
- entry point;
- supported variants;
- expected outputs;
- resource requirements.

The runner supplies pinned model and dependency revisions and executes the module in CI.

This should stay intentionally boring. The useful abstraction is the contract and runner, not a new testing DSL.

## Ownership

Hugging Face should own the compatibility contracts and expected results.

Model authors continue to own their model repositories.

The reason for separating the two is operational. We need to be able to:

- update a test when framework behavior changes;
- regenerate a fixture;
- change an expectation after review;
- retire obsolete coverage;
- fix a broken contract.

Those operations should not depend on getting write access to a creator repository or waiting for the creator to maintain framework CI.

A possible layout is:

| Piece | Default location | Example | Owner |
| --- | --- | --- | --- |
| Real model, config, processor | `org/model` | `meta-llama/Llama-3.2-1B` | Creator |
| Contract and expectations | HF-owned integration repo | `hf-internal-testing/Llama-3.2-1B-integration` | Hugging Face |
| Small fixture | HF-owned fixture repo | `hf-internal-testing/Llama-3.2-1B-slim` | Hugging Face |
| Converted artifacts | HF-owned companion repo | `onnx-community/Llama-3.2-1B-ONNX` | Hugging Face |

The exact repository names do not matter yet.

Creators could opt into hosting or co-owning contracts later. That should not be a requirement for the initial system.

## What runs in CI

We should use small fixtures for PR-level coverage and keep real checkpoints for the slower validation paths.

Roughly:

| Trigger | Contract and expectations | Weights | Expectations used |
| --- | --- | --- | --- |
| Framework PR | integration repo, pinned | fixture repo, pinned | slim baseline |
| Merge to `main` | integration repo, pinned | creator repo (or HF conversion), pinned | real-checkpoint baseline |
| Nightly | same, full fleet | creator repos, pinned | real-checkpoint baseline |
| Release candidate | same, full fleet | creator repos, pinned | real-checkpoint baseline |
| Model update | integration repo | new creator revision | candidate baseline, reviewed before approval |

The contract code is the same for every trigger; only the weights change. The integration repo therefore keeps two baselines per model: one for the slim fixture and one for the real checkpoint, since the fixture's outputs are unrelated to the real model's. Frameworks that need a converted artifact (for example Transformers.js) load the HF-owned conversion, such as `onnx-community/<model>-ONNX`, instead of the creator repo.

This gives us both directions of compatibility testing:

1. Does a framework change break existing models?
2. Does a model update break supported framework versions?

The important part is that PR checks are cheap enough to be required.

## Where the code lives

The test code that runs is not in the Transformers repository. Transformers CI decides when to run and which revisions to use; the contract itself is downloaded from the Hub at a pinned revision.

| Piece | Lives in | Example |
| --- | --- | --- |
| Contract code, manifest, slim and real baselines | HF integration repo on the Hub | `hf-internal-testing/Llama-3.2-1B-integration` |
| Fixture weights, shrunk config, copied processor files | HF fixture repo on the Hub | `hf-internal-testing/Llama-3.2-1B-slim` |
| Shared runner (fetch, isolate, run, compare) | one shared package used by every framework | name to be decided |
| Fleet lock, Transformers adapter, CI workflow | Transformers repo | e.g. `tests/contracts/fleet.lock`, `.github/workflows/contracts.yml` |
| Framework-behavior tests (compile, cache, batching, generic pipelines) | Transformers repo, `tests/` | unchanged |
| Existing per-model integration tests | Transformers repo, `tests/models/` | unchanged during the pilot |

On a Transformers PR, the workflow reads the fleet lock, fetches each listed integration and fixture repo at its pinned revision, and runs the contract against the PR's build. A failure points to a Hub repo and revision, and the same command should reproduce it locally from a Transformers checkout.

Changing a test therefore has two steps: update the contract or expectations in the Hub repo (a Hub PR, reviewed by HF), then bump that revision in the fleet lock. When a Transformers PR intentionally changes behavior, the expectation update and the lock bump should land with it, so the PR shows the expectation diff.

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

- model;
- processor;
- converted artifacts;
- contract;
- fixture;
- expected results;
- dependency environment.

Re-running an old framework commit should therefore use its old compatibility fleet.

Expected outputs should never update automatically.

The basic flow should be:

1. A new run produces candidate expectations.
2. A maintainer reviews them against existing tests or known-good behavior.
3. CI compares future runs against that approved baseline.
4. If framework behavior intentionally changes, the PR shows the expectation diff.
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

Each framework needs a small adapter that installs the build under test and invokes the integration module.

For Python, the contract could receive:

```text
MODEL_ID
MODEL_REVISION
MODEL_KWARGS
```

The integration itself can remain ordinary Python:

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    **model_kwargs,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)
```

The final stdout line contains structured JSON. Logs go to stderr.

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

Pick 3–5 models covering:

- a standard text model;
- a processor-heavy or multimodal model;
- a model with meaningful GPU requirements;
- a model that is currently painful to maintain;
- at least one Transformers.js integration.

For those models:

1. build the contracts;
2. generate small fixtures;
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
- tested “Use this model” snippets;
- Colabs backed by the same integration module.

A compatibility result should be an attestation for a specific set of revisions and environment, not a permanent “supported” badge.

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

- HF-owned contracts;
- frozen small fixtures, checked for input sensitivity, with the real processor files copied unchanged;
- contracts that call the public entry points users call, with the default runtime configuration;
- recorded logits and other useful numerical outputs;
- targeted training checks where training is supported;
- pinned baselines;
- required fixture-based PR CI;
- real-checkpoint validation after merge, nightly, and before release.

Do not require:

- creators to modify their repositories;
- creators to maintain the contracts;
- overfit fixtures, except where a random fixture fails the input-sensitivity check;
- new Hub UI;
- migration of every existing integration test.

## Open questions

The pilot should resolve:

1. How many architectures can we shrink reliably?
2. Which failures require the real checkpoint or real hardware?
3. What is the right unit for selecting affected contracts on a PR?
4. How expensive is baseline review in practice?
5. Which expectations remain stable across supported hardware?
6. How do we handle licensed, gated, or private models in fleet validation?
7. When should a model/framework combination be considered unsupported versus simply unvalidated?
8. What amount of contract maintenance is acceptable before this becomes another test suite we have to carry?

If the pilot shows good real-regression coverage at substantially lower PR cost, without creating large ongoing maintenance overhead, we can expand the fleet and then build the Hub-facing pieces.
