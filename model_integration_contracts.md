# Proposal: Model integration contracts — CI first, Hub later

## Summary

We should add a small executable integration contract for each model/framework combination we want to support.

A contract contains:

- a normal usage example;
- a small model fixture where practical;
- expected outputs or numerical checks;
- metadata describing how to run it.

The contract is run from framework CI.

The model repository itself does not need to move or give Hugging Face write access. Contracts, fixtures, and expected results can live in Hugging Face-owned companion repositories on the Hub. For example, for `meta-llama/Llama-3.2-1B`:

- `meta-llama/Llama-3.2-1B`: the real model, owned by the creator, unchanged;
- `hf-internal-testing/Llama-3.2-1B-integration`: the contract and expected results;
- `hf-internal-testing/Llama-3.2-1B-slim`: the small fixture.

The first goal is CI, not a new Hub feature. If this works, the same contracts can later back compatibility status, tested snippets, Colabs, APIs, or agent usage.

## Problem

Model compatibility is currently expensive to verify.

A lot of integration coverage depends on real checkpoints, large downloads, GPUs, framework-specific test code, or manual checks. That limits how much of the model fleet we can realistically run on every PR.

It also means we often discover regressions late: after merge, in nightly jobs, during a release, or from users.

There is also duplicated maintenance. The code we use to validate a model is often separate from the code we show users, which is separate again from Hub snippets and other examples.

The proposal is to make one executable usage path the unit of compatibility.

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

## Small fixtures

The small fixture should preserve enough of the real model structure to exercise its integration path, without requiring the full checkpoint.

There are two useful starting points.

### 1. Frozen random fixture

Start from a reduced model configuration, initialize it deterministically, freeze the resulting weights and inputs, and record numerical outputs.

This is cheap and does not require training the fixture.

A random fixture is only useful if its outputs actually depend on its inputs. When the fixture is built, we should perturb or swap its inputs and require the recorded logits to move well beyond tolerance. Some tiny random models wash out their inputs (see the Whisper result below); such a fixture cannot catch preprocessing or label bugs, whatever we assert.

### 2. Overfit fixture

Optionally train the reduced model on a few examples so that it produces human-readable expected outputs.

This is useful when exact behavior adds coverage, but it should not be a requirement unless we can show that it catches bugs the numerical checks miss. So far that only happened for fixtures that failed the input-sensitivity check above, so overfitting is the fallback for those fixtures rather than the default.

The fixture still needs to preserve processor-coupled dimensions and model-specific configuration where those affect behavior: vocabulary, special tokens, image sizes, patch sizes, audio dimensions, RoPE settings, MoE structure, and so on.

Not every model will be shrinkable with a generic recipe. A failed fixture build should be visible as missing coverage, not silently skipped.

Only the weights and model config should be shrunk. Tokenizer, processor, and preprocessing files should be copied from the real model byte for byte, not loaded and re-saved. Re-saving with the fixture builder's framework version silently normalizes legacy configs, and in the historical replay below that hid two real tokenizer-loading regressions from the slim fixture while the real checkpoint still showed them.

Copying costs almost nothing. Across eight families, the original processor files were within about ±1.5 MB of the re-saved ones (UMT5: +4.5 MB on an 87 MB fixture), and loaded as fast or faster locally. The real cost is that fixtures keep the full vocabulary, so processor files are 20–50% of each fixture; that is needed to keep the real tokenizer and is independent of copying.

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

The contract should run the configuration users get by default. If users get SDPA attention, the contract records and checks SDPA; eager can be an extra variant, not a replacement chosen for determinism. In the replay below, a UMT5 regression that made decoder self-attention non-causal under SDPA was invisible to eager-pinned checks on both the slim fixtures and the real checkpoint.

Inputs need to be deliberate. A contract that only decodes "Paris" and "Berlin" cannot see a decoding regression that only affects punctuation spacing.

## Public API coverage

The integration should exercise the public entry point we expect users to use.

This matters because some regressions occur above `AutoModel`.

For example, the recent `image-to-text` issue was visible through generated Hub usage but would not necessarily have been caught by a direct low-level model call.

That does not mean all pipeline behavior moves out of Transformers. Generic framework behavior should remain framework tests.

The contract is for the model-specific end-to-end usage path.

The historical replay below makes this concrete: generic forward/generate fixtures missed every in-family regression we replayed, while the same small fixtures called through the affected public entry point (`processor(..., text_kwargs=...)`, `batch_decode(..., decode_with_timestamps=True)`, batch `decode`) caught them.

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

## Current fixture work

There is already related work in Transformers around generating small random models from test configurations and real processors.

That is a useful first stage because it gives us broad smoke coverage cheaply.

A second stage would start from the actual model configuration and processor, apply architecture-specific shrink rules, freeze the resulting fixture, check that it is sensitive to its inputs, and overfit it only when it is not.

Starting from the real config should preserve more integration-relevant behavior, but it will not automatically work for every architecture. Nested configs, coupled dimensions, MoE models, and processor assumptions will need explicit handling.

## Local experiment

I ran a small experiment comparing frozen random fixtures with overfit fixtures ([script](model_integration_contracts_experiment.py), [raw results](model_integration_contracts_results.json)).

The setup used reduced SmolLM and CLIP models, three seeds, and 14 synthetic faults, giving 42 fault/seed cases per policy.

Detection was:

| Checks | Random | Overfit |
| --- | ---: | ---: |
| Successful inference only | 0/42 | 0/42 |
| Recorded discrete outputs | 19/42 | 18/42 |
| Recorded logits | 33/42 | 33/42 |
| Logits + targeted gradients/update | 39/42 | 39/42 |
| Above + recorded loss | **42/42** | **42/42** |

The important result is not that the fixture caught every synthetic fault. These were handpicked faults.

The useful result is that overfitting did not add detection beyond the numerical and training checks in this experiment.

In fact, exact overfit answers still passed for some faults that disabled RoPE, removed masks, or changed pixel scaling. The numerical outputs caught those cases.

Targeted gradient checks caught disconnected paths, and recording loss caught shuffled labels even when gradients remained finite and parameters still updated.

There were no false alarms in 60 control runs covering repetition, save/reload, and eager-to-SDPA execution.

The fixtures were also much smaller than the full checkpoints, but these local timings should not be treated as fleet-level speedup estimates. They exclude installation, loading, downloads, CI startup, fixture maintenance, and several other real costs.

### More model families

I then ran the same script, checks, and tolerances on four more families from the `hf-tiny-v2` fast-integration PR list: BERT (masked LM), T5 (encoder-decoder), Whisper (speech seq2seq), and Wav2Vec2 (CTC). That gave 28 faults and 84 fault/seed cases per policy ([raw results](model_integration_contracts_results_more.json)).

| Checks | Random | Overfit |
| --- | ---: | ---: |
| Successful inference only | 0/84 | 0/84 |
| Recorded discrete outputs | 30/84 | 24/84 |
| Recorded logits | 57/84 | 60/84 |
| Logits + targeted gradients/update | 69/84 | 72/84 |
| Above + recorded loss | **78/84** | **84/84** |

BERT, T5, and Wav2Vec2 behaved like SmolLM and CLIP: both policies caught everything.

Whisper is the exception. The tiny random Whisper decoder barely depends on its audio, so reversing the audio moved logits by about 5e-5 (below the 1e-4 tolerance), and shuffling labels across clips left the loss unchanged. It missed both faults on every seed. The overfit Whisper had to tell the clips apart, so the same faults moved logits by 1–24 and loss from about 0.01 to 4–7.

Exact answers were still weak on their own: the overfit fixtures' answers only caught 1–3 faults per fixture. Overfitting helped by making the numbers depend on the input, not by making answers checkable.

Overfitting cost under 4 s per fixture for BERT, T5, and Wav2Vec2, and 23–51 s for Whisper. One Wav2Vec2 seed did not converge in 500 steps, which is the kind of fixture maintenance the pilot needs to measure. There were no false alarms in 120 control runs.

So the current default should be:

**Use frozen random fixtures with recorded numerical outputs, loss, and targeted training checks, gated by an input-sensitivity check. Overfit only the fixtures that fail that check.**

This experiment shows local sensitivity. It does not establish historical bug coverage or maintenance savings.

## Historical regression replay

To test real bugs rather than synthetic faults, I replayed documented Transformers regressions fixed since May 2026 ([freeze/check](model_integration_contracts_replay.py), [per-case driver](model_integration_contracts_replay_case.py), [usage repros](model_integration_contracts_usage_repros.py)).

Method:

1. Freeze random and overfit slim fixtures, plus real-checkpoint baselines, on a known-good build (`main` at `4d536e1f`).
2. Extract the bad build (the fix commit's parent) and the fixed build.
3. Run the frozen checks on both builds without regenerating anything. A case counts as caught when the bad build is flagged and the fixed build is not.
4. Separately, run the affected public call on the real checkpoint and on the slim fixture under both builds.

Builds before 2026-08-17 need `tokenizers` 0.22.x (and builds before mid-June an older `kernels`), so those ran with a dependency overlay. Old builds cannot always run in today's environment; the fleet lock needs to pin the dependency environment, not just model revisions.

| Regression (fix PR) | Real checkpoint | Generic slim checks | Slim fixture, affected public call |
| --- | --- | --- | --- |
| CLIP processor drops `return_tensors` when `text_kwargs` is passed (#46195) | crash | missed | crash, caught |
| Whisper batch `decode_with_timestamps` (#47997) | crash | missed | crash, caught |
| T5 batch `decode` ignores `clean_up_tokenization_spaces` (#47793) | caught | missed | caught only with punctuation in the inputs |
| `additional_special_tokens` lost with legacy tokenizer config (#47848) | caught | missed | missed: re-saved tokenizer hides it |
| OLMo tokenizer appends a stray EOS (#47988) | caught | missed: re-saved tokenizer hides it | — |
| UMT5 decoder self-attention not causal under SDPA (#49135) | caught under SDPA only | missed (eager) | caught under SDPA: future-token leak 2.9–15 vs 2e-6 |
| T5 tied-weights order (#48238) | not reproduced with `t5-small` | clean | — |

The pattern is consistent:

- **Generic checks are not enough.** A forward pass plus greedy generate on frozen inputs caught none of these regressions. They live in processors, tokenizers, decoding helpers, or a non-default attention path.
- **The usage path catches them, and the small fixture is enough for it.** Whenever the contract made the affected call, the slim fixture caught the bug just as the real checkpoint did. None of these needed real weights.
- **Fixture packaging matters.** Two tokenizer-loading regressions were visible on the real checkpoint but not on the slim fixture, because the fixture builder had re-saved the tokenizer with a fixed build.
- **Runtime defaults matter.** The UMT5 bug only exists under SDPA, the default.

To measure false alarms, I replayed 10 benign changes from the same window that touched the same code paths (RoPE, greedy decoding, image processing, masking, attention refactors, auto-generated inits). Across 20 builds and 42 frozen checks per build, there were **no false alarms** with eager attention. Rechecking all 20 builds under SDPA, the default, also produced no false alarms, including the change that moved Wav2Vec2 to SDPA dispatch and an attention refactor for continuous batching.

The candidate search also showed that documented regressions inside a given model family are relatively rare in a five-month window; most bugs that matter to these models are in shared processor, tokenizer, and attention code. That supports organizing contracts around public usage paths rather than per-model forward checks.

This is a small sample: seven regressions, ten benign changes, CPU only, and no GPU-specific case.

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

A synthetic fault suite is not enough. The first replay (above) is a start; the pilot should extend it with GPU-specific regressions and with bugs found by the existing nightly and slow tests.

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
