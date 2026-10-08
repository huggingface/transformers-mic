# MICs: technical design

Companion to the [proposed standard](../README.md#the-proposed-standard). Results are in [Evidence](evidence.md); fixture details in [Small fixtures](fixtures.md); limits in [Problems and opened questions](problems-and-opened-questions.md).

## The standard: contract layout

A MIC covers one usage path (an architecture and a task) and every checkpoint that shares it. The [full example](../example/README.md) covers SmolLM text generation: three checkpoints, two fixtures, Transformers and Transformers.js.

The pieces live in three places:

```text
hf-internal-testing/SmolLM-integration       # Hub, owned by Hugging Face: what to run
  integration.json                           # manifest
  integrations/
    transformers/usage.md                    # runnable markdown: documentation and test
    transformers.js/integration.js
hf-internal-testing/proto-SmolLM-slim        # Hub, owned by Hugging Face: one fixture per fixture key
  model.safetensors, config.json, tokenizer files copied unchanged, build.json
transformers/tests/contracts/                # framework repo, owned by its maintainers: what to expect
  fleet.lock                                 # pinned contract, fixtures, checkpoints, artifacts, dependencies
  expectations/SmolLM/cpu-arm64/proto-SmolLM-slim.json   # one reviewed baseline per target, platform, and variant
  expectations/SmolLM/cpu-arm64/SmolLM-135M.json
  expectations/SmolLM/cpu-arm64/SmolLM-135M.eager.json
```

The contract is ordinary user code. It loads a model, calls the public API, and emits a small structured result. The runner, not the contract, compares that result with the framework's baseline. Keep it boring: the abstraction is the contract and the runner, not a new testing DSL.

The manifest declares:

- the usage path and the checkpoints it covers;
- which fixture covers each checkpoint;
- per framework: entry point, format, artifacts (e.g. ONNX conversions), supported variants, and unsupported targets with a reason;
- whether training checks apply;
- minimum hardware per target;
- the comparison rule for each output.

Reference inputs need an immutable identity (pinned asset revision and content hash), not a mutable URL. Model, processor, and input revisions go in the result record.

### Minimum hardware

The manifest's `resources` gives each target (`fixture` for all fixtures, or a checkpoint repo) its minimum specs, with optional per-variant overrides:

```json
"resources": {
  "fixture": {"memory_gb": 1},
  "HuggingFaceTB/SmolLM-1.7B": {"memory_gb": 12, "disk_gb": 8, "measured": "peak RSS 10.6 GB, Apple M5"},
  "org/large-moe": {"devices": ["cuda"], "gpus": 4, "gpu_memory_gb": 80, "memory_gb": 64, "disk_gb": 400,
                    "variants": {"bfloat16": {"gpus": 2}}}
}
```

| Field | Meaning |
| --- | --- |
| `devices` | Where the target can run (default CPU and CUDA). |
| `gpus` | Above 1 means multi-GPU; the contract handles it (e.g. `device_map="auto"`). |
| `gpu_memory_gb` | Per GPU. |
| `memory_gb` | Host RAM. |
| `disk_gb` | Free disk for the download. |

- The runner checks these before downloading. A target the host cannot run is `unsupported` with the reason (`needs cuda; this host has no GPU`), never a failure.
- CI picks the runner class from the same numbers.
- Fixtures run anywhere, so a GPU-only checkpoint still gets PR coverage on CPU.
- Measure, don't guess. SmolLM-1.7B was guessed at 20 GB and measured at 10.6 GB; a 16 GB runner was still killed recording it because two processes held the model at once.

### Authoring: runnable markdown

Where a framework's docs support it, the contract is written as [doc-builder runnable code blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) ([example](../example/hub/SmolLM-integration/integrations/transformers/usage.md)). One file is then:

- a usage example doc-builder renders, with runner plumbing hidden by `# doc-builder: hide`;
- a test doc-builder's pytest plugin runs against a real checkpoint;
- the contract, which the runner executes on every fixture and checkpoint. It collects blocks with doc-builder's own collector, so it runs exactly what the docs show.

The expected answer stays outside the snippet. Exact-output asserts cannot hold across fixture and checkpoint, or across CPU, GPU, and dtypes, so expected results live in reviewed baselines in each framework repo ([why](evidence.md#on-gpu)).

The hidden lines carry the target the runner selects, the variant's load arguments, and the result record. Variants currently need a second, hidden load call; a doc-builder hook for parameterizing a block would remove it.

A published example must name the path actually tested. Both [agent walkthroughs](agent-walkthrough/README.md) had to reconstruct a working loader from several sources. A workaround must be explicit and reviewed; a passing patched example is not evidence that the unmodified model-card snippet passes.

### Grouping checkpoints

Checkpoints share a fixture when they share a **fixture key**: processor files byte for byte, plus the config fields that survive shrinking (attention layout, RoPE, tied embeddings, vocabulary, special tokens). Width and depth are shrunk, so checkpoints that differ only in size share a fixture.

- SmolLM-135M and -360M share one fixture.
- SmolLM-1.7B gets its own: different `tokenizer.json` and `tokenizer_config.json`, and full multi-head attention instead of grouped KV heads.
- The fixture build fails if an assigned checkpoint has a different key.

Real-checkpoint baselines stay per checkpoint. PRs run one check per fixture; merges, nightlies, and releases run the checkpoints (all, or a rotating sample).

## Ownership

| Piece | Default location | Example | Owner |
| --- | --- | --- | --- |
| Real model, config, processor | `org/model` | `HuggingFaceTB/SmolLM-135M` | Creator |
| Contract and manifest, per usage path | HF integration repo | `hf-internal-testing/SmolLM-integration` | Hugging Face |
| Small fixture, per fixture key | HF fixture repo | `hf-internal-testing/proto-SmolLM-slim` | Hugging Face |
| Converted artifacts | HF companion repo | `onnx-community/SmolLM-135M-ONNX` | Hugging Face |
| Shared runner (fetch, isolate, run, compare) | one package for all frameworks | name to be decided | Hugging Face |
| Fleet lock, expected results, adapter, CI workflow | framework repo | `tests/contracts/fleet.lock`, `tests/contracts/expectations/`, `.github/workflows/contracts.yml` | Framework maintainers |

Repository names are placeholders.

The split is operational. Updating a test, regenerating a fixture, approving an expectation, retiring coverage, or fixing a contract must not wait on write access to a creator's repo. Expected results change mostly with the framework and differ per framework and platform, so they live where framework changes are reviewed.

Creators could opt into hosting or co-owning contracts later; it is not required.

## What runs in CI

Small fixtures on PRs; real checkpoints on the slower paths. PR checks must be cheap enough to be required.

| Trigger | Contract | Weights | Expectations used |
| --- | --- | --- | --- |
| Framework PR | integration repo, pinned | each fixture, pinned | fixture baselines |
| Merge to `main` | integration repo, pinned | listed checkpoints (or HF conversions), pinned | per-checkpoint baselines |
| Nightly | same, full fleet | creator repos, pinned | per-checkpoint baselines |
| Release candidate | same, full fleet | creator repos, pinned | per-checkpoint baselines |
| Model update | integration repo | new creator revision | candidate baseline, reviewed before approval |

The contract code is the same for every trigger; only the weights change. Fixture and checkpoint outputs are unrelated, so each gets its own baseline (per variant and platform). Frameworks that need a converted artifact (e.g. Transformers.js) load the HF conversion instead of the creator repo.

This tests both directions: does a framework change break models, and does a model update break supported framework versions?

The prototype runs these paths in its own repo, outside Transformers CI:

- [shadow job](../.github/workflows/shadow.yml): fixtures on Transformers `main`, daily or on demand for any commit or PR;
- [shadow backfill](../.github/workflows/shadow-backfill.yml): fixtures on every `main` commit in a past window ([results](evidence.md#shadow-backfill));
- [nightly job](../.github/workflows/nightly.yml): checkpoints and fixtures on an A10G runner's CPU and GPU.

## Where the code lives

The contract code is not in the Transformers repo. Transformers CI decides when to run and at which revisions, and keeps the expected results; contracts and fixtures are fetched from the Hub at pinned revisions (see [Ownership](#ownership)). Framework-behavior tests (compile, cache, batching, generic pipelines) stay in `tests/`. Per-model integration tests in `tests/models/` keep running during the pilot.

On a Transformers PR, the workflow:

1. reads the fleet lock;
2. fetches each integration and fixture repo at its pinned revision;
3. runs the contracts against the PR's build;
4. compares with `tests/contracts/expectations/`.

A failure names a contract, a target, and a baseline file, and the same command reproduces it locally.

Two kinds of change:

- **Framework behavior changes intentionally:** the same PR updates the affected baselines, so reviewers see the expectation diff.
- **The contract changes** (new inputs, checkpoint, or variant): a Hub PR to the integration repo, reviewed by Hugging Face, then a revision bump in each framework's fleet lock.

## What we should check

A forward pass that runs is too weak. Where applicable, a contract checks:

- generated tokens or task outputs;
- selected logits or other numerical outputs;
- finite loss;
- gradients on the expected modules;
- parameter updates after an optimizer step.

Training checks apply only where training is supported. Tolerances are calibrated per fixture and environment; exact token equality only where it is stable.

**Baselines are per platform class and dtype; the fixture is shared.** Logits drift across platforms and dtypes beyond tolerance ([evidence](evidence.md#on-gpu)). Each baseline records its device, dtype, and attention implementation, and the [runner](../example/run.py) refuses to compare across them.

**A match means agreement with reviewed behavior** on fixed inputs and one runtime, not semantic correctness on arbitrary inputs. Fixture and real-checkpoint results stay distinct: a passing fixture is not a reference answer for the real model. For vision tasks, records include the input identity and the checked outputs (labels, boxes) with explicit comparison rules.

**Run the default configuration.** If users get SDPA, the contract checks SDPA; eager is an extra variant, not a replacement chosen for determinism. Eager-pinned checks missed the UMT5 SDPA regression on both fixtures and the real checkpoint ([replay](evidence.md#historical-regression-replay)).

**Choose inputs deliberately.** A contract that only decodes "Paris" and "Berlin" cannot see a regression in punctuation spacing.

## Public API coverage

Contracts call the public entry point users call, because some regressions sit above `AutoModel`. The recent `image-to-text` issue showed through generated Hub usage, not necessarily through a low-level model call. In the [replay](evidence.md#historical-regression-replay), generic forward/generate checks missed every processor and tokenizer regression; the same fixtures called through the affected API (`processor(..., text_kwargs=...)`, `batch_decode(..., decode_with_timestamps=True)`, batch `decode`) caught them.

Generic pipeline behavior stays in framework tests. The contract covers the model-specific end-to-end path.

## Revisions and baselines

Everything required CI uses is pinned. Each framework repo keeps a fleet lock with exact revisions of:

- each checkpoint (model and processor);
- converted artifacts;
- the contract;
- fixtures;
- the dependency environment.

Expected results sit next to the lock, so a framework commit pins them too. Re-running an old commit uses its old fleet and expectations.

**Expected outputs never update automatically.** A build under test must not define its own expected answer.

1. A run produces candidate expectations.
2. A maintainer reviews them against existing tests or known-good behavior.
3. CI compares future runs against the approved baseline.
4. An intentional framework change updates its baselines in the same PR.
5. A model update is tested at its new revision before its baseline is approved.

## What should move out of Transformers

Not every integration test. Good candidates say "this checkpoint, through this public path, produces this output". Framework behavior stays in Transformers:

- compile/export;
- cache;
- batching;
- offloading;
- multi-GPU;
- speculative decoding;
- long context;
- generic pipelines;
- framework-level attention implementations.

Model-specific variants can live in the contract when they are supported ways to run that model. One usage example cannot replace several tests that exercise different behavior.

An initial heuristic scan suggests a large share of the integration suite fits the contract model. Those numbers are provisional: not a migration target until the scan is reproducible and reviewed.

## Runner

Each framework needs a small adapter that installs the build under test and runs the contract once per target: each fixture, then each checkpoint. The [example runner](../example/run.py) shows the shape.

The contract receives its target through the environment:

```text
MIC_MODEL            # local path: the runner resolves the pinned fixture or checkpoint revision
MIC_MODEL_REVISION   # for remote artifacts (e.g. an ONNX conversion)
MIC_MODEL_KWARGS     # the variant's load arguments, as JSON
```

Resolving revisions in the runner keeps the visible snippet identical to user code: `from_pretrained(model_id)`, no revision plumbing.

The last stdout line is a JSON result, including the runtime actually used (device, dtype, attention). The runner picks the baseline from target, variant, and platform. Missing fields, non-finite values, failed assertions, and mismatches fail.

The [runner prototype](../example/README.md#runner-prototype) implements this for six contracts:

| Feature | What it does |
| --- | --- |
| Schemas | Manifest, fleet lock, fixture build record, baseline, tolerances, and result each have a JSON schema ([`schemas/`](../example/schemas)), validated on every load and before every result. The manifest is readable without running contract code. |
| Batching | Runnable-markdown contracts run in the runner's process, so imports are paid once ([cost](evidence.md#stability-and-cost)). `--isolate` gives one process per target; script contracts (Transformers.js) always get their own. |
| Pinned fixtures | The fleet lock pins file hashes. A missing or different fixture is an error naming the build command, not a skip. |
| Tolerances per platform | Manifest defaults; `expectations/<contract>/<platform>/tolerances.json` overrides one platform, with a required reason. |
| Reproducible baselines | Non-finite outputs fail and are never recorded. `--record` reruns each target in a fresh process and refuses differing outputs. This caught a seeding mistake and a Wav2Vec2 uninitialized-weight bug ([details](../example/README.md#runner-prototype)). |
| Minimum hardware | `resources` checked before download; an unrunnable target is `unsupported` with the reason. |
| Known failures | `known_failures` in the fleet lock, with an issue: failure reports `xfail`, a pass fails the run so the entry gets removed. Or `local_patches` names a workaround, applied in every contract process and recorded in each result (e.g. [#49385](https://github.com/huggingface/transformers/issues/49385)). |
| Replays | [`replay.py`](../example/replay.py) runs the unchanged runner on a regression's bad and fixed builds. |

Not implemented: isolation for untrusted contract code, private-artifact rules, choosing the CI runner class from `resources`.

## Converted artifacts

Frameworks need not consume the same repo: Transformers may use the creator's PyTorch checkpoint while Transformers.js uses an HF ONNX conversion. Record separately:

- source model revision;
- converted artifact revision;
- exporter version;
- exporter options;
- conversion validation.

A conversion is not the same artifact or revision as its source.

### Golden references for kernels and performance

A reviewed real-checkpoint baseline is a golden reference for kernel and backend work (e.g. WebGPU in Transformers.js):

1. Run the contract on the new backend.
2. Compare discrete outputs with an existing platform's baseline. Logits only compare within a platform class, so they are reported as drift ([why](evidence.md#on-gpu)).
3. Once reviewed, the backend gets its own baselines.

The [example runner](../example/run.py) does this with `--golden cpu`. On an Apple M5 it passed WebGPU fp32 and caught fp16 and q4f16 returning all-zero logits ([repro](../experiments/webgpu_browser/README.md)).

Records also carry timings (load, first and warm generate, forward). They are stored, not compared, so a kernel change can be tracked for speed on the same inputs that check its correctness. Speed needs real checkpoints on target hardware; fixtures are too small.

## Pilot

Run a small CI pilot before any Hub product work. Pick 3–5 usage paths, including:

- a standard text model;
- a processor-heavy or multimodal model;
- a model with real GPU requirements;
- a model that is painful to maintain today;
- at least one Transformers.js integration.

For each:

1. build the contracts;
2. generate fixtures, shared by fixture key;
3. implement the shared runner;
4. add PR, nightly, and model-update CI paths;
5. keep existing tests running in parallel;
6. exercise at least one contribution with a private framework build before upstream merge.

Run long enough to see real maintenance, not just setup.

## What we need to measure

### Bug coverage

Does a small contract catch regressions that matter? Verify each documented regression against the real checkpoint, the fixture, and the proposed checks. Synthetic faults are not enough.

- The [first replay](evidence.md#historical-regression-replay) is a start. Extend it with GPU-specific regressions and bugs found by the existing nightly and slow tests.
- Replaying a framework's own history avoids picking cases. The [shadow backfill](evidence.md#shadow-backfill) does this with frozen contracts and also records what changed-file selection would have run.
- Contracts catch only the calls they make; the [improvement loop](improving-contracts.md) closes gaps, and its detections are counted separately.

### Cost

How much cheaper is PR validation at fleet scale? Measure loading, setup, fixture storage, and runner overhead.

### Maintenance

Probably the biggest unknown. Track:

- shrink-rule work;
- broken fixtures;
- baseline reviews;
- dependency-related expectation changes;
- special cases;
- fixture regeneration;
- manual overrides;
- [contract improvements](improving-contracts.md).

### Signal quality

Track false positives, time to diagnose, and time to fix. Classify every failure the way the [backfill](evidence.md#shadow-backfill) did: regression, intended change, or benign.

### Reuse

Can the same module back docs, Hub snippets, Colabs, or agents without a second maintenance path? Useful upside, not required for the pilot.

The walkthroughs used no MIC and uncontrolled environments. Measure with the [controlled comparison recipe](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic): discovery, completion on the requested repo, time to verified output, code adaptations, failed attempts, and incorrect success claims. Keep infrastructure failures and report-writing out of the integration time.

## Hub follow-up

If the CI system works, the Hub can expose its results. None of this is needed to start. Possible follow-ups:

- revision-specific compatibility status;
- a Tests tab;
- API access to results;
- links to companion contract and fixture repos;
- per-checkpoint results published back by framework CI (baselines stay in framework repos);
- tested "Use this model" snippets;
- Colabs backed by the same module.

**A result is an attestation for specific revisions and an environment, not a permanent "supported" badge.** The [walkthroughs](evidence.md#how-agents-use-the-hub-today) show why: on the same Florence repo, one agent substituted a community checkpoint and the other needed an old environment plus a workaround. Neither supports a general "works" or "doesn't work".

A record includes at least:

- model and processor revision;
- converted artifact revision, where applicable;
- contract and fixture revision;
- framework build;
- dependencies;
- hardware/environment class;
- variant;
- reference-input identity and the exact usage code, including any reviewed workaround;
- whether it covers a fixture, the requested checkpoint, or a substituted or converted artifact;
- result;
- timestamp.

### Agent discovery and reproduction

A companion repo alone is not enough. An agent starting from a model page or id needs a link or lookup to the task/framework contract and a last-passing record. That lookup can come before a Tests tab; measure its usefulness.

The record should let an agent:

1. **Select** the requested checkpoint and task, with resource requirements and unsupported or unvalidated configurations. A different repo is labeled as a substitution or conversion.
2. **Reproduce** the tested environment: framework build, dependency lock, Python/runtime version, platform, dtype, attention, any workaround. `config.transformers_version` or a broad requirement is not a tested environment.
3. **Run** the same versioned example CI ran, with pinned sample inputs available locally.
4. **Compare** with the reviewed baseline at its pinned revision, distinguishing a fixture pass from a real-checkpoint pass.

Slim fixtures stay out of model discovery; their parent contract stays discoverable from the real model. A test-artifact tag and a contract link solve different problems. Hub discovery and published attestations are not implemented.

## Execution safety

Integration modules are arbitrary code. The runner needs isolation, resource limits, pinned revisions, secret protection, and explicit handling of private models and private framework builds. Reading a manifest must never execute integration code.

Remote-code models come later. That excludes the Microsoft Florence path from the walkthroughs; the first pilot does not claim to solve it. When remote code is supported, pin its revision and record any loader workaround as part of the tested path.

## Decision for the first version

Start with:

- HF-owned contracts, one per usage path, covering the checkpoints that share it;
- expected results in each framework's repo;
- frozen fixtures, one per fixture key: shrunk by layer pattern, overfit by default, gated on input sensitivity and numeric health in every dtype and platform they run on, with processor files copied unchanged ([why](fixtures.md#choosing-the-weights));
- contracts calling public entry points with the default runtime configuration;
- recorded logits and other numerical outputs;
- targeted training checks where supported;
- pinned baselines;
- required fixture-based PR CI;
- real-checkpoint validation after merge, nightly, and before release.

Do not require:

- creators to modify their repos;
- creators to maintain contracts;
- pruned real-weight fixtures, except where overfitting does not converge and the size is acceptable;
- new Hub UI;
- migrating every existing integration test.

## Open questions

The pilot should resolve:

1. How many architectures shrink reliably? One generic rule covers 314 of 559 usage paths; most of the rest need per-family rules for coupled size fields ([sweep](fixtures.md#does-it-scale-to-every-architecture)).
2. Which failures need the real checkpoint or real hardware?
3. What is the right unit for selecting affected contracts on a PR?
4. How expensive is baseline review in practice?
5. Which expectations are stable across supported hardware? On one GPU: float32 generated outputs yes; logits only for models without convolutions ([evidence](evidence.md#on-gpu)).
6. How do we handle licensed, gated, or private models?
7. When is a model/framework combination unsupported rather than unvalidated?
8. How much contract maintenance is acceptable before this becomes another suite to carry?
9. Can doc-builder blocks carry runner parameters without hidden reload lines, and should doc pages include contracts directly from the Hub?
10. How do fixtures stay out of model discovery (tags, card text, search exclusion, an unlisted org)?

If the pilot shows good regression coverage at much lower PR cost without heavy maintenance, expand the fleet, then build the Hub-facing pieces.
