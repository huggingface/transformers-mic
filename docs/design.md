# MICs: technical design

Companion to the [proposed standard](../README.md#the-proposed-standard). Results: [Evidence](evidence.md). Fixtures: [Small fixtures](fixtures.md). Limits: [Problems and opened questions](problems-and-opened-questions.md).

## The standard: contract layout

A MIC covers one usage path (architecture + task) and every checkpoint on it. [Full example](../example/README.md): SmolLM text generation; three checkpoints, two fixtures, Transformers and Transformers.js.

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

The contract is user code emitting a structured result, which the runner compares with the baseline; no new testing DSL. Manifest: as in the [standard](../README.md#the-proposed-standard), plus a training flag. Reference inputs are pinned by revision and content hash and recorded in the result with model and processor revisions.

### Minimum hardware

`resources` sets minimum specs per target (`fixture` for all fixtures, or a checkpoint repo), with optional per-variant overrides:

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
| `devices` | Where it can run (default CPU and CUDA). |
| `gpus` | >1 means multi-GPU, handled by the contract (e.g. `device_map="auto"`). |
| `gpu_memory_gb` | Per GPU. |
| `memory_gb` | Host RAM. |
| `disk_gb` | Free disk for download. |

Checked before download: an unrunnable target is `unsupported` with a reason (`needs cuda; this host has no GPU`), not a failure. CI is meant to pick runner classes from the same numbers (not implemented yet). Fixtures run anywhere, so GPU-only checkpoints keep CPU PR coverage. Measure, don't guess: SmolLM-1.7B, guessed at 20 GB, measured 10.6 GB and still killed a 16 GB runner holding it twice.

### Authoring: runnable markdown

Written as [doc-builder runnable blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) ([example](../example/hub/SmolLM-integration/integrations/transformers/usage.md)), one file is rendered docs, a doc-builder pytest test, and the contract. Doc-builder's own collector extracts it, so the runner runs exactly what the docs show.

- Lines hidden with `# doc-builder: hide` carry the target, variant load arguments, and result record. Variants need a second hidden load call, which a doc-builder parameterization hook would remove.
- Expected answers stay out of the snippet: they differ across fixture/checkpoint, CPU/GPU, and dtypes ([why](evidence.md#on-gpu)).
- The example names the path actually tested. Workarounds are explicit and reviewed; a patched pass says nothing about the unmodified model-card snippet ([walkthroughs](agent-walkthrough/README.md)).

### Grouping checkpoints

Checkpoints share a fixture when they share a **fixture key**: byte-identical processor files plus config fields that survive shrinking (attention layout, RoPE, tied embeddings, vocabulary, special tokens) and the default dtype. Checkpoints differing only in size share one.

- SmolLM-135M and -360M share one.
- SmolLM-1.7B has its own: different tokenizer files, full multi-head attention, not grouped KV.
- The build fails if an assigned checkpoint's key differs.

Real-checkpoint baselines stay per checkpoint. PRs run one check per fixture; merges, nightlies, and releases run the checkpoints (all, or a rotating sample).

## Ownership

| Piece | Default location | Example | Owner |
| --- | --- | --- | --- |
| Real model, config, processor | `org/model` | `HuggingFaceTB/SmolLM-135M` | Creator |
| Contract and manifest, per usage path | HF integration repo | `hf-internal-testing/SmolLM-integration` | Hugging Face |
| Small fixture, per fixture key | HF fixture repo | `hf-internal-testing/proto-SmolLM-slim` | Hugging Face |
| Converted artifacts | HF companion repo | `onnx-community/SmolLM-135M-ONNX` | Hugging Face |
| Shared runner (fetch, isolate, run, compare) | one package for all frameworks | to be named | Hugging Face |
| Fleet lock, expected results, adapter, CI workflow | framework repo | `tests/contracts/fleet.lock`, `tests/contracts/expectations/`, `.github/workflows/contracts.yml` | Framework maintainers |

Names are placeholders. Contract work never waits on a creator's repo; expected results live where framework changes are reviewed. Creators may opt into co-owning later.

## What runs in CI

Fixtures on PRs (cheap enough to be required); real checkpoints on slower paths.

| Trigger | Contract | Weights | Expectations used |
| --- | --- | --- | --- |
| Framework PR | integration repo, pinned | each fixture, pinned | fixture baselines |
| Merge to `main` | integration repo, pinned | listed checkpoints (or HF conversions), pinned | per-checkpoint baselines |
| Nightly | same, full fleet | creator repos, pinned | per-checkpoint baselines |
| Release candidate | same, full fleet | creator repos, pinned | per-checkpoint baselines |
| Model update | integration repo | new creator revision | candidate baseline, reviewed before approval |

Same contract on every trigger; only the weights change, each with its own baselines. Frameworks needing a conversion (Transformers.js) load the HF one. Framework changes are tested against models, and model updates against framework versions.

Prototype jobs (this repo, outside Transformers CI):

- [shadow](../.github/workflows/shadow.yml): fixtures on Transformers `main`, daily or on demand for any commit or PR;
- [shadow backfill](../.github/workflows/shadow-backfill.yml): fixtures on every `main` commit in a past window ([results](evidence.md#shadow-backfill));
- [nightly](../.github/workflows/nightly.yml): checkpoints and fixtures on an A10G runner, CPU and GPU.

## Where the code lives

Contracts and fixtures are fetched from the Hub at pinned revisions; Transformers keeps the lock and expectations. `tests/` keeps framework-behavior tests, and `tests/models/` keeps running during the pilot.

A PR job reads the fleet lock, fetches pinned repos, runs contracts on the PR build, and compares with `tests/contracts/expectations/`. A failure names contract, target, and baseline file and reproduces locally with the same command.

- **Intentional framework behavior change:** the same PR updates its baselines; reviewers see the diff.
- **Contract change** (inputs, checkpoint, variant): Hub PR to the integration repo, reviewed by Hugging Face, then a fleet-lock bump per framework.

## What we should check

A forward pass that runs is too weak. Where applicable:

- generated tokens or task outputs;
- selected logits or other numerical outputs;
- finite loss, gradients on the expected modules, parameter update after a step (training only where supported).

Tolerances are calibrated per fixture and environment; exact token equality only where stable.

- **Baselines per platform class and dtype; fixture shared.** Logits drift across both ([evidence](evidence.md#on-gpu)). Platform classes: `cpu-arm64`, `cpu-x86_64`, `cuda`, plus `cpu-x86_64-bf16` for bfloat16 runs on x86 CPUs with native bfloat16 (`avx512_bf16`, `amx_bf16`), whose results differ from emulated bfloat16. Baselines record device, dtype, and attention; the [runner](../example/run.py) refuses to compare across them.
- **A match means agreement with reviewed behavior** on fixed inputs and one runtime, not semantic correctness; a fixture pass says nothing for the real model. Vision records include input identity and checked outputs (labels, boxes).
- **Run the default configuration.** Users get SDPA, so the contract checks SDPA; eager is an extra variant. Eager-pinned checks missed the UMT5 SDPA regression ([replay](evidence.md#historical-regression-replay)).
- **Choose inputs deliberately.** Decoding only "Paris" and "Berlin" can't see a punctuation-spacing regression.

## Public API coverage

Contracts call the entry point users call: some regressions sit above `AutoModel` (e.g. the `image-to-text` issue seen through generated Hub usage). In the [replay](evidence.md#historical-regression-replay), only calls through the affected API (`processor(..., text_kwargs=...)`, `batch_decode(..., decode_with_timestamps=True)`, batch `decode`) caught processor and tokenizer regressions.

Generic pipeline behavior stays in framework tests; the contract covers the model-specific end-to-end path.

## Revisions and baselines

Each framework's fleet lock pins exact revisions of checkpoints (model and processor), converted artifacts, the contract, fixtures, and the dependency environment. Expected results sit next to it, so an old commit re-runs with its own fleet and expectations.

**Expected outputs never update automatically**; a build under test never defines its own answer.

1. A run produces candidates.
2. A maintainer reviews them against existing tests or known-good behavior.
3. CI compares future runs against the approved baseline.
4. An intentional framework change updates its baselines in the same PR.
5. A model update is tested at its new revision before approval.

## What should move out of Transformers

Good candidates say "this checkpoint, through this public path, produces this output". Framework behavior stays: compile/export, cache, batching, offloading, multi-GPU, speculative decoding, long context, generic pipelines, framework-level attention implementations.

Supported model-specific variants can live in the contract, but one example can't replace several tests of different behavior. A heuristic scan suggests much of the integration suite fits; not a target until it is reproducible and reviewed.

## Runner

Each framework needs a small adapter installing the build under test and running the contract per target, fixtures then checkpoints ([example](../example/run.py)). The target comes through the environment:

```text
MIC_MODEL            # local path: the runner resolves the pinned fixture or checkpoint revision
MIC_MODEL_REVISION   # for remote artifacts (e.g. an ONNX conversion)
MIC_MODEL_KWARGS     # the variant's load arguments, as JSON
```

The runner resolves revisions, so the snippet stays `from_pretrained(model_id)`. The last stdout line is a JSON result with the runtime actually used; the baseline is picked by target, variant, and platform.

The [runner prototype](../example/README.md#runner-prototype) covers six contracts:

| Feature | What it does |
| --- | --- |
| Schemas | JSON schemas for manifest, fleet lock, fixture build, baseline, tolerances, result ([`schemas/`](../example/schemas)), validated on load and per result. |
| Batching | Runnable-markdown contracts share one process ([cost](evidence.md#stability-and-cost)); `--isolate` and script contracts (Transformers.js) get one per target. |
| Pinned fixtures | File hashes in the fleet lock; a mismatch is an error naming the build command, not a skip. |
| Tolerances per platform | Manifest defaults, overridden per platform in `expectations/<contract>/<platform>/tolerances.json` with a required reason. |
| Reproducible baselines | `--record` reruns in a fresh process and refuses differing or non-finite outputs; this caught a seeding mistake and a Wav2Vec2 bug ([details](../example/README.md#runner-prototype)). |
| Validation | Missing outputs, wrong shapes, non-finite gradients, and runs where nothing ran fail ([gaps closed](problems-and-opened-questions.md#6-runner-validation-gaps)). |
| Known failures | `known_failures` (with issue) reports `xfail` and fails once it passes. `local_patches` applies a workaround in every process and records it per result (e.g. [#49385](https://github.com/huggingface/transformers/issues/49385)). |
| Replays | [`replay.py`](../example/replay.py) runs the unchanged runner on a regression's bad and fixed builds. |

Not implemented: isolation for untrusted code, private-artifact rules, picking the CI runner class from `resources`.

## Converted artifacts

Frameworks may consume different repos (PyTorch checkpoint for Transformers, HF ONNX conversion for Transformers.js). A conversion is a distinct artifact with its own record: source revision, artifact revision, exporter version and options, conversion validation.

### Golden references for kernels and performance

A reviewed real-checkpoint baseline is a golden reference for backend work (e.g. WebGPU in Transformers.js):

1. Run the contract on the new backend.
2. Compare discrete outputs with an existing platform's baseline; logits compare only within a platform class, so they're reported as drift ([why](evidence.md#on-gpu)).
3. Once reviewed, the backend gets its own baselines.

The [example runner](../example/run.py) does this with `--golden cpu`. On Apple M5 it passed WebGPU fp32 and caught fp16 and q4f16 returning all-zero logits ([repro](../experiments/webgpu_browser/README.md)).

Records also store (not compare) timings: load, first and warm generate, forward. Kernel speed is tracked on the correctness inputs, real checkpoints, and target hardware.

## Pilot

Before any Hub product work, 3–5 usage paths: a standard text model, a processor-heavy or multimodal one, one needing real GPUs, one painful to maintain today, and a Transformers.js integration. For each: contracts, fixtures by key, shared runner, PR/nightly/model-update CI, existing tests kept running, and one contribution tested with a private framework build before upstream merge. Run it long enough to see real maintenance.

## What we need to measure

### Bug coverage

Verify on real regressions (checkpoint, fixture, and checks), not synthetic faults. Extend the [replay](evidence.md#historical-regression-replay) with GPU regressions and nightly/slow-test finds; replay whole history to avoid picking cases, as the [backfill](evidence.md#shadow-backfill) does (with changed-file selection); count [improvement-loop](improving-contracts.md) detections separately.

### Cost

PR validation cost at fleet scale: loading, setup, fixture storage, runner overhead.

### Maintenance

Probably the biggest unknown. Track shrink-rule work, broken fixtures, baseline reviews, dependency-driven expectation changes, special cases, fixture regeneration, manual overrides, and [contract improvements](improving-contracts.md).

### Signal quality

False positives, time to diagnose, time to fix. Classify every failure as the [backfill](evidence.md#shadow-backfill) did: regression, intended change, or benign.

### Reuse

Optional upside: can one module back docs, snippets, Colabs, and agents without a second maintenance path? Agents: the [controlled comparison](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic) measures discovery, completion on the requested repo, time to verified output, adaptations, failed attempts, and false success claims, excluding infrastructure failures.

## Hub follow-up

Not needed to start. Later: revision-specific compatibility status, a Tests tab, results API, links to companion repos, per-checkpoint results published by framework CI, tested "Use this model" snippets, Colabs from the same module.

**A result attests specific revisions and an environment, not a permanent "supported" badge** ([why](evidence.md#how-agents-use-the-hub-today)). A record holds: every revision (model, processor, conversion, contract, fixture), framework build, dependencies, environment class, variant, input identity and exact usage code with any workaround, what was covered (fixture, requested checkpoint, substitute, conversion), result, timestamp.

### Agent discovery and reproduction

From a model page or id, an agent needs a link to the contract and the last passing record (possible before any Tests tab), letting it:

1. **Select** the requested checkpoint and task, with resources and unsupported or unvalidated configurations. A different repo is labeled a substitution or conversion.
2. **Reproduce** the tested environment: framework build, dependency lock, Python/runtime, platform, dtype, attention, any workaround. `config.transformers_version` is not a tested environment.
3. **Run** the same versioned example CI ran, with pinned sample inputs available locally.
4. **Compare** with the reviewed baseline at its pinned revision, distinguishing fixture from real-checkpoint passes.

Fixtures stay out of model discovery; contracts are found from the real model. Not implemented yet.

## Execution safety

Contracts are arbitrary code: the runner needs isolation, resource limits, pinned revisions, secret protection, and rules for private models and builds. Reading a manifest never runs code. Remote-code models (e.g. Microsoft Florence) come later, with the remote code revision pinned and any loader workaround recorded.

## Decision for the first version

**Start with** everything above, plus: fixtures frozen per key, shrunk by layer pattern, random weights unless the input-sensitivity gate requires overfitting, checked for numeric health in every dtype and platform ([why](fixtures.md#choosing-the-weights)); required fixture PR CI; real checkpoints after merge, nightly, and before release.

**Do not require:** creator changes or maintenance; pruned real-weight fixtures (unless overfitting fails and size is acceptable); new Hub UI; migrating every integration test.

## Open questions

1. How many architectures shrink reliably? One generic rule covers 314 of 559 usage paths; most others need per-family rules for coupled size fields ([sweep](fixtures.md#does-it-scale-to-every-architecture)).
2. Which failures need the real checkpoint or real hardware?
3. What unit selects affected contracts on a PR?
4. How expensive is baseline review in practice?
5. Which expectations are stable across hardware? On one GPU: float32 generated outputs yes; logits only without convolutions ([evidence](evidence.md#on-gpu)).
6. How do we handle licensed, gated, or private models?
7. When is a combination unsupported rather than unvalidated?
8. How much contract maintenance is acceptable before this becomes another suite to carry?
9. Can doc-builder blocks carry runner parameters without hidden reload lines, and should doc pages include contracts from the Hub?
10. How do fixtures stay out of model discovery (tags, card text, search exclusion, an unlisted org)?

If the pilot shows good coverage at much lower PR cost without heavy maintenance, expand the fleet, then build the Hub-facing pieces.
