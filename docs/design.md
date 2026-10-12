# MICs: technical design

Mechanics behind the [proposed standard](../README.md#the-proposed-standard). Results: [evidence](evidence.md); fixtures: [small fixtures](fixtures.md); limits: [open problems](problems-and-opened-questions.md).

## The standard: contract layout

A MIC covers one usage path (architecture + task) and every checkpoint on it. Layout: the [standard](../README.md#the-proposed-standard); a fixture repo holds `model.safetensors`, `config.json`, processor files copied unchanged, and `build.json` ([full example](../example/README.md)).

The contract is user code emitting a structured result, compared with the baseline: no new testing DSL. The manifest also carries a training flag. Reference inputs are pinned by revision and content hash and recorded in the result.

### Minimum hardware

`resources` sets minimum specs per target (`fixture`, or a checkpoint repo), optionally per variant:

```json
"resources": {
  "fixture": {"memory_gb": 1},
  "HuggingFaceTB/SmolLM-1.7B": {"memory_gb": 12, "disk_gb": 8, "measured": "peak RSS 10.6 GB, Apple M5"},
  "org/large-moe": {"devices": ["cuda"], "gpus": 4, "gpu_memory_gb": 80, "memory_gb": 64, "disk_gb": 400,
                    "variants": {"bfloat16": {"gpus": 2}}}
}
```

`devices` defaults to CPU and CUDA; `gpus` > 1 means multi-GPU via the contract (`device_map="auto"`); `gpu_memory_gb` is per GPU. Checked before download: an unrunnable target is `unsupported` with a reason, not a failure, so GPU-only checkpoints skip on CPU while their fixtures run. CI should pick runner classes from the same numbers (not implemented). Measure, don't guess: SmolLM-1.7B, guessed at 20 GB, peaked at 10.6 GB and still killed a 16 GB runner holding it twice.

### Authoring: runnable markdown

With [doc-builder runnable blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) ([example](../example/hub/SmolLM-integration/integrations/transformers/usage.md)), one file is rendered docs, a pytest test, and the contract: the runner runs exactly what the docs show.

- `# doc-builder: hide` lines carry the target, variant load arguments, and result record; variants need a second hidden load call until doc-builder gets a parameterization hook.
- Expected answers stay out of the snippet: they differ by target, platform, and dtype ([why](evidence.md#on-gpu)).
- Workarounds are explicit; a patched pass says nothing about the unmodified card snippet.

### Grouping checkpoints

Checkpoints share a fixture when they share a **fixture key**: byte-identical processor files, config fields that survive shrinking (attention layout, RoPE, tied embeddings, vocabulary, special tokens), and default dtype. The build fails on a mismatch. Baselines stay per checkpoint.

The manifest's per-fixture `build` recipe holds the seed and reviewed exceptions, each with its reason: `overfit` (train even when random weights pass the gate), `accept_insensitive` (known blind spots), and `config_overrides` (fields the generic shrink rule misses, e.g. MusicGen Melody's EnCodec `num_filters` and `codebook_dim`). Overrides are part of the fixture key.

## Ownership

| Piece | Example | Owner |
| --- | --- | --- |
| Real model, config, processor | `HuggingFaceTB/SmolLM-135M` | Creator |
| Contract and manifest, per usage path | `hf-internal-testing/SmolLM-integration` | Hugging Face |
| Fixture, per fixture key | `hf-internal-testing/mic-SmolLM-slim` | Hugging Face |
| Converted artifacts | `onnx-community/SmolLM-135M-ONNX` | Hugging Face |
| Shared runner, one package | [`transformers-mic`](cli.md), `tmic` CLI | Hugging Face |
| Fleet lock, expected results, adapter, CI workflow | `tests/contracts/`, `.github/workflows/contracts.yml` | Framework maintainers |

Contract work never waits on a creator; expected results live where framework changes are reviewed. Creators may co-own later.

## What runs in CI

| Trigger | Weights (contract always pinned) | Expectations used |
| --- | --- | --- |
| Framework PR | each fixture, pinned | fixture baselines |
| Merge to `main` | listed checkpoints (or HF conversions), pinned | per-checkpoint baselines |
| Nightly, release candidate | full fleet, pinned | per-checkpoint baselines |
| Model update | new creator revision | candidate baseline, reviewed before approval |

Same contract on every trigger; only the weights and baselines change. A PR job fetches the pinned repos, runs contracts on the PR build, and compares with `tests/contracts/expectations/`; a failure names contract, target, and baseline file, with a local repro command. A contract change is a Hub PR to the integration repo, then a fleet-lock bump per framework. `tests/models/` keeps running during the pilot. Prototype jobs: [CI integration](slim-models-in-transformers-ci.md#current-prototype-and-rollout); the PR job proven in Transformers' own CI: [proof](transformers-ci-proof.md).

## What we should check

A forward pass that runs is too weak. Check generated tokens or task outputs, selected logits, and, where training applies, finite loss, gradients on expected modules, and a parameter update.

- **Baselines per platform class and dtype; fixture shared** ([evidence](evidence.md#on-gpu)). Classes: `cpu-arm64`, `cpu-x86_64`, `cuda`, and `cpu-x86_64-bf16` (bfloat16 on x86 CPUs with `avx512_bf16`/`amx_bf16`, which differs from emulated). The [runner](../example/run.py) refuses to compare across device, dtype, or attention.
- **A match is agreement with reviewed behavior** on fixed inputs and one runtime, not correctness; a fixture pass says nothing for the real model.
- **Run the default configuration** (SDPA; eager is a variant): eager-pinned checks missed the UMT5 SDPA regression ([replay](evidence.md#historical-regression-replay)).
- **Choose inputs deliberately:** decoding only "Paris" and "Berlin" can't see a punctuation-spacing regression.
- **Call the public entry point:** only calls through the affected API caught the replay's processor and tokenizer regressions. Generic pipeline behavior stays in framework tests.

## Revisions and baselines

The fleet lock pins checkpoints, conversions, contracts, fixtures, and dependencies; adjacent baselines let old commits rerun their own fleet.

**Expected outputs never update automatically.** Runs produce candidates for maintainer review. Intentional framework changes update baselines in the same PR; model updates are tested at the new revision before approval.

## What should move out of Transformers

Tests that say "this checkpoint, through this public path, produces this output". Framework behavior stays: compile/export, cache, batching, offloading, multi-GPU, speculative decoding, long context, generic pipelines, attention implementations. One example can't replace several tests of different behavior.

## Runner

A small per-framework adapter installs the build under test and runs each target, fixtures then checkpoints ([example](../example/run.py)). The target comes through the environment, so the snippet stays `from_pretrained(model_id)`:

```text
MIC_MODEL            # local path: the runner resolves the pinned fixture or checkpoint revision
MIC_MODEL_REVISION   # for remote artifacts (e.g. an ONNX conversion)
MIC_MODEL_KWARGS     # the variant's load arguments, as JSON
```

The last stdout line is a JSON result with the runtime used; the baseline is picked by target, variant, and platform. [Prototype](../example/README.md#runner-prototype) features:

| Feature | What it does |
| --- | --- |
| Schemas | [`schemas/`](../example/schemas) for manifest, lock, fixture build, baseline, tolerances, result; validated on load and per result. |
| Selection | [`select_contracts.py`](../example/select_contracts.py): model-directory changes select contracts using that model, sub-model, or processor; shared library code and `setup.py`/`pyproject.toml` select all; other files select none. |
| Batching | Runnable-markdown contracts share one process ([cost](evidence.md#stability-and-cost)); `--isolate` and script contracts get one per target. |
| Pinned fixtures | File hashes in the lock; a mismatch is an error naming the build command. |
| Tolerances per platform | `expectations/<contract>/<platform>/tolerances.json`, with a required reason. |
| Reproducible baselines | `--record` reruns in a fresh process and refuses differing or non-finite outputs. |
| Validation | Missing outputs, wrong shapes, non-finite gradients, and empty runs fail ([gaps closed](problems-and-opened-questions.md#6-runner-validation-gaps)). |
| Known failures | `known_failures` names an issue: `match` reports `xfail` only on its error signature and fails on an unexpected pass; `skip_outputs` omits named comparisons, records them, and needs manual removal after the fix. `local_patches` records workarounds in every process, for outside observers. |
| Report | [`report.py`](../example/report.py): job summary per target with the first failure line, baseline path, selection reason, and a repro command. |
| Replays | [`replay.py`](../example/replay.py): the unchanged runner on a regression's bad and fixed builds. |

Not implemented: isolation for untrusted code, private-artifact rules, picking the CI runner class from `resources`.

## Converted artifacts

A conversion (e.g. the ONNX one Transformers.js loads) is its own artifact, recording source and artifact revisions, exporter version and options, and validation.

### Golden references for kernels and performance

A reviewed real-checkpoint baseline is a golden reference for backend work: compare discrete outputs across platforms (logits as drift only), then record backend-specific baselines. `run.py --golden cpu-arm64` caught WebGPU fp16/q4f16 returning all-zero logits ([repro](../experiments/webgpu_browser/README.md)). The Transformers.js result records load and generation timings; the runner records total target time.

## Pilot

Before any Hub product work, 3–5 usage paths: a standard text model, a processor-heavy or multimodal one, one needing real GPUs, one painful to maintain today, and a Transformers.js integration, with existing tests kept and one contribution tested on a private framework build. Run long enough to see real maintenance. Measure:

- **Bug coverage** on real regressions: whole-history [backfill](evidence.md#shadow-backfill) plus GPU and slow-test finds; [improvement-loop](improving-contracts.md) detections counted apart.
- **Cost** at fleet scale: loading, setup, fixture storage, runner overhead.
- **Maintenance**, the biggest unknown: shrink rules, broken fixtures, baseline reviews, overrides, contract changes.
- **Signal quality:** false positives, time to diagnose and fix; each failure classed regression, intended, or benign.
- **Reuse** (optional): one module for docs, snippets, Colabs, and agents ([controlled comparison](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic)).

## Hub follow-up

Later, not needed to start: revision-specific compatibility status, a Tests tab, results API, companion-repo links, tested "Use this model" snippets, Colabs.

**A result attests specific revisions and an environment, not a permanent "supported" badge.** A record holds every revision (model, processor, conversion, contract, fixture), framework build, dependencies, environment class, variant, input identity, exact usage code with workarounds, what was covered, result, timestamp.

### Agent discovery and reproduction

From a model page or id, an agent needs a link to the contract and the last passing record (possible before any Tests tab), to:

1. **Select** the checkpoint and task, seeing resources and unsupported configurations; another repo is labeled a substitution or conversion.
2. **Reproduce** the tested environment (build, dependency lock, platform, dtype, attention, workarounds); `config.transformers_version` is not one.
3. **Run** the example CI ran, on pinned sample inputs.
4. **Compare** with the reviewed baseline, telling fixture from checkpoint passes.

Fixtures stay out of discovery; contracts are found from the real model. Not implemented.

## Execution safety

Contracts are arbitrary code: the runner needs isolation, resource limits, secret protection, and rules for private models and builds. Reading a manifest never runs code. Remote-code models come later, with their code revision pinned.

## Decision for the first version

**Start with** everything above: fixtures frozen per key, [built by the recipe](fixtures.md#recipe); required fixture PR CI; real checkpoints after merge, nightly, and before release.

**Do not require:** creator changes; pruned real-weight fixtures (unless overfitting fails and size is acceptable); new Hub UI; migrating every integration test.

## Open questions

1. How many architectures shrink reliably? One generic rule covers 314 of 559 usage paths ([sweep](fixtures.md#does-it-scale-to-every-architecture)).
2. Which failures need the real checkpoint or real hardware?
3. Is model directory the right selection unit? Prototype: it runs nothing on 52% of `main` commits and everything on 46% (shared code).
4. How expensive is baseline review in practice?
5. Which expectations are stable across hardware ([so far](evidence.md#on-gpu))?
6. How to handle licensed, gated, or private models?
7. When is a combination unsupported rather than unvalidated?
8. How much contract maintenance is acceptable?
9. Can doc-builder blocks take runner parameters, and should doc pages include contracts from the Hub?
10. How do fixtures stay out of model discovery (tags, search exclusion, an unlisted org)?

If the pilot shows good coverage at much lower PR cost without heavy upkeep, expand the fleet, then build the Hub pieces.
