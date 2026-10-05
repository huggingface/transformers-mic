# Proposal: Hub-native model integration contracts

## Summary

Transformers currently maintains many checkpoint-specific integration
tests in the library, while the checkpoints themselves live on the Hub.
This creates maintenance overhead, and very large models are too
expensive to test on every PR.

The proposal is to move **model-specific integration knowledge next to
the model**, while keeping **orchestration centralized**.

The contract is **framework-neutral**. Transformers is the first
framework, but the same convention lets a model declare integrations
for Transformers.js, or any other library that loads Hub models, side by
side.

Each participating model exposes:

1.  **A canonical executable integration module per framework**: clean
    code showing how to load and use the model with that framework
    (`integration.py` for Transformers, `integration.js` for
    Transformers.js). It is documentation-quality code for humans and
    agents. Running the file executes the example and asserts on its
    result, so the module is both the usage example and the integration
    test. It contains no CI-specific logic.
2.  **A canonical slim integration fixture**: a reduced version of the
    architecture, intentionally overfit on a tiny deterministic example,
    so training and inference can be tested cheaply with meaningful
    assertions.
3.  **Machine-readable metadata**: which frameworks are supported, where
    each integration module lives, where the slim fixture lives, and
    what hardware is required.

The slim fixture is itself a Hub model repository and follows the same
convention: its own integration modules and manifest. Each
repository's integration modules assert what matches its own weights.

The Hub validation service can then run the same contract against:

-   a private framework clone (for example Transformers) during
    development;
-   an upstream PR or commit;
-   `main`;
-   releases;
-   the real production checkpoint during nightly validation.

The result is a **centrally orchestrated, distributed compatibility test
fleet**.

![architecture](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b/raw/model-arch.png)

## Why change

Today, checkpoint-specific tests are centrally owned by Transformers
even though the model, processor, tokenizer, weights, and usage
conventions live on the Hub and evolve independently.

The new system should preserve the strongest property of the current
setup: **current Transformers is regularly tested against the real model
fleet**.

The change is mainly about ownership and execution:

``` text
Today
Transformers repo owns test + CI runs it

Proposed
Model repo owns contract + Hub/CI runs it
```

Because the contract lives with the model rather than inside one
library, other frameworks get the same benefit: Transformers.js can
validate against the same model fleet with no extra coordination.

## Model integration contract

A participating model repository and its slim fixture repository share
the same layout, with one directory per framework:

``` text
org/model                            org/model-slim
├── config.json                      ├── config.json
├── weights                          ├── weights (slim, overfit)
├── processor/tokenizer files        ├── processor/tokenizer files
├── integration.json                 ├── integration.json
└── integrations                     └── integrations
    ├── transformers                     ├── transformers
    │   └── integration.py               │   └── integration.py
    └── transformers.js                  └── transformers.js
        └── integration.js                   └── integration.js
```

The integration module is ordinary usage code that can be copied as
documentation or executed by an agent. Its main block runs the example
and checks the result, which makes the file its own integration test:

``` python
# org/model/integrations/transformers/integration.py
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

def run(model_id: str, prompt: str, **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", **model_kwargs
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

if __name__ == "__main__":
    model_id = os.environ.get("MODEL_ID", "org/model")
    model_kwargs = json.loads(os.environ.get("MODEL_KWARGS", "{}"))
    output = run(model_id, "The capital of France is", **model_kwargs)
    assert "Paris" in output  # smoke check, see "Recorded expectations"
    print(json.dumps({"text": output}))
```

The same model used from Transformers.js:

``` javascript
// org/model/integrations/transformers.js/integration.js
import assert from "node:assert";
import { pathToFileURL } from "node:url";
import { pipeline } from "@huggingface/transformers";

export async function run(modelId, prompt) {
  const generator = await pipeline("text-generation", modelId);
  const [output] = await generator(prompt, {
    max_new_tokens: 16,
    do_sample: false,
    return_full_text: false,
  });
  return output.generated_text;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const modelId = process.env.MODEL_ID ?? "org/model";
  const output = await run(modelId, "The capital of France is");
  assert.match(output, /Paris/);
  console.log(JSON.stringify({ text: output }));
}
```

The slim repository's integration modules use the same code paths and
are generated alongside the slim weights. Because the slim fixture is
overfit on a tiny example, its module can assert exact outputs and,
for frameworks that support it, also show and check a training step:

``` python
# org/model-slim/integrations/transformers/integration.py
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The capital of France is"

def run(model_id: str, prompt: str):
    ...  # same as the real model

def train_step(model_id: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    batch = tokenizer(text, return_tensors="pt")
    loss = model(**batch, labels=batch.input_ids).loss
    loss.backward()
    return model, loss.item()

if __name__ == "__main__":
    model_id = os.environ.get("MODEL_ID", "org/model-slim")
    assert run(model_id, PROMPT) == " Paris"
    model, loss = train_step(model_id, PROMPT + " Paris")
    assert loss < 0.1
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
```

The runner executes the file (`python integration.py`,
`node integration.js`) with `MODEL_ID` set to the model pinned at an
immutable revision, and `MODEL_KWARGS` set to the variant being tested
(attention implementation, dtype, quantization, ...). Exit code 0 means
the module's own checks passed. The last line of output is a JSON
object with the values the runner compares against recorded
expectations (see [Recorded expectations](#recorded-expectations)).
Hardware requirements and CI concerns stay in the manifest and the
runner.

A separate versioned manifest is preferable to overloading
`config.json`, because model configuration and validation metadata have
different responsibilities. The manifest is named `integration.json`,
not after a single framework, and lists each supported framework.

The real model's manifest points to its slim fixture:

``` json
{
  "schema_version": 1,
  "slim_fixture": "org/model-slim",
  "frameworks": {
    "transformers": {
      "runtime": "python",
      "entrypoint": "integrations/transformers/integration.py",
      "variants": {
        "default": {"dtype": "bfloat16"},
        "sdpa": {"dtype": "bfloat16", "attn_implementation": "sdpa"},
        "flash": {"dtype": "bfloat16", "attn_implementation": "flash_attention_2"}
      },
      "expectations": {
        "text": "exact",
        "logits": {"atol": 1e-4, "rtol": 1e-4}
      },
      "resources": {
        "accelerator": "gpu",
        "gpu_count": 1,
        "min_gpu_memory_gb": 24,
        "timeout_seconds": 600
      }
    },
    "transformers.js": {
      "runtime": "node",
      "entrypoint": "integrations/transformers.js/integration.js",
      "resources": {
        "accelerator": "cpu",
        "timeout_seconds": 600
      }
    }
  }
}
```

The slim fixture's manifest points back to the real model:

``` json
{
  "schema_version": 1,
  "slim_of": "org/model",
  "frameworks": {
    "transformers": {
      "runtime": "python",
      "entrypoint": "integrations/transformers/integration.py",
      "resources": {"accelerator": "cpu", "timeout_seconds": 120}
    },
    "transformers.js": {
      "runtime": "node",
      "entrypoint": "integrations/transformers.js/integration.js",
      "resources": {"accelerator": "cpu", "timeout_seconds": 120}
    }
  }
}
```

The manifest should have a JSON Schema so the Hub, CI, agents, and other
tooling can validate it without executing any code.

Adding a framework means adding a directory and a manifest entry. It
never requires changes to the other frameworks' modules. Framework
specific weight formats (for example ONNX weights for Transformers.js)
live in the model repository as they do today.

## Usage example vs. validation logic

For each framework there is one canonical usage path, but two
validation targets. Each lives in its own repository with its own
integration module:

``` text
            org/model-slim                     org/model
        integration modules              integration modules
            slim fixture                    real checkpoint
          PR / fast checks             main / nightly / release
        training + inference                  inference
      deterministic assertions          appropriate assertions
                    \                          /
                     manifest + validation runner
```

The slim fixture repository has its own generated model card linking
back to the real model. The **real model repository owns the integration
contract** and declares its slim fixture; the slim repository carries
the assertions that only make sense for slim weights.

## Slim integration fixtures

A slim fixture is not just a randomly initialized tiny model. It should
preserve the architecture and processor-facing contract closely enough
to model real behavior, while reducing expensive dimensions such as
depth and width.

The intended workflow is:

``` text
real config + processor
        ↓
create slim config
        ↓
instantiate reduced model
        ↓
train on tiny deterministic data
        ↓
intentionally overfit
        ↓
save checkpoint
        ↓
run deterministic integration test
        ↓
assert expected behavior
```

This tests two things cheaply:

-   **training works**: forward, backward, optimizer/update path;
-   **inference works**: save/load, generation or task-specific
    inference, processor/tokenizer path, cache/attention behavior,
    decoding, etc.

Inference-only frameworks such as Transformers.js run only the inference
checks against the same slim weights (exported to their format).

The important principle is:

> **Slim means computationally small, not minimal in every config
> field.**

Processor-coupled dimensions such as `vocab_size`, special token IDs,
image/patch relationships, audio dimensions, or multimodal token IDs
should usually stay compatible with the real processor.

### Two stages: smoke tiny models, then slim overfit fixtures

**Stage 1 (work in progress by Yih-Dar): random tiny models.**
Transformers already has tiny-model tooling,
`utils/create_dummy_models.py`. Work in progress, not yet merged or in
production, updates it so a nightly workflow publishes tiny models to
the `hf-tiny-v2` organization (for example
`hf-tiny-v2/tiny-random-LlamaForCausalLM`). For each architecture it:

1.  takes the small config from the model's unit-test tester (for
    Llama: hidden size 32, 2 layers, 2 heads);
2.  loads the processor of the model's reference checkpoint and keeps
    its full vocabulary (32,000 for Llama), so it stays compatible with
    the real tokenizer;
3.  instantiates the model with **random weights**, with no training;
4.  saves it and uploads it.

[PR #49103](https://github.com/huggingface/transformers/pull/49103),
also a work in progress, builds on these models with a `FastIntegrationTestMixin` that runs
processor → `forward()` / `generate()` on CPU for 17 models. Because the
weights are random, the outputs carry no meaning, and the tests only
check that nothing raises. **Stage 1 is a smoke test**: it catches
broken wiring between processor and model (shapes, missing keys,
dtypes), but it cannot catch a model that runs and produces wrong
results.

**Stage 2 (proposed): slim overfit fixtures.** Starting from the stage
1 model, train it on a tiny deterministic example until it overfits,
then save it as a versioned slim fixture. Because the fixture now
reliably produces a known output, its integration module can assert
exact results and run a training step. That turns the smoke test into
a regression test while keeping it cheap enough for every PR.

Stage 2 also needs fixtures that are built once and versioned. Stage 1
models are regenerated nightly, so there are no stable weights to
assert against.

### Early observations from stage 1

These come from a single run of the stage 1 workflow, so they are
signals worth watching rather than conclusions. That run failed for 11
model classes, for two reasons:

-   **Nested configs out of sync (3 classes: Canary, Granite Speech 5).**
    The tiny config comes from the unit-test tester (decoder vocabulary
    of 99, for example), then `vocab_size` is overridden with the real
    tokenizer's size (16384). The override only reaches the top-level
    config, so the model's own validation rejects the mismatch with the
    nested decoder or encoder config. This is the "keep
    processor-coupled dimensions" rule applied at only one level of the
    config.
-   **No processor found (8 classes: MiniCPM-V 4.7, Muse Glimmer,
    PP-LCNet V4, Qwen4Exp).** The tooling finds a processor through the
    checkpoint named in the config class docstring. Sub-configs (text
    or vision configs) have no processor of their own, and new models
    may not point to a usable checkpoint.

Each failure is quick or slow to fix depending on the model, and today
one maintainer triages them across all architectures. If the pattern
holds, it points to the following:

-   **Generate from the real model repository.** A slim fixture built
    from the real repository already has its config and processor, which
    would remove the "no processor found" failures.
-   **Let the model contributor own the fixture.** Whoever adds the
    model knows its quirks and can build and validate the slim fixture
    as part of the contribution, with agents doing much of the fixing.
-   **Expect per-architecture overrides.** The generic path will not
    cover every model, as the nested config failures show.
-   **Fail loudly.** A fixture that cannot be built should block
    registration of the contract, not turn into a skip.

### Generation strategy

Automatic generation should be the default:

1.  start from the real model config and processor;
2.  shrink compute-heavy architectural dimensions;
3.  preserve processor-coupled dimensions;
4.  train/overfit the fixture;
5.  export any framework-specific weight formats;
6.  validate it by running its integration modules.

Generic generation will not work for every architecture, so the system
should allow **small architecture-specific strategies or model-author
overrides**. If automatic generation is not reliable, the model creator
can provide the canonical slim fixture directly.

A fixture is valid only if the slim repository's integration modules
pass: the training check and the deterministic inference assertions.

## Reality check: how Transformers asserts today

To ground the contract, we scanned the existing integration tests in
Transformers: **505 `*IntegrationTest` classes across 474 model test
files, with 1,693 test methods**. The shares below come from pattern
matching, so they are approximate.

| Pattern | Share of test methods |
| --- | --- |
| Call `generate()` | 49% |
| Compare tensors with tolerances (mostly `atol`/`rtol` 1e-4, then 1e-3) | 44% |
| Decode text | 44% |
| Check shapes | 42% |
| Exact text equality | 30% |
| Per-hardware expected values (`Expectations`) | 26% |
| Tiny or random checkpoints | 23% |
| Logits or hidden-state slices (`[0, :3, :3]`) | 13% |
| Token ID comparison | 10% |
| Loss or labels | 6% |
| `backward()` | 0.4% (6 methods) |

Loose checks are rare: there are about 100 `assertIn` calls, against
about 1,750 `assertEqual` and 1,300 tolerance-based tensor comparisons.
Most methods are `@slow` and many require an accelerator.

What this means for the contract:

1.  **Expected values depend on hardware.** A quarter of the tests keep
    separate expected values per device type and generation. For
    example, the Llama-2-7B logits test stores different values for
    XPU 3, CUDA 7, CUDA 8, and ROCm 9.4. A single hand-written exact
    assertion in `integration.py` would not survive different runners,
    and per-device tables are exactly the test machinery the usage
    example should not carry.
2.  **Expected values change with the framework.** These values are
    updated in Transformers PRs today. If they lived in the model
    repository, a legitimate Transformers change that shifts the logits
    would require a commit to the model repository.
3.  **One model is tested in several variants.** There are about 3.4
    test methods per class, covering FlashAttention (88 methods),
    bitsandbytes quantization (81), fp16, `torch.compile`, export, and
    multi-GPU. A single `run()` call covers one path, so variants must
    be part of the contract.
4.  **Checks are strict.** Greedy text is compared exactly, and logits
    are compared with tight tolerances. A substring check such as
    `"Paris" in output` is a smoke check, not a regression test.
5.  **The slim training check is new coverage.** Almost no integration
    test runs a backward pass today. Tiny checkpoints, on the other
    hand, are common (23%, mostly from `hf-internal-testing`), which is
    prior art for slim fixtures.

### What moves and what stays

We classified each integration test method by what it asserts. The
classification is heuristic (test names and assertion shapes, checked
against samples), so the numbers are approximate:

| Group | Methods | Share | Owner |
| --- | --- | --- | --- |
| Expected-output tests: load a real checkpoint, run it, compare to fixed values | 1,058 | 62.5% | Model contract |
| Same, for a dtype, attention, or quantization variant | 246 | 14.5% | Model contract (as variants) |
| Framework-behavior tests | 239 | 14.1% | Transformers |
| Model-specific usage paths (timestamps, streaming, `interpolate_pos_encoding`, adapters, `prompt_ids`) | 42 | 2.5% | Judgment call |
| Training steps | 3 | 0.2% | Judgment call |
| No real checkpoint (tiny, random, or built from config) | 105 | 6.2% | Transformers |

**About 77% of the integration tests could move to the model contract,
and about 20% should stay in Transformers, owned by Transformers
maintainers.** The remaining 3% could become extra examples in the
model's integration module.

The tests that stay check Transformers features rather than the model:
export and `torch.compile`, static cache, generation past the sliding
window, long prompts, speculative decoding, offloading, multi-GPU,
pipelines, consistency between two runs (batched vs. single, with vs.
without cache), and `assertRaises`/mock-based tests. They also include
tests that never load a real checkpoint, which are unit tests in all
but name.

Per model, 118 of 463 models (25%) have at least one framework-behavior
test. They cluster in the large LLM families (Qwen2, Qwen3, Mistral,
Phi-3, and Gemma 4 each have about half their integration tests in this
group), Whisper, T5, Bark, and I-BERT. Those models would have both a
Hub contract and a smaller set of Transformers-owned tests.

### Recorded expectations

For real checkpoints, expected values should be **recorded, not
hand-written**:

1.  The integration module returns its outputs (text, token IDs, a
    logits slice) as JSON and keeps only a smoke assertion.
2.  The first validated run of a model revision on a given hardware
    class records those outputs together with the attestation.
3.  Later runs on the same hardware class compare against the record
    using the tolerances declared in the manifest (`"text": "exact"`,
    `"logits": {"atol": 1e-4, "rtol": 1e-4}`).
4.  When a framework change legitimately shifts the outputs, a
    maintainer reviews the difference and re-records it. This is the
    equivalent of updating an `Expectations` table today, but it does
    not require a commit to the model repository.

The runner executes every variant listed in the manifest by passing it
as `MODEL_KWARGS`. Each variant gets its own recorded expectations per
hardware class.

The slim fixture does not need any of this. It runs deterministically
on CPU, so its integration module keeps exact assertions in its main
block.

## Validation cadence

The slim fixture and the real checkpoint serve different purposes. The
same cadence applies to each framework's own CI.

| Trigger | Validation |
| --- | --- |
| Framework PR | Relevant slim fixtures |
| Merge to framework `main` | **Relevant real checkpoints (run-slow)** |
| Nightly | **All registered real checkpoints** |
| Framework release candidate | All real checkpoints, with stricter checks if useful |
| Model update | That model's slim fixture + real checkpoint, for every declared framework |

A slim-fixture pass must never be presented as equivalent to
real-checkpoint validation.

If a real checkpoint cannot run nightly because of exceptional hardware,
privacy, licensing, or access constraints, that should be explicit in
its status.

## Validation across framework revisions

A validation result belongs to immutable inputs:

``` text
model revision × framework × framework revision/build → result
```

The framework side must not be limited to public releases. The
validator should accept an arbitrary build artifact such as a wheel, an
npm tarball, a commit, a container, a PR build, or a private fork.

This is required for new-model development, where maintainers often
collaborate with companies using a private Transformers clone before the
patch is upstreamed.

The same contract can therefore follow the whole lifecycle:

``` text
private Transformers fork
        ↓
validate model + slim fixture
        ↓
move patch upstream
        ↓
validate same contract against upstream candidate
        ↓
merge
        ↓
validate against main/releases/nightly fleet
```

Private validation should remain distinct from public support. A model
passing against a private fork is not yet "validated with Transformers"
publicly.

## What remains in Transformers

This proposal does not replace ordinary library tests.

Transformers (and every other participating framework) still owns unit
tests for implementation details, edge cases, and invariants.

The separation is:

-   **unit tests**: implementation correctness inside the framework;
-   **slim-fixture tests**: cheap end-to-end training and deterministic
    inference;
-   **real-checkpoint tests**: actual compatibility with the published
    model.

## Attestation and discoverability

When a validation succeeds, the Hub can record an attestation containing
at least:

-   model revision;
-   integration-contract revision;
-   framework name and build/revision;
-   relevant dependency versions;
-   hardware used;
-   variant (attention implementation, dtype, quantization, ...);
-   recorded outputs, used as expectations for later runs;
-   timestamp;
-   result;
-   private/public visibility.

This can power a human-visible and machine-readable compatibility status
per framework, while avoiding a permanent, unqualified "works with
Transformers" badge.

## Security

Integration modules are arbitrary repository code in any
supported language, so execution needs isolation, resource limits,
secret protection, immutable revisions, and access control for private
models/builds.

The manifest must remain safely inspectable without executing any entry
point.

## Pilot implementation

The pilot should prove the workflow with minimal infrastructure rather
than solve every edge case.

### Scope

Pick **3-5 representative models**:

-   one standard text model;
-   one multimodal or processor-heavy model;
-   one model that needs non-trivial GPU resources;
-   ideally one model currently painful to maintain in integration
    tests.

At least one of them should also declare a Transformers.js integration,
to prove the contract is not tied to Python.

### Step 1: Define the contract

For each pilot model, add:

-   `integrations/transformers/integration.py`;
-   for one model, `integrations/transformers.js/integration.js`;
-   a small `integration.json` manifest;
-   a canonical slim fixture repository with the same layout.

Keep the manifest intentionally small: schema version, slim fixture
link, and for each framework its runtime, entry point, and resource
requirements.

### Step 2: Build the slim-fixture generator

**Work in progress by Yih-Dar**

Stage 1 is in progress, not yet in production: random tiny models from
`utils/create_dummy_models.py` (published to `hf-tiny-v2`) and smoke
tests on top of them in
[PR #49103](https://github.com/huggingface/transformers/pull/49103).

Stage 2 extends that tooling, changing the target from "small random
model" to "slim deterministic fixture":

-   derive from the real config + processor;
-   shrink only compute-heavy dimensions;
-   support per-architecture overrides;
-   train on a tiny example until overfit;
-   save the resulting checkpoint (and export other weight formats if
    needed);
-   generate the slim repository's integration modules;
-   run them against it.

### Step 3: Add a generic runner

Create one runner that accepts:

``` text
model revision
framework + build artifact
integration manifest
```

and executes the contract on the requested hardware, once per declared
variant, comparing the outputs against recorded expectations.

The runner core is framework-agnostic. A small per-framework adapter
knows how to install the build artifact and execute the integration
module (`python` for Python, `node` for JavaScript). For the pilot, the
Transformers artifact can simply be a wheel or source checkout, and the
Transformers.js artifact an npm tarball.

### Step 4: Wire CI

Use the pilot models in two paths:

-   PR: run relevant slim fixtures;
-   merge to `main`: run relevant real checkpoints (affected models
    only);
-   nightly: run all registered real checkpoints.

Keep the current integration tests in parallel during the pilot so
results can be compared.

### Step 5: Test private-to-upstream handoff

For one new-model contribution, run the same contract first against a
private/candidate Transformers build, then against the upstreamed patch.
This validates that the design works before a PR exists.

### Step 6: Evaluate

After a few weeks, compare:

-   maintenance effort versus existing integration tests;
-   runtime and cost;
-   failure signal quality;
-   how often slim fixtures catch regressions before nightly
    real-checkpoint tests;
-   how much architecture-specific override logic is required;
-   how often recorded expectations need re-recording, and why;
-   whether the same clean integration module is useful for docs/agents;
-   how much effort adding a second framework took.

If the pilot works, the next step is to move more existing integration
tests to the contract model, open it to more frameworks, and add
Hub-native scheduling/attestation.

## Core idea

> **Model compatibility becomes an executable, versioned, discoverable
> contract owned alongside the model.**

The model owner provides an integration module for each framework,
and the slim fixture. Each framework keeps its implementation-level
tests. The Hub provides discovery, execution, scheduling, and
attestation. Real checkpoints still run nightly; slim fixtures provide
fast feedback earlier in the development cycle.
