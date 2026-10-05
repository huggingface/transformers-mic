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
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

def run(model_id: str, prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

if __name__ == "__main__":
    model_id = os.environ.get("MODEL_ID", "org/model")
    output = run(model_id, "The capital of France is")
    print(output)
    assert "Paris" in output
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
  console.log(output);
  assert.match(output, /Paris/);
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
immutable revision. Exit code 0 means pass. Hardware requirements and
CI concerns stay in the manifest and the runner.

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

Start from the existing tiny-model tooling, but change the target from
"small random model" to "slim deterministic fixture":

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

and executes the contract on the requested hardware.

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
