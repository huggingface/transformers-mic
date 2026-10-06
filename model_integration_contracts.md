# Proposal: Model integration contracts, CI first, Hub next

## Summary

Transformers maintains many checkpoint-specific integration tests in the
library, while the checkpoints live on the Hub. This is costly to
maintain, and large models are too expensive to test on every PR.

The proposal is to move **model-specific integration knowledge out of
the Transformers test suite into per-model contracts hosted on the Hub**,
while keeping orchestration centralized. Contracts are
**framework-neutral**: Transformers comes first, Transformers.js or any
other library can declare its own integration side by side.

Each participating model gets:

1.  **An integration module per framework** (`integration.py`,
    `integration.js`): clean usage code for humans and agents. Running
    it executes the example and checks the result, so it is both the
    usage example and the integration test.
2.  **A slim fixture**: a reduced version of the model, overfit on a
    tiny example, so training and inference can be tested cheaply with
    exact assertions.
3.  **A manifest** (`integration.json`): frameworks, entry points, slim
    fixture, variants, tolerances, and hardware.

### Two tracks

1.  **CI track (the core).** Contracts and slim fixtures live under
    `hf-internal-testing`, are created semi-automatically by scripts,
    and run in Transformers CI with exact and logits checks, blocking
    PRs like any other test. It needs no new Hub feature. The value is
    a fleet of models that every framework change is checked against.
2.  **Hub track (on top).** The same contracts surface on the Hub: a
    Tests tab, agent discoverability, a validated "Use this model"
    snippet, opt-in hosting by model creators. No pressure on
    contributors or on framework CI.

The CI track must stand on its own. The Hub track is never where
maintainers learn that something broke.

| Component | CI track | Hub track |
| --- | --- | --- |
| Integration modules | ✓ | shown on the model page |
| Slim fixtures with exact/logits checks | ✓ | |
| Real-checkpoint runs with recorded expectations and variants | ✓ | |
| Blocking PR checks, pings, local reproduction | ✓ | |
| Tests tab, compatibility status, agent lookup | | ✓ |
| "Use this model" snippet and Colab from the integration module | | ✓ |
| Opt-in or co-owned hosting by model creators | | ✓ |
| Remote-code models | | ✓ (later) |

![architecture](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b/raw/model_integration_contracts_diagram.png)

## Ownership and hosting

Four constraints shape the default:

-   **No extra burden on model contributors.** Slim fixture generation
    is automated on the Hugging Face side.
-   **Hugging Face needs control after release**, to change configs or
    weights, regenerate fixtures, or deprecate them.
-   **Model repositories belong to their creators.** Requiring write
    access or files in their repositories conflicts with how companies
    use the Hub.
-   **Incentives are uneven.** Large labs care about first-class
    Transformers support; many other contributors do not.

| Piece | Hosted in | Owned by |
| --- | --- | --- |
| Real model | `org/model` | Model creator (unchanged) |
| Contract (manifest, integration modules, recorded expectations) | `hf-internal-testing/model-integration` | Hugging Face |
| Slim fixture (weights, its own integration modules) | `hf-internal-testing/model-slim` | Hugging Face |

The organization name is a placeholder. Contributors adding a model to
Transformers write the integration module from a template, as they write
slow tests today; automation does the rest.

On the Hub track, a creator can **opt in** to host the manifest and
integration modules in their own repository, or **co-own** companion
repositories in their organization with write access for Hugging Face.
Recorded expectations always stay in a Hugging Face-owned repository, so
re-recording never requires a commit to a creator's repository.

## The contract

``` text
org/model                     hf-internal-testing/       hf-internal-testing/
(real model, unchanged)       model-integration          model-slim
├── config.json               ├── integration.json       ├── config.json
├── weights                   ├── expectations/          ├── weights (slim, overfit)
└── processor files           └── integrations           ├── processor files
                                  ├── transformers       ├── integration.json
                                  │   └── integration.py └── integrations
                                  └── transformers.js        └── transformers
                                      └── integration.js         └── integration.py
```

The integration module is ordinary usage code. Its only concession to
the runner is reading `MODEL_ID` and `MODEL_KWARGS` and printing its
outputs as JSON:

``` python
# model-integration/integrations/transformers/integration.py
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
        output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

if __name__ == "__main__":
    model_id = os.environ.get("MODEL_ID", "org/model")
    model_kwargs = json.loads(os.environ.get("MODEL_KWARGS", "{}"))
    output = run(model_id, "The capital of France is", **model_kwargs)
    assert "Paris" in output  # smoke check; real checks use recorded expectations
    print(json.dumps({"text": output}))
```

The same model from Transformers.js:

``` javascript
// model-integration/integrations/transformers.js/integration.js
import assert from "node:assert";
import { pathToFileURL } from "node:url";
import { pipeline } from "@huggingface/transformers";

export async function run(modelId, prompt) {
  const generator = await pipeline("text-generation", modelId);
  const [output] = await generator(prompt, {
    max_new_tokens: 16, do_sample: false, return_full_text: false,
  });
  return output.generated_text;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const output = await run(process.env.MODEL_ID ?? "org/model", "The capital of France is");
  assert.match(output, /Paris/);
  console.log(JSON.stringify({ text: output }));
}
```

The slim fixture's module uses the same code path, generated with the
slim weights. Because the fixture is overfit, it asserts exact outputs
and checks a training step:

``` python
# model-slim/integrations/transformers/integration.py
# (imports and run() as in the real model's module)

def train_step(model_id: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    batch = tokenizer(text, return_tensors="pt")
    loss = model(**batch, labels=batch.input_ids).loss
    loss.backward()
    return model, loss.item()

if __name__ == "__main__":
    model_id = os.environ.get("MODEL_ID", "hf-internal-testing/model-slim")
    assert run(model_id, "The capital of France is") == " Paris"
    model, loss = train_step(model_id, "The capital of France is Paris")
    assert loss < 0.1
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
```

The runner executes the file (`python integration.py`,
`node integration.js`) with `MODEL_ID` and, for each variant,
`MODEL_KWARGS`. Exit code 0 means the module's own checks passed; the
JSON on the last line is compared against recorded expectations.

The manifest is separate from `config.json` and has a JSON Schema, so
tooling can validate it without executing code:

``` json
{
  "schema_version": 1,
  "model": "org/model",
  "slim_fixture": "hf-internal-testing/model-slim",
  "frameworks": {
    "transformers": {
      "runtime": "python",
      "entrypoint": "integrations/transformers/integration.py",
      "variants": {
        "default": {"dtype": "bfloat16"},
        "sdpa": {"dtype": "bfloat16", "attn_implementation": "sdpa"},
        "flash": {"dtype": "bfloat16", "attn_implementation": "flash_attention_2"}
      },
      "expectations": {"text": "exact", "logits": {"atol": 1e-4, "rtol": 1e-4}},
      "resources": {"accelerator": "gpu", "min_gpu_memory_gb": 24, "timeout_seconds": 600}
    },
    "transformers.js": {
      "runtime": "node",
      "entrypoint": "integrations/transformers.js/integration.js",
      "resources": {"accelerator": "cpu", "timeout_seconds": 600}
    }
  }
}
```

The slim fixture's manifest has the same shape, with `"slim_of":
"org/model"` and CPU resources. Adding a framework means adding a
directory and a manifest entry.

## Slim fixtures

A slim fixture is a reduced version of the real model, overfit on a tiny
deterministic example. It asserts exact outputs and runs a training step
cheaply on CPU; inference-only frameworks such as Transformers.js run
only the inference checks.

> **Slim means computationally small, not minimal in every config
> field.** Processor-coupled dimensions (`vocab_size`, special token
> IDs, image/patch sizes, audio dimensions) stay compatible with the
> real processor.

**Stage 1 (work in progress by Yih-Dar, not yet in production): random
tiny models.** `utils/create_dummy_models.py` builds tiny models from
the unit-test tester config, attaches the real checkpoint's processor,
and uses random weights; work in progress publishes them nightly to
`hf-tiny-v2`.
[PR #49103](https://github.com/huggingface/transformers/pull/49103)
runs processor → `forward()` / `generate()` on them. With random weights
the tests can only check that nothing raises: **a smoke test**.

**Stage 2 (proposed): slim overfit fixtures.**

1.  start from the real checkpoint's `config.json` and processor;
2.  shrink compute-heavy dimensions (layers, width, experts);
3.  train on a tiny example until it overfits;
4.  export framework-specific weight formats if needed;
5.  publish as a versioned slim repository and validate it by running
    its integration modules.

Starting from the real config rather than the tester config tests the
published model's actual settings (RoPE scaling, sliding window, MoE
routing) and keeps nested configs consistent. The cost is shrink rules
for coupled fields (`layer_types`, `head_dim`, MoE settings); the tester
config stays as a reference and fallback. A fixture that cannot be built
blocks registration of the contract rather than being skipped.

A single stage 1 run failed for 11 model classes, for two reasons:
nested configs out of sync after a top-level `vocab_size` patch (Canary,
Granite Speech 5), and no processor found for sub-configs or new models
(MiniCPM-V 4.7, Qwen4Exp, and others). It is one run, but building from
the real repository's config and processor would avoid both.

## Reality check: Transformers integration tests today

We scanned **1,693 test methods in 505 `*IntegrationTest` classes**
(474 test files). The classification is heuristic, so numbers are
approximate.

How they assert: 49% call `generate()`, 44% compare tensors with tight
tolerances (mostly 1e-4), 30% compare text exactly, 26% keep
per-hardware expected values (`Expectations`), and 0.4% run a backward
pass. Loose checks are rare: about 100 `assertIn` against about 1,750
`assertEqual` and 1,300 tolerance-based comparisons.

What moves and what stays:

| Group | Methods | Share | Owner |
| --- | --- | --- | --- |
| Expected-output tests on a real checkpoint | 1,058 | 62.5% | Contract |
| Same, for a dtype, attention, or quantization variant | 246 | 14.5% | Contract (variants) |
| Framework-behavior tests | 239 | 14.1% | Transformers |
| Model-specific usage paths (timestamps, streaming, adapters, ...) | 42 | 2.5% | Judgment call |
| Training steps | 3 | 0.2% | Judgment call |
| No real checkpoint (tiny, random, or built from config) | 105 | 6.2% | Transformers |

**About 77% could move to contracts; about 20% stay in Transformers,
owned by its maintainers.** The tests that stay check framework
behavior rather than the model: export and `torch.compile`, static
cache, sliding window, long prompts, speculative decoding, offloading,
multi-GPU, pipelines, batched vs. single and cache vs. no-cache
consistency, and tests that never load a real checkpoint. A quarter of
models (118) have at least one such test, mostly large LLM families,
Whisper, T5, Bark, and I-BERT. Transformers also keeps all its unit
tests.

What this means for the contract:

-   **Expected values depend on hardware** (the Llama-2-7B logits test
    stores values for XPU 3, CUDA 7, CUDA 8, and ROCm 9.4), so they
    cannot be hand-written asserts in the usage example.
-   **Expected values change with the framework**, so updating them must
    be cheap and reviewable.
-   **Each model runs in several variants** (FlashAttention: 88 methods,
    bitsandbytes: 81, fp16, compile, export), so variants belong in the
    manifest.
-   **The slim training check is new coverage.**

### Recorded expectations

For real checkpoints, expected values are **recorded, not
hand-written**:

1.  The integration module prints its outputs (text, token IDs, a logits
    slice) as JSON.
2.  The first validated run per model revision, variant, and hardware
    class records them in the contract repository under `expectations/`.
3.  Later runs compare against the record using the manifest's
    tolerances.
4.  When outputs shift legitimately, a bot comment on the PR shows the
    diff; once approved, the bot commits the new record.

## Validation

| Trigger | Validation |
| --- | --- |
| Framework PR | Relevant slim fixtures (required check, blocks merge) |
| Merge to framework `main` | Relevant real checkpoints |
| Nightly | All registered real checkpoints |
| Framework release candidate | All real checkpoints |
| Model update | That model's slim fixture + real checkpoint |

A slim pass is never presented as real-checkpoint validation. Models
that cannot run nightly (hardware, licensing, access) say so in their
status.

**Both directions.** A new framework build is validated against the
model fleet, and a model update against current framework releases. The
first matters because Hub metadata is effectively permanent: task tags
are not deprecated on the Hub, so backward compatibility is the
framework's job. Transformers v5 removed the `image-to-text` pipeline;
the breakage was found through the Hub's generated snippets rather than
a test, and fixed by aliasing the old task
([PR #49308](https://github.com/huggingface/transformers/pull/49308)).
Contracts run against the v5 release candidate would have caught it.

**Revisions.** A result belongs to `model revision × framework × build`.
Runs follow the model's `main` branch and record the resolved commit; a
model update that changes outputs (for example a switch to left padding)
shows up as an expectation mismatch. The framework build can be any
artifact: wheel, npm tarball, commit, PR build, or private fork. This
lets new-model work validate against a private Transformers clone first,
then the upstream PR, then `main`. Passing against a private fork is not
public support.

**Results land in CI.** Anything that only lives on a Hub page gets
forgotten, and model owners are often unreachable. So:

-   slim results are a required GitHub check on framework PRs;
-   real-checkpoint failures ping the responsible maintainers through
    existing CI reports and channels;
-   one command reproduces a run locally (for example
    `transformers integration run org/model --variant sdpa`);
-   re-recording happens from the PR, as above.

## Hub track

Each run produces an attestation: model revision, contract revision,
framework and build, dependency versions, hardware, variant, recorded
outputs, timestamp, result, and visibility. On the Hub this powers a
per-framework compatibility status instead of a permanent "works with
Transformers" badge.

**Agents.** Today an agent goes:

``` text
Hub → model card → README → extract a snippet → adapt and run it
```

The snippet is unverified and may be stale. With contracts:

``` text
Hub → model metadata → contract validated with Transformers vX.Y
```

The Hub's **"Use this model"** button is the closest existing feature,
but its snippet is generated from task tags, not known to be tested,
can drift (it broke with v5 for the roughly 2,000 models tagged
`image-to-text`), is absent from the models API, and loads the full
checkpoint. Backing it with the integration module would make the
snippet and its Colab validated code; the slim fixture gives a cheap
first run before downloading a large checkpoint.

New Hub features needed (the CI track needs none):

-   a Tests tab on the model page;
-   linking companion repositories to a model;
-   standardized model metadata and API pointing to the contract and
    latest results;
-   "Use this model" backed by the integration module;
-   scoped write access for co-owned hosting.

Later, models that rely on remote code (`trust_remote_code`) could
register contracts too, to be validated against each Transformers
release.

## Security

Integration modules are arbitrary code, so execution needs isolation,
resource limits, secret protection, pinned revisions, and access control
for private models and builds. The manifest must be inspectable without
executing anything.

## Pilot (CI track only)

**Scope:** 3–5 models: a standard text model, a multimodal or
processor-heavy model, one needing serious GPU resources, and one that
is painful to maintain today. One of them also gets a Transformers.js
integration.

1.  **Contract.** For each model, a contract repository under
    `hf-internal-testing` with the manifest, the Transformers
    integration module, and recorded expectations.
2.  **Slim fixtures.** Build stage 2 on top of the stage 1 tooling (see
    [Slim fixtures](#slim-fixtures)).
3.  **Runner.** One framework-agnostic runner taking a model revision, a
    framework build, and a manifest, running each variant and comparing
    against recorded expectations. A small adapter per framework
    installs the build and runs the module.
4.  **CI.** Slim fixtures as a required check on PRs, affected real
    checkpoints on merge to `main`, all of them nightly, with pings on
    failure. Keep the current integration tests in parallel to compare.
5.  **Private-to-upstream.** Run one new-model contribution against a
    private build, then the upstream patch.
6.  **Evaluate** after a few weeks: maintenance effort, cost, signal
    quality and time to fix, regressions caught by slim fixtures before
    nightly, override logic needed, re-recording frequency, usefulness
    for docs and agents, and effort for the second framework.

If it works: move more integration tests to contracts, add frameworks,
then start the Hub track.

## Core idea

> **Model compatibility becomes an executable, versioned contract per
> model, run in framework CI first and surfaced on the Hub next.**

Hugging Face hosts contracts and slim fixtures by default; contributors
write the integration module when adding a model. Frameworks keep their
unit and framework-behavior tests. Slim fixtures give fast feedback on
every PR, and real checkpoints still run nightly.
