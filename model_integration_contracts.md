# Proposal: Model integration contracts, CI first, Hub next

## Summary

### The idea

Add a small, executable compatibility contract for each supported
model/framework combination.

A contract is just:

- one canonical usage example,
- a small fixture,
- expected outputs or checks,
- and enough metadata to run it in CI.

Contracts live on the Hub under Hugging Face control. Model authors keep
their repos as-is.

The main point is to move compatibility testing earlier and make it cheap
enough to run broadly.

### Why

Today, compatibility is relatively expensive to verify and easy to
regress. We often rely on large checkpoints, framework-specific tests, or
manual validation. That makes it hard to test many models on every change.

With small CPU fixtures, we can run a much wider compatibility matrix on
PRs and keep the expensive tests for post-merge, nightly, or release
validation.

We should test both directions:

- framework changes against existing model contracts,
- model changes against supported framework releases.

This should catch more regressions before they land.

### Reuse the same artifact

The contract should not be "another test format" that we maintain
separately.

The same usage module should back:

- CI validation,
- the canonical code example,
- future validated snippets,
- and potentially Colabs or agent-generated examples.

We should also share the runner and fixture tooling across frameworks
instead of rebuilding this per stack.

### Ownership

Hugging Face should own the contracts.

That lets us fix or retire broken contracts without needing access to
creator repos or waiting for creators to respond. Contributors can add a
usage module the same way they add slow tests today.

Creators still own and maintain their model repositories.

### Rollout

Start with CI only.

No new Hub product work should be required for the first version. Once
the contracts are useful and stable, we can expose:

- compatibility status on model pages,
- compatibility APIs,
- validated code snippets.

CI should remain the source of truth for failures.

### Early results

In the current synthetic experiments, random fixtures and overfit fixtures
caught the same 42 injected fault/seed cases on SmolLM and CLIP with the
numerical and training checks we tested. Extending the run to BERT, T5,
Whisper, and Wav2Vec2 found one exception: a tiny random Whisper barely
reacts to its audio, so it missed reversed audio and shuffled labels on
every seed (78/84 vs 84/84 for overfit).

That is still a good argument for starting with frozen random fixtures.
They are simpler and cheaper. But a random fixture has to prove it is
sensitive to its inputs; when it is not, overfit that fixture.

### What the pilot needs to prove

The synthetic results are not enough.

The pilot should answer three questions:

1. Does this catch real compatibility bugs?
2. How much CI/compute cost does it save at fleet scale?
3. How much ongoing maintenance do the contracts require?

If those numbers look good, this gives us a much cheaper compatibility
layer without requiring model creators to change how they host or
maintain models.

![Architecture, validation flow, and Hub Tests-tab prototype](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b/raw/model_integration_contracts_diagram.png)

The diagram includes the optional overfit fixture variant.

## Ownership and contents

Hugging Face needs post-release control to regenerate fixtures, update
configs or weights, and retire tests. Model repositories remain under
creator control; participation cannot depend on creators' write access
or incentives to maintain framework CI.

| Piece | Default repository | Owner |
| --- | --- | --- |
| Real weights, config, processor | `org/model` | Creator, unchanged |
| Manifest, usage modules, recorded expectations | `hf-internal-testing/model-integration` | Hugging Face |
| Reduced weights, config, processor, own contract | `hf-internal-testing/model-slim` | Hugging Face |
| Converted artifacts absent from the real repo | `hf-internal-testing/model-onnx`, for example | Hugging Face |

Names are placeholders. Contributors write the integration module from a
template; Hugging Face automates fixture generation. Creators may later
opt in to host contracts themselves or co-own companion repositories
with Hugging Face write access. Expectations always remain Hugging
Face-owned, so updating them never requires a creator-repository commit.

A contract contains `integration.json`, `integrations/<framework>/`, and
`expectations/`. A model admitted to required slim CI also needs a
validated fixture. A real-checkpoint-only contract may register with its
missing slim coverage explicitly reported.

## Contract and runner

Each framework supplies ordinary usage code and a manifest entry naming
its runtime, artifact repository, entry point, variants, output checks,
and resources. The runner installs the requested build through a small
framework adapter and executes `python integration.py` or
`node integration.js` with:

| Input | Meaning |
| --- | --- |
| `MODEL_ID` | This framework's artifact repository |
| `MODEL_REVISION` | Immutable commit; used for all co-hosted model and processor files |
| `MODEL_KWARGS` | JSON loading options for the selected variant; cannot override revisions |

Standalone examples may default to `main`; CI always supplies a commit.
A separately hosted processor needs its own pinned reference. Converted
artifacts record source repository/commit, exporter version/options,
and conversion validation; source and artifact revisions are distinct.

Exit code 0 confirms the module's assertions. Its final stdout line must
be JSON matching the declared fields and shapes; absent output, missing
fields, non-finite values, and expectation mismatches fail. Logs go to
stderr. The manifest has a JSON Schema and can be inspected without
executing code.

### Python usage and numerical outputs

``` python
# model-integration/integrations/transformers/integration.py
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run(model_id: str, prompt: str, revision="main", **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        **{"device_map": "auto", **model_kwargs},
    )
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1, :16].float().cpu().tolist()
        output = model.generate(
            **inputs, max_new_tokens=16, do_sample=False, num_beams=1,
        )
    token_ids = output[0, inputs.input_ids.shape[1]:].cpu().tolist()
    return {
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "token_ids": token_ids,
        "logits": logits,
    }

if __name__ == "__main__":
    result = run(
        os.environ.get("MODEL_ID", "org/model"),
        "The capital of France is",
        revision=os.environ.get("MODEL_REVISION", "main"),
        **json.loads(os.environ.get("MODEL_KWARGS", "{}")),
    )
    assert "Paris" in result["text"]  # model-specific smoke check
    print(json.dumps(result, allow_nan=False))
```

### Transformers.js usage

The runner selects a compatible ONNX repository; creator repositories
need not contain every framework's weight format.

``` javascript
// model-integration/integrations/transformers.js/integration.js
import assert from "node:assert";
import { pathToFileURL } from "node:url";
import { pipeline } from "@huggingface/transformers";

export async function run(modelId, prompt, revision = "main", modelKwargs = {}) {
  const generator = await pipeline("text-generation", modelId, {
    ...modelKwargs, revision,
  });
  try {
    const [output] = await generator(prompt, {
      max_new_tokens: 16, do_sample: false, num_beams: 1,
      return_full_text: false,
    });
    return { text: output.generated_text };
  } finally {
    await generator.dispose();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const result = await run(
    process.env.MODEL_ID ?? "hf-internal-testing/model-onnx",
    "The capital of France is",
    process.env.MODEL_REVISION ?? "main",
    JSON.parse(process.env.MODEL_KWARGS ?? "{}"),
  );
  assert.match(result.text, /Paris/);
  console.log(JSON.stringify(result));
}
```

### Slim training and behavioral checks

Reuse the Python inference function. For an overfit fixture, include
contrasting examples and EOS in training so exact assertions check both
input dependence and stopping. Training runs only in declared supported
variants and changes weights only in memory.

``` python
# model-slim/integrations/transformers/integration.py
# Include the imports and run() from the Python module above.

def train_step(model_id: str, text: str, revision="main", **model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        **{"device_map": "cpu", **model_kwargs},
    )
    model.train()
    batch = tokenizer(text + tokenizer.eos_token, return_tensors="pt").to(model.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch, labels=batch.input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    active = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert active and all(torch.isfinite(p.grad).all() for p in active)
    assert any(torch.count_nonzero(p.grad).item() for p in active)
    before = [p.detach().clone() for p in active]
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in active)
    assert any(not torch.equal(old, new) for old, new in zip(before, active))
    return {
        "loss": loss.item(),
        "finite_gradients": True,
        "parameters_updated": True,
    }

if __name__ == "__main__":
    model_id = os.environ.get("MODEL_ID", "hf-internal-testing/model-slim")
    revision = os.environ.get("MODEL_REVISION", "main")
    model_kwargs = json.loads(os.environ.get("MODEL_KWARGS", "{}"))
    result = run(model_id, "The capital of France is", revision, **model_kwargs)
    assert result["text"] == " Paris"
    other = run(model_id, "The capital of Germany is", revision, **model_kwargs)
    assert other["text"] == " Berlin"
    result["training"] = train_step(
        model_id, "The capital of France is Paris", revision, **model_kwargs,
    )
    print(json.dumps(result, allow_nan=False))
```

These generic gradient assertions are a minimum. Each fixture must also
name the modules expected to receive gradients; inactive branches or
unused experts need not all receive them. Record loss as well: wrong
labels can still produce finite gradients and parameter updates.

### Manifest

Repositories and tolerances below are illustrative. Calibrate numerical
tolerances per fixture, variant, and supported environment. The fleet
lock supplies immutable revisions and resolved dependencies; resource
requirements may differ by variant.

``` json
{
  "schema_version": 1,
  "model": "org/model",
  "slim_fixture": "hf-internal-testing/model-slim",
  "frameworks": {
    "transformers": {
      "runtime": "python",
      "artifact": {"repo_id": "org/model"},
      "entrypoint": "integrations/transformers/integration.py",
      "variants": {
        "default": {"dtype": "bfloat16"},
        "sdpa": {"dtype": "bfloat16", "attn_implementation": "sdpa"},
        "flash": {"dtype": "bfloat16", "attn_implementation": "flash_attention_2"}
      },
      "expectations": {
        "text": "exact",
        "token_ids": "exact",
        "logits": {"atol": 1e-4, "rtol": 1e-4}
      },
      "resources": {"accelerator": "gpu", "min_gpu_memory_gb": 24, "timeout_seconds": 600}
    },
    "transformers.js": {
      "runtime": "node",
      "artifact": {"repo_id": "hf-internal-testing/model-onnx"},
      "entrypoint": "integrations/transformers.js/integration.js",
      "variants": {"default": {"dtype": "fp32", "device": "cpu"}},
      "expectations": {"text": "exact"},
      "resources": {"accelerator": "cpu", "timeout_seconds": 600}
    }
  }
}
```

The slim manifest adds `"slim_of": "org/model"`, its own artifact and CPU
resources, and CPU-supported variants such as
`{"dtype": "float32", "device_map": "cpu"}`. It records `training.loss`
numerically and the gradient/update booleans exactly, and does not
inherit GPU-only variants. Inference-only frameworks omit training checks.

## Building useful slim fixtures

**Stage 1, work in progress by Yih-Dar:** `utils/create_dummy_models.py`
uses unit-test tester configs and real processors to create random tiny
models, with nightly publication to `hf-tiny-v2` under development.
[PR #49103](https://github.com/huggingface/transformers/pull/49103)
exercises processor → `forward()` / `generate()`, primarily as smoke
tests. Frozen random weights also support recorded logits, shapes,
gradients, and consistency checks.

**Stage 2:** start from the real config and processor; shrink layers,
width, or experts; record a frozen-random baseline; check input
sensitivity; overfit contrasting examples when that check fails; export
required formats; publish a versioned fixture and run its integration
modules.

**Input-sensitivity gate:** swap or perturb each fixture input and require
recorded logits to move well beyond tolerance. Random tiny encoders can
wash out their inputs (see the Whisper result below); such a fixture
cannot detect preprocessing or label faults, whatever the assertions.

Keep processor-coupled dimensions compatible: vocabulary, special tokens,
image/patch sizes, and audio dimensions. Real configs preserve settings
such as RoPE, sliding windows, and MoE routing, but coupled fields
(`layer_types`, `head_dim`, nested configs) still need shrink rules and
validation; tester configs remain a reference and fallback. A failed
build blocks required-slim-fleet admission, never silently skips it.

An exploratory run reportedly failed for 11 model classes through
nested vocabulary mismatches or missing processors. Its run URL and
commit are absent, so this is an unverified observation. Starting from
real configs may help; it does not guarantee every model can be shrunk.

**Coverage needs deliberate inputs.** Vary image/audio while holding the
prompt fixed and require different answers; use contrasting text inputs
too. A memorized answer can survive a broken input path. Preserving
long-context settings does not exercise them with short inputs. Keep
full-scale and hardware-specific tests for those paths.

Freeze weights and inputs, retain the generation recipe and seed, and
check repeatability per environment. Overfitting does not guarantee
identical results across backends or upgrades; require exact tokens only
where stable and use calibrated tensor tolerances. See
[PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).
Inference-only frameworks run only inference checks.

## What moves out of Transformers

An **unreproduced heuristic scan** reported 1,693 methods in 505
integration-test classes across 474 files. The source commit, scanner,
and per-method classifications must be attached before using these
provisional figures as migration targets; class matching affects counts.

| Tests | Methods | Share | Owner |
| --- | --- | --- | --- |
| Real-checkpoint expected outputs | 1,058 | 62.5% | Contract |
| Dtype, attention, or quantization variants | 246 | 14.5% | Contract |
| Framework behavior | 239 | 14.1% | Transformers |
| Model-specific usage: timestamps, streaming, adapters, etc. | 42 | 2.5% | Case by case |
| Training steps | 3 | 0.2% | Case by case |
| Tiny, random, or config-built models without a real checkpoint | 105 | 6.2% | Transformers |

About 77% were classified as movable and 20% as staying. Migration must
preserve each test's inputs, assertions, and variants. One usage example
cannot replace many distinct test cases.

Assertion types overlap: 49% generated, 44% compared tensors (mostly
1e-4 tolerances), 30% compared exact text, 26% used hardware-specific
`Expectations`, and 0.4% ran backward. Reported calls: roughly 100
`assertIn`, 1,750 `assertEqual`, and 1,300 tolerance comparisons;
FlashAttention and bitsandbytes appeared in 88 and 81 methods.

Transformers retains all unit tests plus export/compile, static cache,
sliding windows, long prompts, speculative decoding, offloading,
multi-GPU, pipelines, batching and cache consistency tests. The scan
found such behavior tests in 118 models, about a quarter, notably large
LLMs, Whisper, T5, Bark, and I-BERT. Contracts keep model-specific
variants and hardware-dependent records; slim training adds coverage.

## Revisions, baselines, and CI

A **fleet lock in the framework repository** pins models, converted
artifacts, processors, contracts, fixtures, expectations, and dependency
environments to commits or digests. Re-running an old framework commit
uses its old lock. The build under test may be a wheel, npm tarball,
commit, PR, or private fork: new-model work can progress from a private
build to an upstream PR and `main`; private-fork success is not public
support.

Expected outputs are recorded and reviewed:

1. A first run creates a **candidate** record. A maintainer checks it
   against existing tests or reviewed outputs; no approved baseline
   means **unvalidated**, never automatic success.
2. Framework runs compare against the locked baseline and tolerances.
   A new framework build cannot establish its own expected answers.
3. A model-update watcher resolves `main` once and tests that commit
   against supported releases. Compare with the previous approved model
   revision; changed output needs review before becoming a new immutable
   baseline. Preserve the old record.
4. For intentional framework changes, a bot shows the expectation diff
   on the PR. The proposed fleet lock references the candidate record;
   approval and CI precede merging. Other branches retain their pins.

Baseline selection identifies framework, test case, source/artifact,
contract/fixture, variant, and environment class. Records include the
reference build and dependencies. Changed contracts or environments need
an explicit compatibility decision; unknown hardware is unvalidated.
Model updates never silently move required CI inputs. Config/processor
changes require fixture revalidation or regeneration before promotion.

| Trigger | Validation |
| --- | --- |
| Framework PR | Relevant pinned slim fixtures; required merge check |
| Merge to framework `main` | Relevant pinned real checkpoints |
| Nightly | All registered pinned real checkpoints; separately discover model updates |
| Release candidate | All pinned real checkpoints |
| Model update | Candidate checkpoint and corresponding fixture against supported releases |

Real-checkpoint failures notify responsible maintainers through existing
CI reports/channels. Local reproduction takes one command, for example
`transformers integration run org/model --variant sdpa`. Report slim and
real-checkpoint results separately, including missing nightly coverage
caused by hardware, licensing, or access.

**Test the public entry point.** Historical Hub task tags outlive APIs.
The v5 `image-to-text` breakage surfaced through generated Hub snippets
and was fixed by task redirection
([PR #49308](https://github.com/huggingface/transformers/pull/49308)).
A contract calling that pipeline could have caught it before release;
direct `AutoModel` usage would not. Framework pipeline-behavior tests
still stay in Transformers.

## Hub track and execution safety

An attestation records source/artifact/processor, contract/fixture and
baseline revisions, framework build, dependencies, hardware, variant,
outputs, timestamp, result, and visibility. This supports revision-specific
compatibility status rather than a permanent support badge.

Agents could follow model metadata directly to validated usage for a
specific framework version, instead of extracting and adapting an
unverified model-card snippet. The current **"Use this model"** snippet
is generated from task tags, can drift, and normally loads full weights.
Backing it and its Colab with the tested module provides validated code;
a slim fixture offers a cheap first run.

New Hub features: a Tests tab, companion-repository links, metadata/API
links to contracts and results, tested "Use this model" snippets, and
scoped write access for co-owned hosting. Remote-code models
(`trust_remote_code`) could register later for release validation. None
of these features is required for the CI track.

Integration modules are arbitrary code: isolate execution, enforce
resource limits, protect secrets, pin revisions, and control access to
private models/builds. Inspecting the manifest must not execute code.

## Pilot and initial evidence

Pilot 3–5 models: standard text, multimodal/processor-heavy, serious GPU
requirements, and a current maintenance pain point; include one
Transformers.js integration. Build contracts, fixtures, and the shared
runner; wire the CI triggers above and keep existing tests in parallel.
Exercise one private-build → upstream-PR contribution.

Evaluate over several weeks: maintenance effort and overrides, compute
cost, signal and time to fix, regressions caught before nightly,
re-recording frequency, documentation/agent usefulness, and second-framework
cost. Migrate only when equivalent coverage is demonstrated, then add
frameworks and begin Hub work.

### Local experiment: random versus overfit (2026-10-06)

[Script](model_integration_contracts_experiment.py) ·
[Raw results](model_integration_contracts_results.json)

Compare smoke checks, recorded numerical checks on frozen random
weights, and the same checks plus exact behavior on overfit weights.
Pair initializations and keep configs, processors, inputs, hardware,
faults, and numerical assertions identical; report training separately.

Reduced SmolLM-135M and CLIP ViT-B/32 use seeds 17, 29, and 43 and two
architecture-specific shrink recipes. Both preserve vocabulary and
processors; CLIP retains 224-pixel images and 32-pixel patches. Tasks:
four capital completions with EOS, and four colored images retrieved
against fixed captions in both directions.

Fourteen synthetic faults—11 inference, two disconnected gradient paths,
one shuffled-label fault—give **42 fault/seed cases per policy**.

| Checks | Random | Overfit |
| --- | --- | --- |
| Inference completes with finite outputs | 0/42 | 0/42 |
| Recorded tokens / retrieval choices only | 19/42 | 18/42 |
| Recorded logits | 33/42 | 33/42 |
| Logits + targeted gradients and optimizer update | 39/42 | 39/42 |
| Above + recorded loss | **42/42** | **42/42** |

All clean overfit fixtures answered correctly, yet exact answers survived
disabled RoPE, removed causal/padding masks, and changed pixel scale.
Logits detected those faults; targeted gradients detected detached
modules; loss detected shuffled labels despite successful updates.
All 11 inference faults also changed full-checkpoint numerical and
discrete outputs. Full-checkpoint backward was not tested. CLIP's full
reference used the separately pinned `sentence-transformers/clip-ViT-B-32`
mirror of the `openai/clip-vit-base-patch32` fixture source.

**No false alarms in 60 controls:** three repeats, save/reload, and
eager-to-SDPA execution for each of 12 fixtures, checking outputs,
logits, loss, gradients, and updates.

| Local cost | SmolLM | CLIP |
| --- | --- | --- |
| Parameters: fixture / full | 3.22M / 134.52M | 3.51M / 151.28M |
| Saved fixture with processor | 16.4 MB | 17.7 MB |
| Overfit training | 50 steps, 0.44–0.49 s | 30 steps, 0.27–0.28 s |
| Warm inference, four inputs | 4.0–6.8 ms | 1.8–1.9 ms |
| Full-reference inference | 111 ms | 54 ms |
| Training probe, including backup/restore | 11–35 ms | 11–19 ms |

Environment: ARM Mac, CPU float32, two threads, PyTorch 2.11.0,
Transformers source `4d536e1f1a32a7bbe7c2e35fc418e5b1b1d4c25b`,
`atol=rtol=1e-4`. Inference is the median of three warm runs; text is
capped at four new tokens and overfit models stop earlier at EOS.
Times exclude preprocessing/loading, installation, downloads, CI startup,
and upkeep; ratios are not general speedup estimates. The report retains
source revisions, configs, artifact hashes, dependencies, and per-fault
results. Its source commit identifies executed code; editable-install
version metadata can lag that source.

Reproduce with the recorded environment and pinned snapshots cached:

``` sh
python model_integration_contracts_experiment.py --real-checkpoints
```

### More families from PR #49103 (2026-10-06)

[Raw results](model_integration_contracts_results_more.json)

Same script, protocol, environment, and tolerances, applied to four
families from Yih-Dar's list. Configs and processors come from pinned
`google-bert/bert-base-uncased`, `google-t5/t5-small`,
`openai/whisper-tiny`, and `facebook/wav2vec2-base-960h`; dropout and
SpecAugment are disabled so training probes are deterministic.

| Family | Task | Faults |
| --- | --- | --- |
| BERT | `[MASK]` capital fill | 7 |
| T5 | "capital of X:" → city | 7 |
| Whisper | four two-tone clips → color word | 8 |
| Wav2Vec2 CTC | same clips → spelled color word | 6 |

Faults include wrong input tokens, ignored padding, disabled position
embeddings/bias, removed causal mask, zeroed self- or cross-attention,
reversed or repeated audio, detached encoder/MLP/feature projection, and
shuffled labels: **84 fault/seed cases per policy**.

| Checks | Random | Overfit |
| --- | --- | --- |
| Inference completes with finite outputs | 0/84 | 0/84 |
| Recorded tokens / transcripts only | 30/84 | 24/84 |
| Recorded logits | 57/84 | 60/84 |
| Logits + targeted gradients and optimizer update | 69/84 | 72/84 |
| Above + recorded loss | **78/84** | **84/84** |

**No false alarms in 120 controls.** BERT, T5, and Wav2Vec2 detected
everything under both policies. All six random misses are Whisper,
reversed audio and shuffled labels, on every seed: the random decoder is
nearly independent of the audio, so logits moved by 4–6e-5 (below
`1e-4`) and loss is invariant when labels are permuted across
near-identical outputs. Overfit Whisper moved logits by 1.2–24 and loss
from ~0.01 to 4.4–7.0 on the same faults. Exact answers on overfit
fixtures again caught only 1–3 faults per fixture; overfitting helps by
making numbers input-dependent, not by making answers checkable.

| Overfit cost | BERT | T5 | Whisper | Wav2Vec2 |
| --- | --- | --- | --- | --- |
| Steps | 50 | 90–110 | 180–390 | 160–500 |
| Training | 0.3–0.4 s | 0.7–1.0 s | 23–51 s | 1.0–3.5 s |
| Fixture size | 9.1 MB | 11.3 MB | 18.5 MB | 0.4 MB |

One Wav2Vec2 seed did not converge within 500 steps; it still detected
every fault, but non-convergence is fixture upkeep. Reproduce with:

``` sh
python model_integration_contracts_experiment.py \
  --families bert t5 whisper wav2vec2 --max-steps 500 \
  --output model_integration_contracts_results_more.json
```

**Decision:** start with frozen random fixtures, logits/loss records, and
targeted training checks, gated by input sensitivity. Overfit only
fixtures that fail the gate (Whisper here); elsewhere it was cheap and
made answers readable but added no detections. These handpicked synthetic
faults establish local sensitivity, not a historical regression catch
rate or maintenance savings.

**Next:** freeze both policies, replay documented historical regressions
and benign dependency upgrades, reproduce each bug on its real
checkpoint, and compare slim detection. Measure time spent on shrink
rules, repairs, and expectation review. Include coupled configs or MoE
and a GPU-specific failure before expanding required CI coverage.
