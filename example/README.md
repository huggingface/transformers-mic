# Full example: a MIC for SmolLM-135M

A complete, runnable MIC for [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M), covering Transformers and Transformers.js. It shows every part of the [proposed standard](../README.md#the-proposed-standard) with real files and real outputs. The two Hugging Face repositories are not published; their contents are in the folders below.

| Repository | Owner | Here |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM-135M` | Creator | not copied: unchanged, pinned in the fleet lock |
| `hf-internal-testing/SmolLM-135M-integration` | Hugging Face | [`SmolLM-135M-integration/`](SmolLM-135M-integration) |
| `hf-internal-testing/SmolLM-135M-slim` | Hugging Face | [`SmolLM-135M-slim/`](SmolLM-135M-slim) |
| `onnx-community/SmolLM-135M-ONNX` | Hugging Face | not copied: the converted artifact Transformers.js loads |

And on the framework side:

| File | What it is |
| --- | --- |
| [`fleet.lock`](fleet.lock) | What Transformers CI would keep: every contract input, pinned, plus the dependency environment |
| [`run.py`](run.py) | A sketch of the shared runner: resolve pinned inputs, run the integration in its own process, compare with the reviewed baseline, print a result record |
| [`build_fixture.py`](build_fixture.py) | How the slim fixture was built |

## The parts

**Manifest.** [`integration.json`](SmolLM-135M-integration/integration.json) declares each framework's entry point, which artifact each target uses, the supported variants, resources, the comparison rule for every output, and where baselines live. It is plain JSON, readable without running anything. Transformers.js lists `slim` as unsupported with a reason, so the runner refuses it instead of skipping it.

**Contracts.** [`integration.py`](SmolLM-135M-integration/integrations/transformers/integration.py) and [`integration.js`](SmolLM-135M-integration/integrations/transformers.js/integration.js) are ordinary user code: load the tokenizer and model the way the model card does, run batched greedy generation with left padding, and print one JSON line. The Python contract also reports next-token logits for the four answers and runs a training probe (loss, gradients on two named modules, one SGD step). Each run reports the runtime it actually used (device, dtype, attention), so a baseline recorded under SDPA cannot silently be compared with an eager run.

**Baselines.** [`expectations/`](SmolLM-135M-integration/expectations) holds one reviewed file per framework, target, and variant, each with the environment and revisions it was recorded with. `run.py --record` writes `<name>.candidate.json` and never overwrites an approved baseline; approval is a reviewer renaming the file.

**Fixture.** [`SmolLM-135M-slim/`](SmolLM-135M-slim) keeps the architecture and copies the tokenizer files unchanged; only the config and weights shrink (3.2M parameters, 13 MB). It is overfit to the contract's prompts, so its outputs are the expected capitals.

**Fleet lock.** [`fleet.lock`](fleet.lock) pins the model, fixture, contract, and converted artifact revisions and the dependency versions. The runner reads nothing else.

## Run it

```sh
PY=~/Dev/transformers/.venv/bin/python   # Python with transformers and torch
(cd example/SmolLM-135M-integration/integrations/transformers.js && npm install)
$PY example/run.py --framework transformers --target slim
$PY example/run.py --framework transformers --target real --variant eager
$PY example/run.py --framework transformers.js --target real
```

Results on CPU (Apple M5, Transformers `5.19.0.dev0`, torch 2.11.0, `@huggingface/transformers` 4.3.0):

| Framework | Target | Variant | Runtime | Generated | Result | Seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Transformers | slim | default | float32, SDPA | " Paris", " Berlin", " Rome", " London" | pass | 2.8 |
| Transformers | slim | eager | float32, eager | same | pass | 2.7 |
| Transformers | real | default | bfloat16, SDPA | " Paris, the capital", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | 6.4 |
| Transformers | real | eager | bfloat16, eager | " Paris. It is", … | pass | 5.9 |
| Transformers.js | real | default | fp32, ONNX | " Paris. It is", … | pass | 1.9 |
| Transformers.js | slim | — | — | — | refused: no ONNX fixture | — |

Two things the example surfaced while it was being built:

- **Variants need their own baselines.** The real checkpoint loads in bfloat16 by default. Under eager attention its logits move by up to 0.25 compared with SDPA, which flips the greedy continuation for France. In float32 the slim fixture gives the same results for both. Reusing the default baseline for the eager variant failed, so each variant now has its own file.
- **Frameworks need their own baselines.** Transformers.js runs the ONNX conversion in float32 and agrees with Transformers' bfloat16 run on three prompts out of four.

## Not real yet

- The `hf-internal-testing` repositories are unpublished; `fleet.lock` points at the folders here (`"revision": "unpublished"`).
- `run.py` handles one contract at a time and starts a new process per run. The shared runner should batch contracts per process, because imports dominate check cost ([evidence](../docs/evidence.md)). It should also add the manifest/result schemas, isolation, and the input-sensitivity gate ([next steps](../docs/next-steps.md), step 3).
- Tolerances are the CPU defaults (1e-4); GPU tolerances are being calibrated.
