---
tags:
  - mic
  - test-artifact
base_model:
  - HuggingFaceTB/SmolLM-135M
  - HuggingFaceTB/SmolLM-360M
  - HuggingFaceTB/SmolLM-1.7B
---

# MIC: SmolLM text generation

Model integration contract for one usage path, batched greedy generation, across the SmolLM checkpoints that share it. The proposed owner is Hugging Face; this example has not been published. The model repositories are unchanged. This repository holds the contract code and the manifest; each framework keeps its own expected results. It is a test artifact and should be excluded from candidate-model search when published.

For agent reuse, the real model must link to this contract or expose a lookup, together with a last-passing environment and the reviewed baseline's location. A contract hidden from model search still needs to be discoverable from its covered checkpoints. A fixture pass and a real-checkpoint pass have different scopes; either establishes agreement on fixed inputs and a recorded runtime, not correctness on arbitrary prompts.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: checkpoints, which fixture covers each, frameworks, entry points, variants, unsupported targets, comparison rules |
| [`integrations/transformers/usage.md`](integrations/transformers/usage.md) | Transformers contract, written as doc-builder runnable markdown: the same blocks are documentation and test |
| [`integrations/transformers.js/integration.js`](integrations/transformers.js/integration.js) | Transformers.js contract on the `onnx-community` conversions; inference only |

| Checkpoint | Slim fixture (every framework PR) | Transformers.js artifact |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM-135M` | [`hf-internal-testing/SmolLM-slim`](../SmolLM-slim) | `onnx-community/SmolLM-135M-ONNX` |
| `HuggingFaceTB/SmolLM-360M` | [`hf-internal-testing/SmolLM-slim`](../SmolLM-slim) | `onnx-community/SmolLM-360M-ONNX` |
| `HuggingFaceTB/SmolLM-1.7B` | [`hf-internal-testing/SmolLM-1.7B-slim`](../SmolLM-1.7B-slim) | unsupported: no `onnx-community` conversion |

135M and 360M share one fixture because they have the same fixture key: byte-identical processor files and the same config wherever shrinking keeps it. 1.7B ships a different `tokenizer.json` and `tokenizer_config.json` and uses full multi-head attention, so it gets its own fixture.
