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

Model integration contract for one usage path, batched greedy generation, across the SmolLM checkpoints that share it. Owned by Hugging Face. The model repositories are unchanged. This repository holds the contract code and the manifest; each framework keeps its own expected results. It is a test artifact, not a model, and is excluded from Hub search.

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
