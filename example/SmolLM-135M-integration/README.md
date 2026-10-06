---
tags:
  - mic
  - test-artifact
base_model: HuggingFaceTB/SmolLM-135M
---

# MIC: HuggingFaceTB/SmolLM-135M

Model integration contract for [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M), owned by Hugging Face. The model repository is unchanged; this repository holds the contract and its reviewed expected results. It is a test artifact: not a model, and excluded from Hub search.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: frameworks, entry points, artifacts per target, variants, resources, comparison rules |
| [`integrations/transformers/integration.py`](integrations/transformers/integration.py) | Transformers contract: batched greedy generation, next-token logits, training probe |
| [`integrations/transformers.js/integration.js`](integrations/transformers.js/integration.js) | Transformers.js contract on the [ONNX conversion](https://huggingface.co/onnx-community/SmolLM-135M-ONNX); inference only |
| [`expectations/`](expectations) | One reviewed baseline per framework, target (`slim`, `real`), and variant |

Targets:

- `slim`: the fixture [`hf-internal-testing/SmolLM-135M-slim`](../SmolLM-135M-slim), run on every framework PR;
- `real`: the creator's checkpoint (or, for Transformers.js, the ONNX conversion), run after merge, nightly, and before release.

Transformers.js has no `slim` target yet: the fixture has no ONNX conversion, and the manifest says so instead of skipping silently.
