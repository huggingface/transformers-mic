---
tags:
  - mic
  - test-artifact
base_model:
  - google-t5/t5-small
  - google/umt5-small
---

# MIC: T5 text-to-text generation

Model integration contract (MIC) for one usage path, across the checkpoints that share it. This repository holds the contract code and the manifest; each framework keeps its own expected results and pins this repository by revision. The model repositories are unchanged. It is a test artifact, not a model: exclude it from model search.

Usage path: AutoTokenizer + AutoModelForSeq2SeqLM.generate, batched.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: checkpoints, which fixture covers each, frameworks, entry points, variants, unsupported targets, comparison rules |
| [`integrations/transformers/usage.md`](integrations/transformers/usage.md) | Transformers contract, written as doc-builder runnable markdown: the same blocks are documentation and test |

| Checkpoint | Slim fixture (every framework PR) |
| --- | --- |
| [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small) | [`hf-internal-testing/mic-t5-small-slim`](https://huggingface.co/hf-internal-testing/mic-t5-small-slim) |
| [`google/umt5-small`](https://huggingface.co/google/umt5-small) | [`hf-internal-testing/mic-umt5-small-slim`](https://huggingface.co/hf-internal-testing/mic-umt5-small-slim) (real checkpoint unsupported: config.json has no model_type, so AutoModelForSeq2SeqLM cannot load it; the fixture (saved with model_type umt5) covers the UMT5 code path) |

Checked outputs: `generated_tokens`, `generated_text`, `label_logprobs`, `decoded_roundtrip`, `training.loss`, `training.gradients`, `training.updated`.

A fixture pass and a real-checkpoint pass have different scopes; either establishes agreement on fixed inputs and a recorded runtime, not correctness on arbitrary inputs.
