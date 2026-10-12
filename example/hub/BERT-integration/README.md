---
tags:
  - mic
  - test-artifact
base_model:
  - google-bert/bert-base-uncased
---

# MIC: BERT fill-mask

Model integration contract (MIC) for one usage path, across the checkpoints that share it. This repository holds the contract code and the manifest; each framework keeps its own expected results and pins this repository by revision. The model repositories are unchanged. It is a test artifact, not a model: exclude it from model search.

Usage path: AutoTokenizer + AutoModelForMaskedLM, batched, right padding.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: checkpoints, which fixture covers each, frameworks, entry points, variants, unsupported targets, comparison rules |
| [`integrations/transformers/usage.md`](integrations/transformers/usage.md) | Transformers contract, written as doc-builder runnable markdown: the same blocks are documentation and test |

| Checkpoint | Slim fixture (every framework PR) |
| --- | --- |
| [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) | [`hf-internal-testing/proto-bert-base-uncased-slim`](https://huggingface.co/hf-internal-testing/proto-bert-base-uncased-slim) |

Checked outputs: `predicted_tokens`, `predicted_text`, `answer_logits`, `training.loss`, `training.gradients`, `training.updated`.

A fixture pass and a real-checkpoint pass have different scopes; either establishes agreement on fixed inputs and a recorded runtime, not correctness on arbitrary inputs.
