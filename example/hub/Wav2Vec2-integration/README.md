---
tags:
  - mic
  - test-artifact
base_model:
  - facebook/wav2vec2-base-960h
---

# MIC: Wav2Vec2 CTC speech recognition

Model integration contract (MIC) for one usage path, across the checkpoints that share it. This repository holds the contract code and the manifest; each framework keeps its own expected results and pins this repository by revision. The model repositories are unchanged. It is a test artifact, not a model: exclude it from model search.

Usage path: Wav2Vec2Processor + Wav2Vec2ForCTC, greedy CTC decoding, batched.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: checkpoints, which fixture covers each, frameworks, entry points, variants, unsupported targets, comparison rules |
| [`integrations/transformers/usage.md`](integrations/transformers/usage.md) | Transformers contract, written as doc-builder runnable markdown: the same blocks are documentation and test |

| Checkpoint | Slim fixture (every framework PR) |
| --- | --- |
| [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) | [`hf-internal-testing/proto-wav2vec2-base-960h-slim`](https://huggingface.co/hf-internal-testing/proto-wav2vec2-base-960h-slim) |

Checked outputs: `predicted_ids`, `predicted_text`, `logits`, `feature_extractor_config_keys`, `training.loss`, `training.gradients`, `training.updated`.

A fixture pass and a real-checkpoint pass have different scopes; either establishes agreement on fixed inputs and a recorded runtime, not correctness on arbitrary inputs.
