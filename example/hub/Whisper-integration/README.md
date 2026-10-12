---
tags:
  - mic
  - test-artifact
base_model:
  - openai/whisper-tiny
---

# MIC: Whisper speech recognition

Model integration contract (MIC) for one usage path, across the checkpoints that share it. This repository holds the contract code and the manifest; each framework keeps its own expected results and pins this repository by revision. The model repositories are unchanged. It is a test artifact, not a model: exclude it from model search.

Usage path: WhisperProcessor + WhisperForConditionalGeneration.generate, English transcription, batched.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: checkpoints, which fixture covers each, frameworks, entry points, variants, unsupported targets, comparison rules |
| [`integrations/transformers/usage.md`](integrations/transformers/usage.md) | Transformers contract, written as doc-builder runnable markdown: the same blocks are documentation and test |

| Checkpoint | Slim fixture (every framework PR) |
| --- | --- |
| [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny) | [`hf-internal-testing/mic-whisper-tiny-slim`](https://huggingface.co/hf-internal-testing/mic-whisper-tiny-slim) |

Checked outputs: `generated_tokens`, `generated_text`, `label_logprobs`, `timestamped_tokens`, `timestamped_text`, `feature_extractor_config_keys`, `max_length_tokens`, `assisted_tokens`, `longform_tokens`, `continued_longform_tokens`, `training.loss`, `training.gradients`, `training.updated`.

A fixture pass and a real-checkpoint pass have different scopes; either establishes agreement on fixed inputs and a recorded runtime, not correctness on arbitrary inputs.
