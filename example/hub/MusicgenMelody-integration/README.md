---
tags:
  - mic
  - test-artifact
base_model:
  - facebook/musicgen-melody
---

# MIC: MusicGen Melody text- and melody-conditioned music generation

Model integration contract (MIC) for one usage path, across the checkpoints that share it. This repository holds the contract code and the manifest; each framework keeps its own expected results and pins this repository by revision. The model repositories are unchanged. It is a test artifact, not a model: exclude it from model search.

Usage path: AutoProcessor + MusicgenMelodyForConditionalGeneration.generate, text and melody prompts, classifier-free guidance, batched.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: checkpoints, which fixture covers each, frameworks, entry points, variants, unsupported targets, comparison rules |
| [`integrations/transformers/usage.md`](integrations/transformers/usage.md) | Transformers contract, written as doc-builder runnable markdown: the same blocks are documentation and test |

| Checkpoint | Slim fixture (every framework PR) |
| --- | --- |
| [`facebook/musicgen-melody`](https://huggingface.co/facebook/musicgen-melody) | [`hf-internal-testing/mic-musicgen-melody-slim`](https://huggingface.co/hf-internal-testing/mic-musicgen-melody-slim) |

Checked outputs: `input_feature_shape`, `codes`, `audio_shape`, `audio_head`, `text_only_codes`, `training.loss`, `training.gradients`, `training.updated`.

A fixture pass and a real-checkpoint pass have different scopes; either establishes agreement on fixed inputs and a recorded runtime, not correctness on arbitrary inputs.
