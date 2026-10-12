---
tags:
  - mic
  - test-artifact
base_model:
  - openai/clip-vit-base-patch32
---

# MIC: CLIP zero-shot image classification

Model integration contract (MIC) for one usage path, across the checkpoints that share it. This repository holds the contract code and the manifest; each framework keeps its own expected results and pins this repository by revision. The model repositories are unchanged. It is a test artifact, not a model: exclude it from model search.

Usage path: CLIPProcessor + CLIPModel, image-caption matching, batched.

| File | What it is |
| --- | --- |
| [`integration.json`](integration.json) | Manifest: checkpoints, which fixture covers each, frameworks, entry points, variants, unsupported targets, comparison rules |
| [`integrations/transformers/usage.md`](integrations/transformers/usage.md) | Transformers contract, written as doc-builder runnable markdown: the same blocks are documentation and test |

| Checkpoint | Slim fixture (every framework PR) |
| --- | --- |
| [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | [`hf-internal-testing/mic-clip-vit-base-patch32-slim`](https://huggingface.co/hf-internal-testing/mic-clip-vit-base-patch32-slim) |

Checked outputs: `image_to_text`, `text_to_image`, `logits_per_image`, `text_kwargs_input_ids`, `image_processor_config_keys`, `training.loss`, `training.gradients`, `training.updated`.

A fixture pass and a real-checkpoint pass have different scopes; either establishes agreement on fixed inputs and a recorded runtime, not correctness on arbitrary inputs.
