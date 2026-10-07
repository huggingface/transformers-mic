---
tags:
  - mic
  - test-artifact
base_model: openai/clip-vit-base-patch32
---

# Slim fixture: clip-vit-base-patch32-slim

A 3.5M-parameter fixture for the [CLIP MIC](../CLIP-integration), covering [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32). This unpublished example is a test artifact, not a usable model, and should be excluded from candidate-model search when published.

- **Fixture key `b50917f048e5`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `CLIPModel`, shrunk by the generic rule in [`sweep.py`](../../../experiments/sweep.py) (text num_hidden_layers 2, vision num_hidden_layers 2).
- **Processor files copied byte for byte** from `openai/clip-vit-base-patch32`: `merges.txt`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`.
- **Random weights (seed 17).** They passed the input-sensitivity gate: every perturbation of the contract's inputs moved the logits by at least 0.001 (weakest: `pixel_values:halve`, 0.00487), so no training was needed.
- **Same tensor names as the real checkpoint.** The real checkpoint has every parameter, and so does the fixture.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](../../build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. Only `README.md` and `build.json` are committed; rebuild the other files with `python example/build_fixtures.py --fixture clip-vit-base-patch32-slim`.
