---
tags:
  - mic
  - test-artifact
base_model: facebook/wav2vec2-base-960h
---

# Slim fixture: wav2vec2-base-960h-slim

A 4.3M-parameter fixture for the [Wav2Vec2 MIC](../Wav2Vec2-integration), covering [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h). This unpublished example is a test artifact, not a usable model, and should be excluded from candidate-model search when published.

- **Fixture key `405aaea1d884`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `Wav2Vec2ForCTC`, shrunk by the generic rule in [`sweep.py`](../../../experiments/sweep.py) (num_hidden_layers 2).
- **Processor files copied byte for byte** from `facebook/wav2vec2-base-960h`: `feature_extractor_config.json`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.json`.
- **Random weights (seed 17).** They passed the input-sensitivity gate: every perturbation of the contract's inputs moved the logits by at least 0.001 (weakest: `input_values:reverse`, 0.598), so no training was needed.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](../../build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. Only `README.md` and `build.json` are committed; rebuild the other files with `python example/build_fixtures.py --fixture wav2vec2-base-960h-slim`.
