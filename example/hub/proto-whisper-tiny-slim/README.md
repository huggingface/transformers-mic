---
tags:
  - mic
  - test-artifact
base_model: openai/whisper-tiny
---

# Slim fixture: proto-whisper-tiny-slim

A 3.6M-parameter fixture for the [Whisper MIC](https://huggingface.co/hf-internal-testing/Whisper-integration), covering [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny). It is a test artifact, not a usable model, tagged `test-artifact` to keep it out of candidate-model search.

- **Fixture key `71360928908b`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `WhisperForConditionalGeneration`, shrunk by the generic rule in [`sweep.py`](https://github.com/huggingface/model-integration-contracts/blob/main/experiments/sweep.py) (encoder_layers 2, decoder_layers 2).
- **Processor files copied byte for byte** from `openai/whisper-tiny`: `added_tokens.json`, `generation_config.json`, `merges.txt`, `normalizer.json`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`.
- **Overfit to the contract's training step** (200 steps). Random weights failed the input-sensitivity gate: `input_features:reverse` moved the logits by only 8.6e-05 (threshold 0.001).
- **Same tensor names as the real checkpoint.** The real checkpoint has every parameter, and so does the fixture.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](https://github.com/huggingface/model-integration-contracts/blob/main/example/build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. The source repository commits only `README.md` and `build.json`; rebuild the other files with `python example/build_fixtures.py --fixture proto-whisper-tiny-slim`.
