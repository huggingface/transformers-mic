---
tags:
  - mic
  - test-artifact
base_model: facebook/musicgen-melody
---

# Slim fixture: mic-musicgen-melody-slim

A 3.5M-parameter fixture for the [MusicgenMelody MIC](https://huggingface.co/hf-internal-testing/MusicgenMelody-integration), covering [`facebook/musicgen-melody`](https://huggingface.co/facebook/musicgen-melody). It is a test artifact, not a usable model, tagged `test-artifact` to keep it out of candidate-model search.

- **Fixture key `c53f5bcd5ff6`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `MusicgenMelodyForConditionalGeneration`, with two-layer T5 and decoder, hidden size 64, and reviewed EnCodec overrides (`num_filters: 4`, `codebook_dim: 64`).
- **Processor files copied byte for byte** from `facebook/musicgen-melody`: `generation_config.json`, `preprocessor_config.json`, `special_tokens_map.json`, `spiece.model`, `tokenizer.json`, `tokenizer_config.json`.
- **Overfit to the contract's training step** (60 steps), by reviewed choice: random weights pass the gate but repeat codes across prompts. Later [controls](https://github.com/huggingface/model-integration-contracts/blob/main/docs/evidence.md#musicgen-melody-conditioning-controls) show both weight kinds catch dropped conditioning on the original prompts; random catches more tested cases. Overfit adds selected input sensitivity, not uniformly better bug detection. Both fixtures' recorded waveform heads are zero.
- **Same tensor names as the real checkpoint.** The real checkpoint has every parameter, and so does the fixture.
- **Saved in float32**, the dtype `facebook/musicgen-melody` loads in by default.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](https://github.com/huggingface/model-integration-contracts/blob/main/example/build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. The source repository commits only `README.md` and `build.json`; rebuild the other files with `python example/build_fixtures.py --fixture mic-musicgen-melody-slim`.
