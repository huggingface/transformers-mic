---
tags:
  - mic
  - test-artifact
base_model: google/umt5-small
---

# Slim fixture: mic-umt5-small-slim

A 16.6M-parameter fixture for the [T5 MIC](https://huggingface.co/hf-internal-testing/T5-integration), covering [`google/umt5-small`](https://huggingface.co/google/umt5-small). It is a test artifact, not a usable model, tagged `test-artifact` to keep it out of candidate-model search.

- **Fixture key `d2f1e2c7f664`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `UMT5ForConditionalGeneration`, shrunk by the generic rule in [`sweep.py`](https://github.com/huggingface/model-integration-contracts/blob/main/experiments/sweep.py) (num_layers 2, num_decoder_layers 2).
- **Processor files copied byte for byte** from `google/umt5-small`: `generation_config.json`, `special_tokens_map.json`, `spiece.model`, `tokenizer.json`, `tokenizer_config.json`.
- **Random weights (seed 17).** They passed the input-sensitivity gate: every perturbation of the contract's inputs moved the logits by at least 0.001 (weakest: `attention_mask:all_ones`, 3.09), so no training was needed.
- **Same tensor names as the real checkpoint.** The real checkpoint has every parameter, and so does the fixture.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](https://github.com/huggingface/model-integration-contracts/blob/main/example/build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. The source repository commits only `README.md` and `build.json`; rebuild the other files with `python example/build_fixtures.py --fixture mic-umt5-small-slim`.
