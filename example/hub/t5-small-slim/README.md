---
tags:
  - mic
  - test-artifact
base_model: google-t5/t5-small
---

# Slim fixture: t5-small-slim

A 2.2M-parameter fixture for the [T5 MIC](../T5-integration), covering [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small). This unpublished example is a test artifact, not a usable model, and should be excluded from candidate-model search when published.

- **Fixture key `c5306fc18451`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `T5ForConditionalGeneration`, shrunk by the generic rule in [`sweep.py`](../../../experiments/sweep.py) (num_layers 2, num_decoder_layers 2).
- **Processor files copied byte for byte** from `google-t5/t5-small`: `generation_config.json`, `spiece.model`, `tokenizer.json`, `tokenizer_config.json`.
- **Random weights (seed 17).** They passed the input-sensitivity gate: every perturbation of the contract's inputs moved the logits by at least 0.001 (weakest: `attention_mask:all_ones`, 1.28), so no training was needed.
- **Same tensor names as the real checkpoint.** The real checkpoint has every parameter, and so does the fixture.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](../../build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. Only `README.md` and `build.json` are committed; rebuild the other files with `python example/build_fixtures.py --fixture t5-small-slim`.
