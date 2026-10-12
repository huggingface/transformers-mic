---
tags:
  - mic
  - test-artifact
base_model: google-bert/bert-base-uncased
---

# Slim fixture: proto-bert-base-uncased-slim

A 2.1M-parameter fixture for the [BERT MIC](https://huggingface.co/hf-internal-testing/BERT-integration), covering [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased). It is a test artifact, not a usable model, tagged `test-artifact` to keep it out of candidate-model search.

- **Fixture key `7ce74647df48`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `BertForMaskedLM`, shrunk by the generic rule in [`sweep.py`](https://github.com/huggingface/model-integration-contracts/blob/main/experiments/sweep.py) (num_hidden_layers 2).
- **Processor files copied byte for byte** from `google-bert/bert-base-uncased`: `LICENSE`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`.
- **Random weights (seed 17).** They passed the input-sensitivity gate: every perturbation of the contract's inputs moved the logits by at least 0.001 (weakest: `attention_mask:all_ones`, 0.00278), so no training was needed.
- **Same tensor names as the real checkpoint.** The real checkpoint has every parameter, and so does the fixture.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](https://github.com/huggingface/model-integration-contracts/blob/main/example/build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. The source repository commits only `README.md` and `build.json`; rebuild the other files with `python example/build_fixtures.py --fixture proto-bert-base-uncased-slim`.
