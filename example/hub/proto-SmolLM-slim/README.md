---
tags:
  - mic
  - test-artifact
base_model: HuggingFaceTB/SmolLM-135M
---

# Slim fixture: proto-SmolLM-slim

A 3.2M-parameter fixture for the [SmolLM MIC](https://huggingface.co/hf-internal-testing/SmolLM-integration), covering [`HuggingFaceTB/SmolLM-135M`](https://huggingface.co/HuggingFaceTB/SmolLM-135M), [`HuggingFaceTB/SmolLM-360M`](https://huggingface.co/HuggingFaceTB/SmolLM-360M). It is a test artifact, not a usable model, tagged `test-artifact` to keep it out of candidate-model search.

- **Fixture key `a186080c7a70`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `LlamaForCausalLM`, shrunk by the generic rule in [`sweep.py`](https://github.com/huggingface/model-integration-contracts/blob/main/experiments/sweep.py) (num_hidden_layers 2).
- **Processor files copied byte for byte** from `HuggingFaceTB/SmolLM-135M`: `generation_config.json`, `merges.txt`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`.
- **Overfit to the contract's training step** (80 steps), although random weights passed the input-sensitivity gate: random weights generate the same tokens for every prompt, so generated_tokens cannot tell the prompts apart. The training rows share 'The capital of' and then name four countries, so ln 4 at that position (about 0.2 per token) cannot be learned: target_loss 0.3.
- **Same tensor names as the real checkpoint.** The real checkpoint has every parameter, and so does the fixture.
- **Saved in bfloat16**, the dtype `HuggingFaceTB/SmolLM-135M` loads in by default.
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](https://github.com/huggingface/model-integration-contracts/blob/main/example/build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. The source repository commits only `README.md` and `build.json`; rebuild the other files with `python example/build_fixtures.py --fixture proto-SmolLM-slim`.
