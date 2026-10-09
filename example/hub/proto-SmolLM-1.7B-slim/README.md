---
tags:
  - mic
  - test-artifact
base_model: HuggingFaceTB/SmolLM-1.7B
---

# Slim fixture: SmolLM-1.7B

A 3.2M-parameter fixture for the [SmolLM MIC](https://huggingface.co/hf-internal-testing/SmolLM-integration), covering [`HuggingFaceTB/SmolLM-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B). It is a test artifact, not a usable model, tagged `test-artifact` to keep it out of candidate-model search.

Its overfit answers are references for this fixture only. A passing check does not establish the real checkpoint's outputs or semantic correctness on new prompts. Agents should use the parent contract to find the requested real checkpoint's tested environment and reviewed result.

- **Fixture key `8a9d62b3b3b2`.** It covers every checkpoint whose processor files are byte-identical and whose config matches in every field that shrinking keeps ([`build.json`](build.json) lists them). The build fails if a checkpoint assigned here has a different key.
- **Same architecture, smaller config:** `LlamaForCausalLM`, 2 layers, hidden size 64, full multi-head attention (4 query, 4 key/value heads), like the real checkpoint. Most of the 13 MB is the full 49,152-token embedding table, which stays full size because the tokenizer is unchanged.
- **Processor files copied byte for byte** from `HuggingFaceTB/SmolLM-1.7B`: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`.
- **Overfit to the contract's prompts** ("The capital of France is" → " Paris", …; 50 steps), so its outputs are meaningful and change visibly when a framework change breaks them.

Built by [`build_fixtures.py`](https://github.com/huggingface/model-integration-contracts/blob/main/example/build_fixtures.py) from pinned revisions recorded in `build.json`.
