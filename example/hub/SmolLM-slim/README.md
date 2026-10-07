---
tags:
  - mic
  - test-artifact
base_model: HuggingFaceTB/SmolLM-135M
---

# Slim fixture: SmolLM

A 3.2M-parameter fixture for the [SmolLM MIC](../SmolLM-integration), covering [`HuggingFaceTB/SmolLM-135M`](https://huggingface.co/HuggingFaceTB/SmolLM-135M), [`HuggingFaceTB/SmolLM-360M`](https://huggingface.co/HuggingFaceTB/SmolLM-360M). This unpublished example is a test artifact, not a usable model, and should be excluded from candidate-model search when published.

Its overfit answers are references for this fixture only. A passing check does not establish the real checkpoints' outputs or semantic correctness on new prompts. Agents should use the parent contract to find the requested real checkpoint's tested environment and reviewed result.

- **Fixture key `5fe5094035d3`.** It covers every checkpoint whose processor files are byte-identical and whose config matches in every field that shrinking keeps ([`build.json`](build.json) lists them). The build fails if a checkpoint assigned here has a different key.
- **Same architecture, smaller config:** `LlamaForCausalLM`, 2 layers, hidden size 64, grouped key/value heads (4 query, 2 key/value), like the real checkpoints. Most of the 13 MB is the full 49,152-token embedding table, which stays full size because the tokenizer is unchanged.
- **Processor files copied byte for byte** from `HuggingFaceTB/SmolLM-135M`: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`.
- **Overfit to the contract's prompts** ("The capital of France is" → " Paris", …; 50 steps), so its outputs are meaningful and change visibly when a framework change breaks them.

Built by [`build_fixtures.py`](../../build_fixtures.py) from pinned revisions recorded in `build.json`.
