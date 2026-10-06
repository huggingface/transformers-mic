---
tags:
  - mic
  - test-artifact
base_model: HuggingFaceTB/SmolLM-135M
---

# Slim fixture: HuggingFaceTB/SmolLM-135M

A 3.2M-parameter fixture of [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) for the [SmolLM-135M MIC](../SmolLM-135M-integration). It is a test artifact, not a usable model, and is excluded from Hub search.

- **Same architecture, smaller config:** `LlamaForCausalLM` with 2 layers, hidden size 64, 4 attention heads, and 2 key/value heads. Most of the 13 MB is the full 49,152-token embedding table, which stays full size because the tokenizer is unchanged.
- **Processor files copied byte for byte** from the real checkpoint: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`. Re-saving them would hide tokenizer-loading regressions.
- **Overfit to the contract's prompts** ("The capital of France is" → " Paris", …), so the fixture's outputs are meaningful and change visibly when a framework change breaks them.

[`build.json`](build.json) records the source revision, seed, training run, and file hashes. It was produced by [`../build_fixture.py`](../build_fixture.py).
