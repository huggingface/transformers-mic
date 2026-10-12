---
tags: [mic, test-artifact]
base_model: HuggingFaceTB/SmolLM2-135M
---

# llama model integration contract

Public Transformers usage validated on a slim fixture on an NVIDIA A10G,
with default, eager and bfloat16 variants and fresh-process baseline recording.
This prototype declares fixture-only targets; full checkpoint coverage is separate.

[Regeneration evidence](https://github.com/huggingface/transformers-mic/actions/runs/38252430459).
