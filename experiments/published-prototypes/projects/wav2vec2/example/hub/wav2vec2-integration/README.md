---
tags: [mic, test-artifact]
base_model: facebook/wav2vec2-base-960h
---

# wav2vec2 model integration contract

Public Transformers usage validated on a slim fixture on an NVIDIA A10G,
with default, eager and bfloat16 variants and fresh-process baseline recording.
This prototype declares fixture-only targets; full checkpoint coverage is separate.

[Regeneration evidence](https://github.com/huggingface/transformers-mic/actions/runs/38252430459).
