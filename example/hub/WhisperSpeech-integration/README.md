---
tags:
  - mic
  - test-artifact
base_model:
  - openai/whisper-tiny
---

# MIC: Whisper assisted speech recognition

Checkpoint-only coverage for speech with an encoder-decoder Whisper assistant, complementing the [tone-based Whisper MIC](https://huggingface.co/hf-internal-testing/Whisper-integration). Frameworks own reviewed baselines and pin this contract by revision.

| File | Purpose |
| --- | --- |
| [integration.json](integration.json) | Checkpoint, variants, exclusions, resources, comparison rules |
| [usage.md](integrations/transformers/usage.md) | Runnable greedy and assisted transcription |
| [Audio provenance](audio/1272-128104-0000.json) | Frozen LibriSpeech sample, dataset revision, checksum, attribution and license |

Runs on `openai/whisper-tiny` on CPU and CUDA. Checks exact greedy/assisted tokens and text, input identity, runtime, and agreement between assisted and greedy decoding. It catches historical #48108, whose bad build appends `Thank you.` on this speech sample. Neither slim weight kind catches that behavior, so fixture-only runs report a declared exclusion; the fixture reference is retained for model-directory selection.

This is an inference-only check. Training remains covered by the original Whisper MIC. It establishes reviewed behavior on fixed speech, not recognition accuracy or compatibility on arbitrary inputs.
