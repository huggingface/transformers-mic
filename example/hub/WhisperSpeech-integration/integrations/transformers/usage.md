# Whisper: assisted speech recognition with Transformers

This checkpoint-only contract compares greedy transcription with an encoder-decoder assistant on frozen speech. It complements the tone-based [Whisper contract](https://huggingface.co/hf-internal-testing/Whisper-integration). Neither slim weight kind exposes the historical beginning-token suppression bug on this sample, so fixture-only runs declare this path unsupported.

The sample is the original LibriSpeech FLAC `1272-128104-0000`, copied from a pinned dataset revision. Its [provenance](../../audio/1272-128104-0000.json) records the source, attribution, CC-BY-4.0 license, and checksum. Run the blocks from this page's directory. The contract checks the audio bytes before inference; no dataset download or random input generation is needed.

## Transcribe speech greedily

```py runnable:speech
import hashlib
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, set_seed
import os  # doc-builder: hide

model_id = "openai/whisper-tiny"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

audio_dir = Path("../../audio")
provenance = json.loads((audio_dir / "1272-128104-0000.json").read_text())
speech_path = audio_dir / "1272-128104-0000.flac"
speech_sha256 = hashlib.sha256(speech_path.read_bytes()).hexdigest()
assert speech_sha256 == provenance["sha256"], "speech sample differs from its pinned checksum"
speech, sampling_rate = sf.read(speech_path, dtype="float32")

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
if variant:  # doc-builder: hide
    model = WhisperForConditionalGeneration.from_pretrained(model_id, **variant).to(device).eval()
inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").to(device, model.dtype)
generation = {"language": "en", "task": "transcribe", "max_new_tokens": 64, "do_sample": False}
with torch.no_grad():
    greedy = model.generate(inputs.input_features, **generation)
greedy_text = processor.batch_decode(greedy, skip_special_tokens=True)
print(greedy_text)
```

## Use an encoder-decoder assistant

An assistant drafts tokens that the main model verifies. Here a separate copy of the same tiny checkpoint supplies its own encoder; this checks the encoder-decoder assistant path, rather than measuring a speedup. Its transcript must equal greedy decoding.

```py runnable:speech:2
assistant = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()
if variant:  # doc-builder: hide
    assistant = WhisperForConditionalGeneration.from_pretrained(model_id, **variant).to(device).eval()
with torch.no_grad():
    assisted = model.generate(inputs.input_features, **generation, assistant_model=assistant)
assisted_text = processor.batch_decode(assisted, skip_special_tokens=True)
print(assisted_text)

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "speech": {"id": provenance["id"], "sha256": speech_sha256, "sampling_rate": sampling_rate},
        "greedy_tokens": greedy.tolist(),
        "greedy_text": greedy_text,
        "assisted_tokens": assisted.tolist(),
        "assisted_text": assisted_text,
        "assisted_matches_greedy": torch.equal(greedy, assisted),
    }
    print(json.dumps(result))
```

This inference-only probe complements the original contract's training checks. Results apply to the pinned checkpoint, speech sample, framework build, and runtime; agreement is not a speech-recognition accuracy measurement.
