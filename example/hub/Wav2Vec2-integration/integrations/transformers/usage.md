# Wav2Vec2: CTC speech recognition with Transformers

This page is the Transformers contract of the Wav2Vec2 MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `facebook/wav2vec2-base-960h`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

The audio is synthetic, so the example runs offline: four one-second clips, each a distinct pair of tones.

## Transcribe a batch

```py runnable:ctc
import math

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "facebook/wav2vec2-base-960h"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = Wav2Vec2ForCTC.from_pretrained(model_id, **variant).to(device)

sampling_rate = 16000
half = torch.arange(sampling_rate // 2) / sampling_rate
tones = [(220, 880), (440, 1320), (660, 330), (990, 550)]
clips = [torch.cat([torch.sin(2 * math.pi * a * half), torch.sin(2 * math.pi * b * half)]).mul(0.5).numpy() for a, b in tones]
inputs = processor(clips, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = logits.argmax(dim=-1)
print(processor.batch_decode(predicted_ids))
```

## One training step

```py runnable:ctc:2
transcripts = ["RED", "GREEN", "BLUE", "YELLOW"]
labels = processor(text=transcripts, return_tensors="pt", padding=True).input_ids.to(device)
labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("wav2vec2.feature_extractor.conv_layers.0.conv.weight", "wav2vec2.encoder.layers.0.attention.q_proj.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "predicted_ids": predicted_ids.tolist(),
        "predicted_text": processor.batch_decode(predicted_ids),
        "logits": logits.float().tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
