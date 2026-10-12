# CSM Voice: text-to-audio with Transformers

This page is the Transformers contract of the CSM Voice MIC and also a usage example. The MIC runner runs the blocks below against each slim fixture and each checkpoint in [`integration.json`](../../integration.json).

## Forward pass (text → audio logits)

```py runnable:csm_voice
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "cartesia/azzurra-voice"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = CsmForConditionalGeneration.from_pretrained(model_id, **variant).to(device)

# Text paired with deterministic 24 kHz target audio exercises the codec,
# backbone and depth decoder through the public processor.
import numpy as np
sr = 24000
t = np.arange(sr // 2, dtype=np.float32) / sr
audio = [0.1 * np.sin(2 * np.pi * frequency * t) for frequency in (220, 440)]
prompts = ["[0]Hello, how are you today?<|AUDIO|>", "[1]The weather is nice outside.<|AUDIO|>"]
inputs = processor(text=prompts, audio=audio, sampling_rate=sr, output_labels=True, return_tensors="pt").to(device)
labels = inputs.pop("labels")
for key, value in inputs.items():
    if value.is_floating_point():
        inputs[key] = value.to(model.dtype)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

# Compact probes: first 4 sequence positions, first 8 audio-vocab entries
audio_logits = logits[:, :4, :8].float().tolist()
print(f"logits shape: {tuple(logits.shape)}")
print(f"audio_logits: {audio_logits}")
```

## One training step

```py runnable:csm_voice:2

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    trainable = [n for n, p in named.items() if p.requires_grad]
    checked = tuple(next(n for n, p in named.items() if component in n and p.grad is not None and torch.count_nonzero(p.grad) > 0) for component in ("backbone_model", "depth_decoder"))
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "audio_logits": audio_logits,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(named[n], before[n]) for n in trainable),
        },
    }
    print(json.dumps(result))
```
