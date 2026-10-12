# Qwen2-Audio: audio understanding with Transformers

This page is the Transformers contract of the Qwen2-Audio MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `Qwen/Qwen2-Audio-7B`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Forward pass with synthetic audio

```py runnable:qwen2_audio
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, set_seed
import numpy as np
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "Qwen/Qwen2-Audio-7B"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **variant).to(device)

# Two synthetic audio clips (16 kHz mono) for meaningful batch perturbation
sr = 16000
rng = np.random.default_rng(0)
audio1 = rng.standard_normal(sr).astype(np.float32)
audio2 = rng.standard_normal(int(sr * 1.5)).astype(np.float32)

prompts = ["<|audio_bos|><|AUDIO|><|audio_eos|> What do you hear?", "<|audio_bos|><|AUDIO|><|audio_eos|> Describe the sound."]

inputs = processor(
    audio=[audio1, audio2],
    sampling_rate=sr,
    text=prompts,
    return_tensors="pt",
    padding=True,
).to(device)
for key, value in inputs.items():
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        inputs[key] = value.to(model.dtype)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits.float()
# Compact probe: first 3 positions, first 8 vocab positions
selected = logits[:, :3, :8].tolist()
print(selected)
```

## One training step

```py runnable:qwen2_audio:2
# Build labels: mask everything except the last 2 tokens of each sequence
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = torch.full_like(input_ids, -100)
for i in range(input_ids.shape[0]):
    seq_len = attention_mask[i].sum().item()
    labels[i, seq_len - 2: seq_len] = input_ids[i, seq_len - 2: seq_len]

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    audio_param = next(n for n in named if "audio_tower" in n and named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0)
    text_param = next(n for n in named if "language_model" in n and named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0)
    checked = (audio_param, text_param)
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "selected_logits": selected,
        "training": {
            "loss": loss.item(),
            "gradients": {
                n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all())
                for n in checked
            },
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
