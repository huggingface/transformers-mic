# SpeechT5 ASR: speech-to-text with Transformers

This page is the Transformers contract of the SpeechT5 ASR MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `microsoft/speecht5_asr`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Inference: speech to text

```py runnable:speech_asr
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide
import numpy as np

model_id = "microsoft/speecht5_asr"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **variant).to(device)
model.eval()

# Two synthetic audio clips of different lengths (1 s and 2 s at 16 kHz)
sr = 16000
audio1 = np.random.randn(sr).astype(np.float32)
audio2 = np.random.randn(sr * 2).astype(np.float32)

inputs = processor(audio=[audio1, audio2], sampling_rate=sr, return_tensors="pt", padding=True).to(device)
for key, value in inputs.items():
    if value.is_floating_point():
        inputs[key] = value.to(model.dtype)

with torch.no_grad():
    decoder_input_ids = torch.full((inputs["input_values"].shape[0], 4), model.config.decoder_start_token_id, device=device, dtype=torch.long)
    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
logits = outputs.logits

# Compact probe: first 4 positions, first 5 vocab entries
selected_logits = logits[:, :4, :5].float()

# Greedy decode
generated_ids = model.generate(**inputs, max_new_tokens=10)
predicted_tokens = generated_ids.tolist()
predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(f"Predicted: {predicted_text}")
print(f"Selected logits shape: {selected_logits.shape}")
```

## One training step

```py runnable:speech_asr:2
target_texts = ["hello world", "good morning"]
labels = processor.tokenizer(target_texts, return_tensors="pt", padding=True).input_ids.to(device)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = tuple(n for n, p in model.named_parameters() if p.grad is not None and p.ndim >= 2 and torch.count_nonzero(p.grad) > 0)[:2]
    result = {
        "runtime": {
            "device": next(model.parameters()).device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_text,
        "selected_logits": selected_logits.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
