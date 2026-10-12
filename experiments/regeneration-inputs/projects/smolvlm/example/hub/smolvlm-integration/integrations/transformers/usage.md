# SmolVLM2: Vision-Language with Transformers

This page is the Transformers contract of the SmolVLM2 MIC and also a usage example. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Image + text inference

```py runnable:usage
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
import json
import os
from PIL import Image
import numpy as np

model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

load_kwargs = {}
for k, v in variant.items():
    if k == "dtype":
        load_kwargs["torch_dtype"] = getattr(torch, v)
    else:
        load_kwargs[k] = v

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).to(device)
model.eval()

# Two distinct synthetic images
rng = np.random.default_rng(0)
img1 = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
img2 = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))

prompts = [
    "<image>\nWhat is in this image?",
    "<image>\nDescribe the scene."
]

inputs = processor(images=[[img1], [img2]], text=prompts, return_tensors="pt", padding=True).to(device)
for key, value in inputs.items():
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        inputs[key] = value.to(model.dtype)

with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

# Compact numeric probe: first 5 positions, first 10 vocab entries of sample 0
selected_logits = logits[0, :5, :10].float().tolist()
print(f"selected_logits: {selected_logits}")
```

## One training step

```py runnable:usage:2
# Preserve inference outputs before training
inference_selected_logits = selected_logits  # doc-builder: hide

# Build labels for causal LM training (mask padded positions)
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    labels=labels,
).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = tuple(n for n, p in named.items() if p.grad is not None and p.ndim >= 2 and torch.count_nonzero(p.grad) > 0)[:2]
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "selected_logits": inference_selected_logits,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
