# GOT-OCR2: Vision-Language OCR with Transformers

This page is the Transformers contract of the GOT-OCR2 MIC and also a usage example. The MIC runner runs the same blocks against each slim fixture and each checkpoint in `integration.json`.

## Inference and training step

```py runnable:got_ocr2
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
import json
import os
import numpy as np
from PIL import Image

model_id = "stepfun-ai/GOT-OCR-2.0-hf"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = AutoProcessor.from_pretrained(model_id)

load_kwargs = {}
if "dtype" in variant:
    load_kwargs["torch_dtype"] = getattr(torch, variant.pop("dtype"))
if variant:
    load_kwargs.update(variant)

model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).to(device)

# Two distinct synthetic images: horizontal bar vs vertical bar
arr1 = np.full((224, 224, 3), 255, dtype=np.uint8)
arr1[100:110, 50:150] = 0
img1 = Image.fromarray(arr1)

arr2 = np.full((224, 224, 3), 255, dtype=np.uint8)
arr2[50:150, 100:110] = 0
img2 = Image.fromarray(arr2)

inputs = processor(
    images=[img1, img2],
    return_tensors="pt",
    padding=True,
).to(device)
for key, value in inputs.items():
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        inputs[key] = value.to(model.dtype)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
inference_logits = outputs.logits

# Compact probe: first 2 samples, last sequence position, first 4 token indices
probe = []
for i in range(min(2, inference_logits.shape[0])):
    for j in range(4):
        probe.append(inference_logits[i, -1, j].item())
```

```py runnable:got_ocr2:2
# Training step
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

labels = torch.full_like(inputs["input_ids"], -100)
labels[0, -2:] = torch.tensor([100, 200], device=device)
labels[1, -2:] = torch.tensor([300, 400], device=device)

loss = model(**inputs, labels=labels).loss
loss.backward()

before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()

# Gradient checks on first two parameters with non-None grad
grad_checks = {}
count = 0
for name, param in model.named_parameters():
    if param.grad is not None and count < 2:
        grad_checks[name] = bool(torch.isfinite(param.grad).all() and torch.count_nonzero(param.grad) > 0)
        count += 1

updated = any(not torch.equal(p, before[n]) for n, p in model.named_parameters())  # doc-builder: hide
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    result = {
        "runtime": {
            "device": device,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "inference_logits": probe,
        "training": {
            "loss": loss.item(),
            "gradients": grad_checks,
            "updated": updated,
        },
    }
    print(json.dumps(result))
```
