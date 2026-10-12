# Table Transformer Detection with Transformers

This page is the Transformers contract of the Table Transformer Detection MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `microsoft/table-transformer-detection`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Detect tables in synthetic images

```py runnable:table_detection
import torch
from transformers import AutoModelForObjectDetection, AutoProcessor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide
import numpy as np
from PIL import Image

model_id = "microsoft/table-transformer-detection"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = AutoModelForObjectDetection.from_pretrained(model_id, **variant).to(device)

# Two distinct synthetic images
np.random.seed(0)
img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

inputs = processor(images=[img1, img2], return_tensors="pt").to(device)

model.eval()
with torch.no_grad(), torch.autocast(device, dtype=model.dtype, enabled=model.dtype != torch.float32):
    outputs = model(**inputs)

# Compact probes: top-3 scores and first 2 boxes for image 0
scores = outputs.logits[0][:3].float()
boxes = outputs.pred_boxes[0][:2].float()
print(f"scores: {scores.tolist()}")
print(f"boxes: {boxes.tolist()}")
```

## One training step

```py runnable:table_detection:2
# Training targets: one box per image
labels = [
    {"class_labels": torch.tensor([0], device=device), "boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device)},
    {"class_labels": torch.tensor([1], device=device), "boxes": torch.tensor([[0.2, 0.2, 0.6, 0.6]], device=device)},
]

labels = [{k: v.to(device) for k, v in item.items()} for item in labels]

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
with torch.autocast(device, dtype=model.dtype, enabled=model.dtype != torch.float32):
    loss = model(pixel_values=inputs["pixel_values"], labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = tuple(n for n, p in named.items() if p.grad is not None and torch.count_nonzero(p.grad) > 0)[:2]
    result = {
        "runtime": {
            "device": next(model.parameters()).device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "inference_scores": scores.tolist(),
        "inference_boxes": boxes.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
