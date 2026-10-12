# DETR: object detection with Transformers

This page is the Transformers contract of the DETR MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `facebook/detr-resnet-50`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Inference on synthetic images

```py runnable:detr_detect
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor, set_seed
import numpy as np
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "facebook/detr-resnet-50"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = DetrImageProcessor.from_pretrained(model_id)
model = DetrForObjectDetection.from_pretrained(model_id, **variant).to(device)

# Two distinct synthetic images so batch perturbation is meaningful
np.random.seed(0)
images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(2)]

inputs = processor(images, size={"shortest_edge": 64, "longest_edge": 64}, return_tensors="pt").to(device)

model.eval()
with torch.no_grad(), torch.autocast(device, dtype=model.dtype, enabled=model.dtype != torch.float32):
    outputs = model(**inputs)

# Compact probes: first 3 queries, first 5 class logits for the first image
top_class_logits = outputs.logits[0, :3, :5].float().tolist()
# First 3 predicted boxes for the first image
pred_boxes = outputs.pred_boxes[0, :3].float().tolist()

print(top_class_logits)
print(pred_boxes)
```

## One training step

```py runnable:detr_detect:2
# Target boxes in normalized (x_center, y_center, width, height) format
labels = [
    {"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]), "class_labels": torch.tensor([1])},
    {"boxes": torch.tensor([[0.4, 0.6, 0.2, 0.2]]), "class_labels": torch.tensor([3])},
]

labels = [{k: v.to(device) for k, v in item.items()} for item in labels]

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
with torch.autocast(device, dtype=model.dtype, enabled=model.dtype != torch.float32):
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
            "device": model.device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "top_class_logits": top_class_logits,
        "pred_boxes": pred_boxes,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
