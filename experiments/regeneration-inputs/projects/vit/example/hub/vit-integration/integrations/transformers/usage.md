# ViT: image classification with Transformers

This page is the Transformers contract of the ViT MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `google/vit-base-patch16-224`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Classify two synthetic images

```py runnable:vit_classify
import torch
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide
import numpy as np

model_id = "google/vit-base-patch16-224"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

config = AutoConfig.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = AutoModelForImageClassification.from_pretrained(model_id, **variant).to(device)

# Two distinct synthetic images (224x224x3, uint8)
img1 = np.zeros((224, 224, 3), dtype=np.uint8)
img1[:112, :112] = 255  # top-left white quadrant
img2 = np.zeros((224, 224, 3), dtype=np.uint8)
img2[112:, 112:] = 255  # bottom-right white quadrant

inputs = processor(images=[img1, img2], return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    logits = model(**inputs).logits

top5 = logits.topk(5, dim=-1)
top5_indices = top5.indices.tolist()
top5_probs = torch.softmax(logits, dim=-1).topk(5, dim=-1).values.float().tolist()

print(f"Top-5 labels: {top5_indices}")
print(f"Top-5 probs: {[[f'{p:.4f}' for p in row] for row in top5_probs]}")
```

## One training step

```py runnable:vit_classify:2
labels = torch.tensor([0, 1], device=device)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
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
        "top5_labels": top5_indices,
        "top5_probs": top5_probs,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
