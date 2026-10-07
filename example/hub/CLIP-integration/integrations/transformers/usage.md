# CLIP: zero-shot image classification with Transformers

This page is the Transformers contract of the CLIP MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `openai/clip-vit-base-patch32`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Match images to captions

```py runnable:clip
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "openai/clip-vit-base-patch32"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = CLIPModel.from_pretrained(model_id, **variant).to(device)

colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0)}
images = [Image.new("RGB", (256, 256), rgb) for rgb in colors.values()]
captions = [f"a photo of a {name} square" for name in colors]
inputs = processor(text=captions, images=images, return_tensors="pt", padding=True).to(device, model.dtype)  # floating-point inputs follow the model's dtype

with torch.no_grad():
    logits_per_image = model(**inputs).logits_per_image
best = logits_per_image.softmax(dim=1).argmax(dim=1)
for name, index in zip(colors, best.tolist()):
    print(f"{name} image -> {captions[index]!r}")
```

## One training step

```py runnable:clip:2
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, return_loss=True).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("visual_projection.weight", "text_projection.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "image_to_text": best.tolist(),
        "text_to_image": logits_per_image.argmax(dim=0).tolist(),
        "logits_per_image": logits_per_image.float().tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
