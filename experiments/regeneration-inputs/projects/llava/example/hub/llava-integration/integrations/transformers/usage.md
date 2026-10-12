# LLaVA: vision-language with Transformers

This page is the Transformers contract of the LLaVA MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `llava-hf/llava-1.5-7b-hf`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Inference

```py runnable:usage
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, set_seed
from PIL import Image
import numpy as np
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "llava-hf/llava-1.5-7b-hf"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **variant).to(device)
model.eval()

np.random.seed(0)
img1 = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
img2 = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))

texts = [
    "<image>\nUSER: What do you see in this image?\nASSISTANT: A",
    "<image>\nUSER: Describe the colors in this image.\nASSISTANT: A",
]

inputs = processor(images=[img1, img2], text=texts, return_tensors="pt", padding=True).to(device)
for key, value in inputs.items():
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        inputs[key] = value.to(model.dtype)

with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

probe_positions = [0, 1, 2, 3, 4]
probe = logits[:, -1, probe_positions].float()
predicted = logits[:, -1].argmax(dim=-1)
print(f"predicted: {predicted.tolist()}")
print(f"probe: {probe.tolist()}")
```

## One training step

```py runnable:usage:2
# Fixed target tokens to ensure a meaningful loss
target_tokens = torch.tensor([100, 200], device=device)
labels = torch.full_like(inputs["input_ids"], -100)
labels[:, -1] = target_tokens

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()

# doc-builder: hide
checked = tuple(n for n, p in model.named_parameters() if p.grad is not None and p.ndim >= 2 and torch.count_nonzero(p.grad) > 0)[:2]
before = {n: p.detach().clone() for n, p in model.named_parameters() if n in checked}
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    result = {
        "runtime": {
            "device": next(model.parameters()).device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "predicted_tokens": predicted.tolist(),
        "probe_logits": probe.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(named[n], before[n]) for n in checked),
        },
    }
    print(json.dumps(result))
```
