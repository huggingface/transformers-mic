# Qwen2.5-VL: image + text with Transformers

This page is the Transformers contract of the Qwen2.5-VL MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `Qwen/Qwen2.5-VL-3B-Instruct`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Inference with synthetic images

```py runnable:usage
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from PIL import Image
import numpy as np
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = AutoProcessor.from_pretrained(model_id, min_pixels=28*28, max_pixels=112*112)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **variant).to(device)

# Two distinct synthetic images
img1 = Image.fromarray(np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8))
img2 = Image.fromarray(np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8))

messages1 = [{"role": "user", "content": [{"type": "image", "image": img1}, {"type": "text", "text": "What color is this?"}]}]
messages2 = [{"role": "user", "content": [{"type": "image", "image": img2}, {"type": "text", "text": "Describe this."}]}]

texts = [
    processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for msgs in [messages1, messages2]
]

inputs = processor(
    text=texts,
    images=[img1, img2],
    padding=True,
    return_tensors="pt",
).to(device)
for key, value in inputs.items():
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        inputs[key] = value.to(model.dtype)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

# Compact probes: last 2 positions, first 5 token logits
probe_pos = range(-2, 0)
probe_tok = range(5)
selected_logits = [logits[0, p, probe_tok].float().tolist() for p in probe_pos]
print(f"selected_logits: {len(selected_logits)}x{len(probe_tok)}")
print(f"sample: {selected_logits[0][:3]}")
```

## One training step

```py runnable:usage:2
labels = inputs["input_ids"].clone()
if processor.tokenizer.pad_token_id is not None:
    labels[labels == processor.tokenizer.pad_token_id] = -100
labels[:, 0] = -100

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
            "device": model.device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "inference": {
            "selected_logits": selected_logits,
        },
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
