# InternVL3-1B: multimodal image+text with Transformers

This page is the Transformers contract of the InternVL3-1B MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `OpenGVLab/InternVL3-1B-hf`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Image+text inference

```py runnable:internvl
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
from PIL import Image
import json  # doc-builder: hide
import os  # doc-builder: hide
import numpy as np  # doc-builder: hide

model_id = "OpenGVLab/InternVL3-1B-hf"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = AutoProcessor.from_pretrained(model_id)
load_kwargs = dict(variant)  # doc-builder: hide
if "dtype" in load_kwargs:  # doc-builder: hide
    load_kwargs["torch_dtype"] = getattr(torch, load_kwargs.pop("dtype"))  # doc-builder: hide
model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).to(device)  # doc-builder: hide

# Two distinct synthetic images matching the model's expected input size (448x448)
img1 = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
img2 = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))

# Prompt-only conversations for inference
prompts = [
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image briefly."}]}],
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]}],
]
prompt_texts = [processor.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]

inputs = processor(images=[img1, img2], text=prompt_texts, return_tensors="pt", padding=True).to(device)

model.eval()
with torch.no_grad():
    out = model(**inputs)
logits = out.logits

# Compact probe: last 5 positions, first 5 vocab entries of sequence 0
probe = logits[0, -5:, :5].float().tolist()
print(f"logits probe: {probe}")
```

## One training step

```py runnable:internvl:2
# Full conversations (user + assistant) for supervised training
full_convs = [
    [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image briefly."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A colorful abstract pattern."}]},
    ],
    [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What do you see?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A random noise image."}]},
    ],
]
full_texts = [processor.apply_chat_template(c, tokenize=False) for c in full_convs]
train_inputs = processor(images=[img1, img2], text=full_texts, return_tensors="pt", padding=True).to(device)

# Labels: -100 for prompt positions, token ids for response positions
prompt_enc = processor(images=[img1, img2], text=prompt_texts, return_tensors="pt", padding=True)
prompt_lens = prompt_enc["input_ids"].ne(processor.tokenizer.pad_token_id).sum(dim=-1).to(device)
labels = train_inputs["input_ids"].clone()
for i in range(labels.shape[0]):
    labels[i, :prompt_lens[i]] = -100
    if processor.tokenizer.pad_token_id is not None:
        labels[i, train_inputs["input_ids"][i] == processor.tokenizer.pad_token_id] = -100

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**train_inputs, labels=labels).loss
loss.backward()
before = {n: p.detach().clone() for n, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    all_names = list(named.keys())
    vis_params = [n for n in all_names if "vision" in n and "layers.0" in n]
    lang_params = [n for n in all_names if "language" in n and "layers.0" in n and "q_proj" in n]
    checked = (vis_params[0] if vis_params else all_names[0], lang_params[0] if lang_params else all_names[-1])
    result = {
        "runtime": {
            "device": device,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "logits_probe": probe,
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
