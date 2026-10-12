# GPT-OSS: causal LM with Transformers

This page is the Transformers contract of the GPT-OSS MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `openai/gpt-oss-20b`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Forward pass and one training step

```py runnable:usage
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "openai/gpt-oss-20b"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = AutoModelForCausalLM.from_pretrained(model_id, **variant).to(device)

prompts = [
    "The capital of France is",
    "The sky is",
]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

model.eval()
with torch.no_grad():
    logits = model(**inputs).logits

# Compact probe: last position, first 8 vocab entries
probe_logits = logits[:, -1, :8].float()
print(probe_logits)

predicted_ids = logits[:, -1].argmax(dim=-1)
print(tokenizer.convert_ids_to_tokens(predicted_ids))
```

```py runnable:usage:2
# Causal-LM training step
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100

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
        "predicted_tokens": predicted_ids.tolist(),
        "answer_logits": probe_logits.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
