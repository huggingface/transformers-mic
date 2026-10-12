# Qwen2: causal-LM with Transformers

This page is the Transformers contract of the Qwen2 MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `Qwen/Qwen2-0.5B`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Inference: next-token prediction

```py runnable:usage
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "Qwen/Qwen2-0.5B"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, **variant).to(device)  # doc-builder: hide
model.eval()

prompts = ["The capital of France is", "The sky is"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    logits = model(**inputs).logits

# compact probes: last-position logits for a small set of token ids
probe_ids = [tokenizer("Paris").input_ids[0], tokenizer("blue").input_ids[0], tokenizer("the").input_ids[0]]
last_pos = inputs["attention_mask"].sum(dim=-1) - 1
selected = logits[torch.arange(logits.size(0), device=device), last_pos][:, probe_ids].float()
predicted = logits[torch.arange(logits.size(0), device=device), last_pos].argmax(dim=-1)
print(tokenizer.convert_ids_to_tokens(predicted))
```

## One training step

```py runnable:usage:2
# build causal-LM labels: shift input_ids right, pad with -100
labels = torch.full_like(inputs["input_ids"], -100)
labels[:, 1:] = inputs["input_ids"][:, :-1]

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "predicted_tokens": predicted.tolist(),
        "selected_logits": selected.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
