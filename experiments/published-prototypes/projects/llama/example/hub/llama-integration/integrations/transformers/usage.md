# SmolLM2-135M: causal LM with Transformers

This page is the Transformers contract of the SmolLM2-135M MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `HuggingFaceTB/SmolLM2-135M`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Inference: next-token logit probes

```py runnable:causal_lm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "HuggingFaceTB/SmolLM2-135M"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = AutoModelForCausalLM.from_pretrained(model_id, **variant).to(device)

prompts = ["The capital of France is", "Hello, my name is"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

# Compact probes: logits at the last non-padding position for a few token ids
seq_lens = inputs["attention_mask"].sum(dim=-1)
probe_ids = [0, 100, 1000]
probes = []
for i in range(inputs["input_ids"].shape[0]):
    pos = seq_lens[i] - 1
    probes.append(logits[i, pos, probe_ids].float().tolist())
print(probes)
```

## One training step

```py runnable:causal_lm:2
model.train()
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100
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
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "logit_probes": probes,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
