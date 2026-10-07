# SmolLM: batched text generation with Transformers

This page is the Transformers contract of the SmolLM MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `HuggingFaceTB/SmolLM-135M`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record. They do not appear in rendered docs.

## Generate

```py runnable:generate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "HuggingFaceTB/SmolLM-135M"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of the United Kingdom is",
]
if variant:  # doc-builder: hide
    model = AutoModelForCausalLM.from_pretrained(model_id, **variant).to(device)
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
print(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

with torch.inference_mode():  # doc-builder: hide
    logits = model(**inputs).logits
    answer_ids = [tokenizer.encode(a, add_special_tokens=False)[0] for a in (" Paris", " Berlin", " Rome", " London")]
    next_token_logits = logits[:, -1, answer_ids].float()
```

## One training step

```py runnable:generate:2
answers = [" Paris", " Berlin", " Rome", " London"]
tokenizer.padding_side = "right"  # pad after the text for training, so every label has real context
batch = tokenizer([p + a for p, a in zip(prompts, answers)], return_tensors="pt", padding=True).to(device)
labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**batch, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "generated_tokens": new_tokens.tolist(),
        "generated_text": tokenizer.batch_decode(new_tokens, skip_special_tokens=True),
        "next_token_logits": next_token_logits.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
