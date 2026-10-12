# GPT-2: next-token prediction with Transformers

This page is the Transformers contract of the GPT-2 MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `openai-community/gpt2`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Predict the next token

```py runnable:next_token
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "openai-community/gpt2"
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

prompts = ["The quick brown fox", "Once upon a time"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

model.eval()
with torch.no_grad():
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

last_positions = inputs["attention_mask"].sum(dim=-1) - 1
last_logits = logits[torch.arange(logits.size(0)), last_positions]
predicted = last_logits.argmax(dim=-1)
print(tokenizer.convert_ids_to_tokens(predicted))
```

## One training step

```py runnable:next_token:2
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("transformer.h.0.attn.c_attn.weight", "transformer.h.0.mlp.c_fc.weight")
    top_k = 5
    top_logits = last_logits.topk(top_k, dim=-1).values.float().tolist()
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "predicted_tokens": predicted.tolist(),
        "predicted_text": tokenizer.convert_ids_to_tokens(predicted),
        "next_token_logits": top_logits,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
