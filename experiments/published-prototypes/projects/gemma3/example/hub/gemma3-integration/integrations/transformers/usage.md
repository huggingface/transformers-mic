# Gemma3: causal LM with Transformers

This page is the Transformers contract of the Gemma3 MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `google/gemma-3-4b-it`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Next-token prediction

```py runnable:gemma3
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "google/gemma-3-4b-it"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = AutoModelForCausalLM.from_pretrained(model_id, **variant).to(device)

prompts = ["The sky is", "Water is"]
inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

# Last-token logits for next-token prediction
seq_lens = inputs["attention_mask"].sum(dim=-1) - 1
last_logits = logits[torch.arange(inputs["input_ids"].shape[0], device=device), seq_lens]
predicted_ids = last_logits.argmax(dim=-1)
predicted_tokens = processor.tokenizer.convert_ids_to_tokens(predicted_ids)
predicted_text = processor.tokenizer.decode(predicted_ids, skip_special_tokens=False)
print(predicted_tokens)
```

## One training step

```py runnable:gemma3:2
# Causal LM labels: shift input_ids left, mask padding
input_ids = inputs["input_ids"]
labels = torch.roll(input_ids, shifts=-1, dims=-1)
if processor.tokenizer.pad_token_id is not None:
    labels[input_ids == processor.tokenizer.pad_token_id] = -100

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
    top3 = last_logits.topk(3, dim=-1).values.float()
    result = {
        "runtime": {
            "device": next(model.parameters()).device.type,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "predicted_tokens": predicted_tokens,
        "predicted_text": predicted_text,
        "answer_logits": top3.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
