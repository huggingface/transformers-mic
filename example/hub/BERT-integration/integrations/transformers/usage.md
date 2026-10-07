# BERT: fill-mask with Transformers

This page is the Transformers contract of the BERT MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `google-bert/bert-base-uncased`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Predict the masked word

```py runnable:fill_mask
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "google-bert/bert-base-uncased"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = AutoModelForMaskedLM.from_pretrained(model_id, **variant).to(device)

countries = ["France", "Germany", "Italy", "the United Kingdom"]
prompts = [f"The capital of {country} is {tokenizer.mask_token}." for country in countries]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    logits = model(**inputs).logits
masked = inputs["input_ids"] == tokenizer.mask_token_id
predicted = logits[masked].argmax(dim=-1)
print(tokenizer.convert_ids_to_tokens(predicted))
```

## One training step

```py runnable:fill_mask:2
answers = ["paris", "berlin", "rome", "london"]
answer_ids = tokenizer.convert_tokens_to_ids(answers)
labels = torch.full_like(inputs["input_ids"], -100)
labels[masked] = torch.tensor(answer_ids, device=device)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("bert.encoder.layer.0.attention.self.query.weight", "bert.encoder.layer.0.intermediate.dense.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "predicted_tokens": predicted.tolist(),
        "predicted_text": tokenizer.convert_ids_to_tokens(predicted),
        "answer_logits": logits[masked][:, answer_ids].float().tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
