# Mistral-Large-4: image-text-to-text with Transformers

This page is the Transformers contract of the Mistral-Large-4 MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `mistralai/Mistral-Large-4-1T-A52B`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Inference: chat completion

```py runnable:mistral_large4_chat
import torch
from transformers import AutoProcessor, MistralLarge4ForConditionalGeneration, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "mistralai/Mistral-Large-4-1T-A52B"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = AutoProcessor.from_pretrained(model_id)
model = MistralLarge4ForConditionalGeneration.from_pretrained(model_id, **variant).to(device)
# CPU and older GPUs dequantize FP8; explicitly align converted projections
# with the requested compute dtype, since loading may retain mixed dtypes.
if not getattr(model, "is_quantized", False):
    compute_dtype = getattr(torch, variant.get("dtype", "bfloat16"))
    model = model.to(dtype=compute_dtype)

messages = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "What is the capital of Germany?"}],
]
texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits

positions = torch.arange(inputs["attention_mask"].shape[1], device=device)
last_positions = positions.expand_as(inputs["attention_mask"]).masked_fill(inputs["attention_mask"] == 0, -1).max(dim=-1).values
batch_idx = torch.arange(len(texts), device=device)
selected_logits = logits[batch_idx, last_positions, :10].float()
predicted_ids = logits[batch_idx, last_positions].argmax(dim=-1)
predicted_tokens = [processor.tokenizer.decode([pid], skip_special_tokens=True) for pid in predicted_ids.tolist()]

print(f"predicted: {predicted_tokens}")
print(f"selected_logits: {selected_logits.tolist()}")
```

## One training step

```py runnable:mistral_large4_chat:2
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
        "predicted_tokens": predicted_tokens,
        "answer_logits": selected_logits.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
