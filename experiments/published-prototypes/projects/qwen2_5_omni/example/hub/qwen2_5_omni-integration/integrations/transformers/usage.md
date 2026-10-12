# Qwen2.5-Omni: text generation with Transformers

This page is the Transformers contract of the Qwen2.5-Omni MIC and also a usage example. The MIC runner runs the blocks below against each slim fixture and each checkpoint in [`integration.json`](../../integration.json).

## Inference and one training step

```py runnable:omni_text
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "Qwen/Qwen2.5-Omni-7B"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)

processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

load_kwargs = {}  # doc-builder: hide
if variant:  # doc-builder: hide
    if "dtype" in variant:
        load_kwargs["torch_dtype"] = variant["dtype"]
    for k, v in variant.items():
        if k != "dtype":
            load_kwargs[k] = v

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **load_kwargs).to(device)  # doc-builder: hide

conversations = [
    [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
    ],
    [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
    ],
]

texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
for k, v in inputs.items():  # doc-builder: hide
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        inputs[k] = v.to(model.dtype)

model.eval()
with torch.no_grad():
    outputs = model.thinker(**inputs)

selected_logits = outputs.logits[0, :5, :5].float().tolist()

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False, return_audio=False)
generated_tokens = generated_ids[0, inputs["input_ids"].size(1):].tolist()
print(f"generated: {generated_tokens}")
```

```py runnable:omni_text:2
# Training step on the thinker causal-LM head
labels = torch.full_like(inputs["input_ids"], -100)
seq_lens = inputs["attention_mask"].sum(dim=1)
for i in range(inputs["input_ids"].size(0)):
    start = max(0, seq_lens[i].item() - 5)
    labels[i, start:seq_lens[i]] = inputs["input_ids"][i, start:seq_lens[i]]

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model.thinker(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels).loss
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
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": getattr(model.config, "_attn_implementation", "unknown"),
        },
        "selected_logits": selected_logits,
        "generated_token_ids": generated_tokens,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
