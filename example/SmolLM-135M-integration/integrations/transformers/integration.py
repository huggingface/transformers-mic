"""MIC for HuggingFaceTB/SmolLM-135M in Transformers: batched greedy generation.

The runner sets MODEL_ID, MODEL_REVISION, and MODEL_KWARGS (JSON) to point this
at the slim fixture or the real checkpoint. The last stdout line is the result;
logs go to stderr.
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ["MODEL_ID"]
revision = os.environ.get("MODEL_REVISION") or None
model_kwargs = json.loads(os.environ.get("MODEL_KWARGS", "{}"))

# Batched with left padding on purpose: padding and masking bugs only show up
# when prompts have different lengths.
prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of the United Kingdom is",
]
answers = [" Paris", " Berlin", " Rome", " London"]

# --- The usage path: what the model card shows users. ---
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **model_kwargs)
model.eval()

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
with torch.inference_mode():
    generated = model.generate(
        **inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    logits = model(**inputs).logits
new_tokens = generated[:, inputs["input_ids"].shape[1] :]

# Next-token logits for each answer's first token: a small numerical
# signature that moves with any change in the forward pass.
answer_ids = [tokenizer.encode(a, add_special_tokens=False)[0] for a in answers]
next_token_logits = logits[:, -1, answer_ids].float()

# --- Training probe: loss on the answers, gradients, one optimizer step. ---
model.train()
rows = [tokenizer.encode(p + a, add_special_tokens=False) for p, a in zip(prompts, answers)]
batch = tokenizer.pad({"input_ids": rows}, return_tensors="pt").to(model.device)
labels = batch["input_ids"].clone()
labels[batch["attention_mask"] == 0] = -100
for index, prompt in enumerate(prompts):
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    padding = int((batch["attention_mask"][index] == 0).sum())
    labels[index, : padding + prompt_length] = -100
before = {name: p.detach().clone() for name, p in model.named_parameters()}
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**batch, labels=labels).loss
loss.backward()
gradients = {
    name: bool(p.grad is not None and torch.count_nonzero(p.grad) > 0)
    for name, p in model.named_parameters()
    if name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight")
}
optimizer.step()
updated = any(not torch.equal(p, before[name]) for name, p in model.named_parameters())

print(
    json.dumps(
        {
            # What this run actually used; the runner checks it against the baseline.
            "runtime": {
                "device": str(model.device),
                "dtype": str(model.dtype).removeprefix("torch."),
                "attn_implementation": model.config._attn_implementation,
            },
            "generated_tokens": new_tokens.tolist(),
            "generated_text": tokenizer.batch_decode(new_tokens, skip_special_tokens=True),
            "next_token_logits": next_token_logits.tolist(),
            "training": {
                "loss": loss.item(),
                "gradients": gradients,
                "updated": updated,
            },
        }
    )
)
