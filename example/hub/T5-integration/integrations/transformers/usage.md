# T5: text-to-text generation with Transformers

This page is the Transformers contract of the T5 MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `google-t5/t5-small`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json), including the UMT5 fixture: the usage path is the same, only the checkpoint changes. Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

## Translate a batch

```py runnable:translate
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "google-t5/t5-small"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **variant).to(device)

sentences = ["The house is wonderful.", "I like to read books.", "Where is the train station?", "Thank you!"]
inputs = tokenizer([f"translate English to German: {s}" for s in sentences], return_tensors="pt", padding=True).to(device)
generated = model.generate(**inputs, max_new_tokens=12, do_sample=False)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))
```

## Decode a batch of token ids

`decode` accepts a batch and returns one string per sequence, cleaning up the spaces the tokenizer puts before punctuation.

```py runnable:translate:3
punctuated = ["Yes , it is .", "Hello , world ! How are you ?"]
roundtrip = tokenizer.decode(tokenizer(punctuated).input_ids, skip_special_tokens=True)
print(roundtrip)
```

## One training step

```py runnable:translate:3
targets = ["Das Haus ist wunderbar.", "Ich lese gerne Bücher.", "Wo ist der Bahnhof?", "Danke!"]
labels = tokenizer(text_target=targets, return_tensors="pt", padding=True).input_ids.to(device)
labels[labels == tokenizer.pad_token_id] = -100

with torch.no_grad():  # doc-builder: hide
    # Teacher-forced log-probabilities of the reference translations.
    logprobs = model(**inputs, labels=labels).logits.float().log_softmax(-1)
    label_logprobs = logprobs.gather(-1, labels.clamp(min=0)[..., None])[..., 0].masked_fill(labels == -100, 0.0)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("encoder.block.0.layer.0.SelfAttention.q.weight", "decoder.block.0.layer.1.EncDecAttention.q.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "generated_tokens": generated.tolist(),
        "generated_text": tokenizer.batch_decode(generated, skip_special_tokens=True),
        "label_logprobs": label_logprobs.tolist(),
        "decoded_roundtrip": roundtrip,
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
