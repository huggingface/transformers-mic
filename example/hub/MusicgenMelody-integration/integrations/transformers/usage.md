# MusicGen Melody: text- and melody-conditioned music generation with Transformers

This page is the Transformers contract of the MusicGen Melody MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `facebook/musicgen-melody`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

The melodies are synthetic, so the example runs offline: two one-second clips, each a pair of notes. MusicGen Melody turns each clip into a chromagram and conditions on it together with the text prompt.

## Generate from a text prompt and a melody

`generate` samples by default; `do_sample=False` decodes greedily so the output is reproducible. `guidance_scale` enables classifier-free guidance, which also runs the model on an empty prompt and pushes the result away from it. With `output_scores=True`, the per-step scores give the audio codes the model chose; `generate` returns the waveform decoded from them.

```py runnable:generate
import math

import torch
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "facebook/musicgen-melody"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenMelodyForConditionalGeneration.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(model_id, **variant).to(device)

sampling_rate = processor.feature_extractor.sampling_rate
half = torch.arange(sampling_rate // 2) / sampling_rate
notes = [(262, 330), (392, 494)]  # C4 then E4; G4 then B4
melodies = [torch.cat([torch.sin(2 * math.pi * f * half) for f in pair]).mul(0.5).numpy() for pair in notes]
prompts = ["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"]

inputs = processor(audio=melodies, sampling_rate=sampling_rate, text=prompts, padding=True, return_tensors="pt").to(device)
inputs["input_features"] = inputs["input_features"].to(model.dtype)  # floating-point inputs follow the model's dtype
generated = model.generate(
    **inputs, do_sample=False, guidance_scale=3, max_new_tokens=8, return_dict_in_generate=True, output_scores=True
)
codes = torch.stack(generated.scores).argmax(-1).reshape(len(generated.scores), len(prompts), -1).permute(1, 2, 0)
audio = generated.sequences  # (batch, channels, samples) at the audio encoder's sampling rate
print(codes.shape, audio.shape, model.config.audio_encoder.sampling_rate)
```

## Generate from text only

The same processor prepares a text-only prompt; without a melody the model conditions on the text alone.

```py runnable:generate:2
text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
text_only = model.generate(
    **text_inputs, do_sample=False, guidance_scale=3, max_new_tokens=8, return_dict_in_generate=True, output_scores=True
)
text_only_codes = torch.stack(text_only.scores).argmax(-1).reshape(len(text_only.scores), len(prompts), -1).permute(1, 2, 0)
print(text_only_codes.shape)
```

## One training step

The decoder is trained on audio codes, one stream per codebook. Here the targets are fixed codes for each prompt, as a fine-tuning batch would provide after encoding audio with `model.audio_encoder`.

```py runnable:generate:3
num_codebooks = model.decoder.num_codebooks
frames = 8
targets = torch.Generator().manual_seed(0)
labels = torch.randint(0, model.decoder.config.vocab_size, (len(prompts), frames, num_codebooks), generator=targets).to(device)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("text_encoder.encoder.block.0.layer.0.SelfAttention.q.weight", "audio_enc_to_dec_proj.weight", "decoder.model.decoder.layers.0.self_attn.q_proj.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "input_feature_shape": list(inputs["input_features"].shape),
        "codes": codes.tolist(),
        "audio_shape": list(audio.shape),
        "audio_head": audio[:, 0, :32].float().tolist(),
        "text_only_codes": text_only_codes.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
