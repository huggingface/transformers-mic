# Whisper: speech recognition with Transformers

This page is the Transformers contract of the Whisper MIC and also a usage example. [doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) renders the `runnable` blocks below as documentation, and its pytest plugin runs them as written against `openai/whisper-tiny`. The MIC runner runs the same blocks against each slim fixture and each checkpoint in [`integration.json`](../../integration.json). Lines marked `# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and print the result record.

The audio is synthetic, so the example runs offline: four one-second clips, each a distinct pair of tones. A real checkpoint transcribes them as noise; the slim fixture is trained to name them.

## Transcribe a batch

```py runnable:transcribe
import math

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "openai/whisper-tiny"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    model = WhisperForConditionalGeneration.from_pretrained(model_id, **variant).to(device)

sampling_rate = 16000
half = torch.arange(sampling_rate // 2) / sampling_rate
tones = [(220, 880), (440, 1320), (660, 330), (990, 550)]
clips = [torch.cat([torch.sin(2 * math.pi * a * half), torch.sin(2 * math.pi * b * half)]).mul(0.5).numpy() for a, b in tones]
inputs = processor(clips, sampling_rate=sampling_rate, return_tensors="pt").to(device, model.dtype)  # floating-point inputs follow the model's dtype

generated = model.generate(inputs.input_features, language="en", task="transcribe", max_new_tokens=8)
print(processor.batch_decode(generated, skip_special_tokens=True))
```

## Transcribe with timestamps

Whisper can predict segment timestamps; decoding with `decode_with_timestamps=True` keeps them in the text. Integrations also read the feature extractor's configuration without building a processor.

```py runnable:transcribe:3
from transformers.models.auto.feature_extraction_auto import get_feature_extractor_config

timestamped = model.generate(inputs.input_features, language="en", task="transcribe", return_timestamps=True, max_new_tokens=8)
timestamped_text = processor.batch_decode(timestamped, decode_with_timestamps=True)
print(timestamped_text)

feature_config = get_feature_extractor_config(model_id)
print(sorted(feature_config))
```

## Limit the transcript length

For Whisper, `max_length` counts transcript tokens: `generate` adds the forced start, language, and task tokens on top.

```py runnable:transcribe:3
limited = model.generate(inputs.input_features, language="en", task="transcribe", max_length=1)
print(processor.batch_decode(limited, skip_special_tokens=True))
```

## Speculative decoding

A smaller decoder-only Whisper can draft tokens that the full model verifies, as [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3) does. Here the checkpoint's own decoder, loaded as `WhisperForCausalLM`, is the assistant; the transcript must not change. Assisted generation takes one clip at a time.

```py runnable:transcribe:3
from transformers import WhisperForCausalLM

assistant = WhisperForCausalLM.from_pretrained(model_id).to(device)
if variant:  # doc-builder: hide
    assistant = WhisperForCausalLM.from_pretrained(model_id, **variant).to(device)
assisted = [
    model.generate(features[None], language="en", task="transcribe", max_new_tokens=8, assistant_model=assistant)[0].tolist()
    for features in inputs.input_features
]
print(processor.batch_decode(assisted, skip_special_tokens=True))
```

## Transcribe long audio in a batch

Audio longer than 30 seconds is transcribed segment by segment. Keep the full audio (`truncation=False`), pad the batch to its longest clip, and pass the attention mask; `condition_on_prev_tokens=True` prompts each segment with the previous one's text.

```py runnable:transcribe:3
long_clips = [
    torch.cat([torch.from_numpy(clips[(start + k) % len(clips)]) for k in range(seconds)]).numpy()
    for start, seconds in ((0, 36), (1, 32))
]
long_inputs = processor(
    long_clips, sampling_rate=sampling_rate, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True
).to(device, model.dtype)
longform = model.generate(
    **long_inputs, language="en", task="transcribe", condition_on_prev_tokens=True, return_timestamps=True,
    temperature=0.0, max_new_tokens=8,
)
print(processor.batch_decode(longform, skip_special_tokens=True))
```

## Continue from transcripts of different lengths

Previous-segment transcripts can have different lengths across a batch. Whisper left-pads the shorter prompt before decoding the next segment. This reproducible example constrains the first segment to one word in the first row and three words in the second; the remaining audio is transcribed freely. The custom processor only chooses the first segment's tokens.

```py runnable:transcribe:3
from transformers import LogitsProcessor

tokenizer = processor.tokenizer
timestamp_begin = tokenizer.convert_tokens_to_ids("<|0.00|>")
previous_token = tokenizer.convert_tokens_to_ids("<|startofprev|>")
first_segment_tokens = [
    [timestamp_begin] + tokenizer.encode(text, add_special_tokens=False) + [timestamp_begin + 1500, model.config.eos_token_id]
    for text in (" red", " green blue yellow")
]

class FirstSegmentTranscript(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if any(previous_token in row.tolist() for row in input_ids):
            return scores
        # The initial prefix is start-of-transcript, English, transcribe.
        step = input_ids.shape[-1] - 3
        constrained = torch.full_like(scores, -float("inf"))
        for row, tokens in enumerate(first_segment_tokens):
            token = tokens[step] if step < len(tokens) else model.config.eos_token_id
            constrained[row, token] = 0
        return constrained

continuation_inputs = processor(
    [clip.repeat(36) for clip in clips[:2]], sampling_rate=sampling_rate, return_tensors="pt",
    truncation=False, padding="longest", return_attention_mask=True,
).to(device, model.dtype)
continued = model.generate(
    **continuation_inputs, language="en", task="transcribe", condition_on_prev_tokens=True,
    return_timestamps=True, temperature=0.0, max_new_tokens=12,
    logits_processor=[FirstSegmentTranscript()],
)
print(processor.batch_decode(continued, skip_special_tokens=True))
```

## One training step

```py runnable:transcribe:3
transcripts = [" red", " green", " blue", " yellow"]
processor.tokenizer.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)
targets = processor.tokenizer(transcripts, return_tensors="pt", padding=True).to(device)
decoder_input_ids = targets.input_ids[:, :-1]
labels = targets.input_ids[:, 1:].masked_fill(targets.attention_mask[:, 1:] == 0, -100)

with torch.no_grad():  # doc-builder: hide
    # Teacher-forced log-probabilities of the reference transcripts.
    logprobs = model(input_features=inputs.input_features, decoder_input_ids=decoder_input_ids).logits.float().log_softmax(-1)
    label_logprobs = logprobs.gather(-1, labels.clamp(min=0)[..., None])[..., 0].masked_fill(labels == -100, 0.0)

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(input_features=inputs.input_features, decoder_input_ids=decoder_input_ids, labels=labels).loss
loss.backward()
before = {name: p.detach().clone() for name, p in model.named_parameters()}  # doc-builder: hide
optimizer.step()
print(f"loss: {loss.item():.4f}")

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    checked = ("model.encoder.conv1.weight", "model.decoder.layers.0.encoder_attn.q_proj.weight")
    result = {
        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },
        "generated_tokens": generated.tolist(),
        "generated_text": processor.batch_decode(generated, skip_special_tokens=True),
        "label_logprobs": label_logprobs.tolist(),
        "timestamped_tokens": timestamped.tolist(),
        "timestamped_text": timestamped_text,
        "feature_extractor_config_keys": sorted(feature_config),
        "max_length_tokens": limited.tolist(),
        "assisted_tokens": assisted,
        "longform_tokens": longform.tolist(),
        "continued_longform_tokens": continued.tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },
    }
    print(json.dumps(result))
```
