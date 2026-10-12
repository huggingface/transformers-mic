import time

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "Qwen/Qwen3.5-0.8B"
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

t0 = time.time()
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype="auto")
print(f"load {time.time() - t0:.1f}s dtype={model.dtype} device={model.device} attn={model.config._attn_implementation}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_url},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
print(f"generate {time.time() - t0:.1f}s")
print(repr(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
