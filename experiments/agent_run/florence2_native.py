import sys, time, requests, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_id = sys.argv[1]
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw)
t0 = time.time()
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id)
print(f"load {time.time()-t0:.1f}s {type(model).__name__}")
for task in ["<CAPTION>", "<OD>"]:
    inputs = processor(text=task, images=image, return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=256, num_beams=3, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    print(task, processor.post_process_generation(text, task=task, image_size=(image.width, image.height)))
