# PI0: camera and state observations to actions

The contract uses the checkpoint documented by Transformers, `lerobot/pi0_base`.
The same public forward and action-sampling methods run against the slim fixture.
Two camera views, different state vectors, fixed diffusion noise and explicit
timesteps make the flow-matching training step reproducible.

```py runnable:pi0_policy
import torch
from transformers import PI0ForConditionalGeneration, set_seed
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "lerobot/pi0_base"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)
model = PI0ForConditionalGeneration.from_pretrained(model_id, **variant).to(device)
config = model.config
vision = config.vlm_config.vision_config
size = vision.image_size
image_tokens = 2 * (size // vision.patch_size) ** 2
input_ids = torch.full((2, image_tokens + 8), config.vlm_config.image_token_index, device=device, dtype=torch.long)
input_ids[:, image_tokens:] = torch.arange(10, 26, device=device).reshape(2, 8)
attention_mask = torch.ones_like(input_ids)
pixels = torch.rand(2, 2, 3, size, size, device=device, dtype=model.dtype)
state = torch.rand(2, config.max_state_dim, device=device, dtype=model.dtype)
noise = torch.randn(2, config.chunk_size, config.max_action_dim, device=device, dtype=model.dtype)
pixel_attention_mask = torch.ones(2, 2, device=device, dtype=torch.bool)
inputs = dict(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixels, pixel_attention_mask=pixel_attention_mask, state=state, noise=noise)

model.eval()
with torch.no_grad():
    actions = model.sample_actions(**inputs, num_steps=2)
print(actions[0, 0, :3].float().tolist())
print(actions[0, -1, :3].float().tolist())
```

```py runnable:pi0_policy:2
target_actions = torch.rand_like(noise)
timestep = torch.tensor([0.25, 0.75], device=device)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = model(**inputs, actions=target_actions, timestep=timestep).loss
loss.backward()
named = dict(model.named_parameters())
checked = tuple(next(n for n, p in named.items() if component in n and p.grad is not None and torch.count_nonzero(p.grad) > 0) for component in ("model.vlm", "model.dit"))
before = {n: named[n].detach().clone() for n in checked}
optimizer.step()
print(float(loss))

if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    print(json.dumps({
        "runtime": {"device": model.device.type, "dtype": str(model.dtype).removeprefix("torch."), "attn_implementation": config._attn_implementation},
        "action_first_step": actions[0, 0, :3].float().tolist(),
        "action_last_step": actions[0, -1, :3].float().tolist(),
        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(torch.isfinite(named[n].grad).all() and torch.count_nonzero(named[n].grad) > 0) for n in checked},
            "updated": any(not torch.equal(named[n], before[n]) for n in checked),
        },
    }))
```
