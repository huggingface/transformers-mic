"""Record actual GPU execution and whether the FP8 backend dequantizes."""
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import MistralLarge4ForConditionalGeneration
from transformers_mic.workspace import Workspace

workspace = Workspace("out/workspace")
pin = workspace.lock["contracts"][workspace.state["contract"]]["fixtures"][workspace.fixture]
artifact = (workspace.lock_path.parent / pin["path"]).resolve()
assert torch.cuda.is_available()
model = MistralLarge4ForConditionalGeneration.from_pretrained(
    artifact, dtype=torch.bfloat16, device_map="cuda"
).eval()
assert all(p.device.type == "cuda" for p in model.parameters())
with torch.no_grad():
    ids = torch.tensor([[1, 42, 43, 44, 45, 46]], device="cuda")
    first = model(input_ids=ids, use_cache=True)
    last = torch.tensor([[47]], device="cuda")
    cached = model(input_ids=last, past_key_values=first.past_key_values, use_cache=True)
    full = model(input_ids=torch.cat([ids, last], dim=1), use_cache=False)
    assert torch.isfinite(first.logits).all() and torch.isfinite(cached.logits).all()
    torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1])
result = {
    "gpu": torch.cuda.get_device_name(),
    "capability": torch.cuda.get_device_capability(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "dequantize": model.hf_quantizer.quantization_config.dequantize,
    "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
    "forward_finite": True,
    "cache_max_absolute_delta": (cached.logits[:, -1].float() - full.logits[:, -1].float()).abs().max().item(),
    "all_parameters_on_cuda": True,
}
Path("out/workspace/.mic/gpu-execution.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
