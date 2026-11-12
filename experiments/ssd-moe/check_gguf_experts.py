"""Check that the GGUF's Q8_0 experts are the safetensors experts: same layout, quantization-sized error only.

Usage: check_gguf_experts.py CHECKPOINT_DIR GGUF OUTPUT_JSON. Exits non-zero when a projection is not a close match
(e.g. gate and up swapped), so a benchmark never runs on wrongly mapped weights.
"""

import json
import sys

import torch

from transformers.integrations.ssd_moe import GGUFExpertReader, SafetensorsReader, dequantize_gguf


checkpoint, gguf_path, output = sys.argv[1:4]
safetensors = SafetensorsReader(checkpoint)
gguf = GGUFExpertReader(gguf_path)
prefix = "model.language_model.layers.{}.mlp.experts."
report = []
for layer, expert in ((0, 0), (17, 101), (39, 255)):
    gate_up = safetensors.read(prefix.format(layer) + "gate_up_proj", expert).float()
    reference = dict(zip(("gate", "up"), gate_up.chunk(2)))
    reference["down"] = safetensors.read(prefix.format(layer) + "down_proj", expert).float()
    for projection, expected in reference.items():
        kind, rows, columns, _, _, _ = gguf.layout(layer, projection)
        actual = dequantize_gguf(gguf.read(layer, projection, expert), kind, rows, columns, torch.float32)
        error = ((actual - expected).norm() / expected.norm()).item()
        report.append({"layer": layer, "expert": expert, "projection": projection, "relative_error": error})
        print(f"layer {layer} expert {expert} {projection}: relative error {error:.4f}")
json.dump(report, open(output, "w"), indent=2)
worst = max(item["relative_error"] for item in report)
if worst > 0.02:
    sys.exit(f"GGUF experts do not match the safetensors experts (worst relative error {worst:.3f})")
