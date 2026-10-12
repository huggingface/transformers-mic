"""Diagnose NumPy promotion at Wav2Vec2's normalization boundary, no model.

Run once with each NumPy version and the same pinned Transformers source.
Writes raw audio, processor outputs and an explicit scalar-precision control.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
half = torch.arange(8000) / 16000
clips = [torch.cat([torch.sin(2 * math.pi * a * half), torch.sin(2 * math.pi * b * half)]).mul(0.5).numpy()
         for a, b in ((220, 880), (440, 1320), (660, 330), (990, 550))]
extractor = Wav2Vec2FeatureExtractor(do_normalize=True, return_attention_mask=False)
actual = extractor(clips, sampling_rate=16000, return_tensors="np").input_values
control = np.stack([(x - x.mean()) / np.float32(np.sqrt(np.float64(x.var()) + np.float64(1e-7))) for x in clips])
arrays = {"raw": np.stack(clips), "processed": actual, "scalar_precision_control": control}
metadata = {"numpy": np.__version__, "torch": torch.__version__,
            "normalization_denominator_dtype": str(np.asarray(np.sqrt(clips[0].var() + 1e-7)).dtype),
            "arrays": {}}
for name, array in arrays.items():
    np.save(args.out / f"{name}.npy", array)
    metadata["arrays"][name] = {"dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
(args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata))
