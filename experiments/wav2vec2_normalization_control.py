"""GPU control: do explicit normalization scalars restore full Wav2Vec2 calls?

Runs the published fixture and checkpoint contracts, including training, with
NumPy 1.26.4, NumPy 2.2.6, then NumPy 2.2.6 with an experimental normalization
control. Production contracts, weights, baselines and tolerances stay fixed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import dependency_study

ROOT = Path(__file__).resolve().parents[1]


def explicit_normalization(input_values, attention_mask, padding_value=0.0):
    import numpy as np

    lengths = np.array(attention_mask, np.int32).sum(-1) if attention_mask is not None else [len(x) for x in input_values]
    normalized = []
    for vector, length in zip(input_values, lengths):
        assert vector.dtype == np.float32, "This control is scoped to float32 processor audio"
        # NumPy 1 computes sqrt(var + epsilon) as float64, then rounds the
        # scalar to float32 when dividing a float32 array. Make both explicit.
        denominator = np.float32(np.sqrt(np.float64(vector[:length].var()) + np.float64(1e-7)))
        value = (vector - vector[:length].mean()) / denominator
        if length < len(value):
            value[length:] = padding_value
        normalized.append(value)
    return normalized


def fingerprints(arrays):
    return [{"shape": list(x.shape), "dtype": str(x.dtype), "sha256": hashlib.sha256(x.tobytes()).hexdigest()}
            for x in arrays]


def observe(variant, mode, out):
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor

    assert torch.cuda.is_available(), "Normalization experiment must run on GPU"
    original = Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
    preprocessing = []

    def normalization(values, attention_mask, padding_value=0.0):
        raw = fingerprints(values)
        output = (explicit_normalization if mode == "explicit" else original)(values, attention_mask, padding_value)
        preprocessing.append({"raw": raw, "normalized": fingerprints(output)})
        return output

    Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm = staticmethod(normalization)
    try:
        dependency_study.observe(variant, out, contract="Wav2Vec2")
    finally:
        Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm = staticmethod(original)
    result = json.loads(out.read_text())
    assert len(result["records"]) == len(preprocessing) == 2
    assert all(o["outputs"]["runtime"]["device"] == "cuda" for o in result["observations"])
    result.update(numpy=np.__version__, mode=mode, preprocessing=preprocessing)
    out.write_text(json.dumps(result, indent=2) + "\n")


def validate(folder):
    comparisons = []
    for variant in ("default", "eager", "bfloat16"):
        for repeat in (1, 2):
            samples = {name: json.loads((folder / name / f"{variant}-{repeat}.json").read_text())
                       for name in ("numpy1", "numpy2", "explicit")}
            for i in range(2):
                a, b, c = [samples[n] for n in ("numpy1", "numpy2", "explicit")]
                row = {"target": a["records"][i]["target"], "variant": variant, "repeat": repeat,
                       "statuses": {n: s["records"][i]["status"] for n, s in samples.items()},
                       "raw_inputs_identical": a["preprocessing"][i]["raw"] == b["preprocessing"][i]["raw"] == c["preprocessing"][i]["raw"],
                       "numpy2_normalization_changed": a["preprocessing"][i]["normalized"] != b["preprocessing"][i]["normalized"],
                       "explicit_normalization_matches_numpy1": a["preprocessing"][i]["normalized"] == c["preprocessing"][i]["normalized"],
                       "explicit_model_outputs_match_numpy1": a["observations"][i]["outputs"] == c["observations"][i]["outputs"],
                       "numpy2_failures": b["records"][i].get("failures", []),
                       "numeric_drift": {n: s["observations"][i]["numeric_drift"] for n, s in samples.items()}}
                comparisons.append(row)
    repeats = []
    for condition in ("numpy1", "numpy2", "explicit"):
        for variant in ("default", "eager", "bfloat16"):
            a, b = [json.loads((folder / condition / f"{variant}-{n}.json").read_text()) for n in (1, 2)]
            for x, y in zip(a["observations"], b["observations"]):
                repeats.append({"condition": condition, "variant": variant, "target": x["target"], "exact": x["outputs"] == y["outputs"]})
    report = {**json.loads((folder / "provenance.json").read_text()), "comparisons": comparisons, "repeats": repeats}
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    assert all(r["statuses"] == {"numpy1": "pass", "numpy2": "fail", "explicit": "pass"} for r in comparisons), comparisons
    assert all(r[k] for r in comparisons for k in ("raw_inputs_identical", "numpy2_normalization_changed",
               "explicit_normalization_matches_numpy1", "explicit_model_outputs_match_numpy1")), comparisons
    assert all(r["exact"] for r in repeats), repeats
    print("36 GPU calls reviewed: NumPy 2 flags reproduced; explicit normalization restores every NumPy 1 output exactly.")


def study(folder):
    folder.mkdir(parents=True, exist_ok=True)
    hashes = dependency_study.frozen_hashes()
    provenance = {"run_id": os.environ.get("GITHUB_RUN_ID"), "harness_sha": os.environ.get("GITHUB_SHA"),
                  "transformers_sha": subprocess.check_output(["git", "-C", "/tmp/transformers", "rev-parse", "HEAD"], text=True).strip(),
                  "frozen_hashes": hashes, "scope": "Float32 processor audio; inference and SGD training on both frozen Wav2Vec2 targets, default/eager/bfloat16."}
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for name, version, mode in (("numpy1", "1.26.4", "native"), ("numpy2", "2.2.6", "native"), ("explicit", "2.2.6", "explicit")):
        out = folder / name
        out.mkdir(exist_ok=True)
        with (out / "install.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", f"numpy=={version}"], stdout=log, stderr=log, check=True)
        (out / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        for variant in ("default", "eager", "bfloat16"):
            for repeat in (1, 2):
                with (out / f"{variant}-{repeat}.log").open("w") as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--observe", variant, "--mode", mode,
                                    "--out", str(out / f"{variant}-{repeat}.json")], cwd=ROOT, stdout=log, stderr=log, check=True)
                print(name, variant, repeat, "completed", flush=True)
        assert dependency_study.frozen_hashes() == hashes, "Frozen inputs or expectations changed"
    validate(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--observe", choices=("default", "eager", "bfloat16"))
    parser.add_argument("--mode", choices=("native", "explicit"), default="native")
    args = parser.parse_args()
    if args.observe:
        observe(args.observe, args.mode, args.out)
    else:
        study(args.out.resolve())
