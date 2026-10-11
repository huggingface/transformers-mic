"""Summarize a downloaded whisper-left-padding GPU artifact and verify controls.

python experiments/summarize_whisper_left_padding.py <artifact-dir> --output <json>
"""

import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifact_dir", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--run-url", required=True)
parser.add_argument("--harness-commit", required=True)
args = parser.parse_args()
root = args.artifact_dir
records = {
    p.stem: json.loads(p.read_text())
    for p in root.glob("*.json")
    if p.stem.split("-")[0] in ["overfit", "random", "checkpoint"]
}
comparisons = []
repeats = []
for kind in ["checkpoint", "overfit", "random"]:
    fix, bad, repeat = [
        records[f"{kind}-{stage}"] for stage in ["fix", "bad", "repeat"]
    ]
    assert fix["weight_sha256"] == bad["weight_sha256"] == repeat["weight_sha256"]
    assert fix["probe_token_ids"] == bad["probe_token_ids"] == repeat["probe_token_ids"]
    for x, y, z in zip(fix["rows"], bad["rows"], repeat["rows"]):
        assert x["case"] == y["case"] == z["case"]
        allclose_flags = []
        deltas = []
        row_deltas = []
        for a, b in zip(x["traces"], y["traces"]):
            aa, bb = np.array(a["last_logits_probe"]), np.array(b["last_logits_probe"])
            if aa.shape != bb.shape or a["decoder_input_ids"] != b["decoder_input_ids"]:
                continue
            allclose_flags.append(bool(np.allclose(aa, bb, atol=1e-4, rtol=1e-4)))
            deltas.append(float(np.max(np.abs(aa - bb))))
            row_deltas.append(np.max(np.abs(aa - bb), axis=-1).tolist())
        entry = dict(
            model=kind,
            case=x["case"],
            generated_tokens_changed=x["sequences"] != y["sequences"],
            segments_changed=x["segments"] != y["segments"],
            fix_left_padded_prefills=x["left_padded_prefills"],
            bad_left_padded_prefills=y["left_padded_prefills"],
            matching_prefill_allclose_1e4=allclose_flags,
            matching_prefill_logit_deltas=deltas,
            matching_prefill_per_row_deltas=row_deltas,
        )
        if x["case"] == "short_unpadded":
            assert not entry["generated_tokens_changed"] and all(d == 0 for d in deltas)
        for t in x["traces"]:
            if t["decoder_attention_mask"] is not None:
                mask = np.array(t["decoder_attention_mask"])
                expected = np.maximum(mask.cumsum(-1) - 1, 0) * mask
                assert t["decoder_position_ids"] == expected.tolist()
        if x["case"] == "long_controlled":
            assert x["left_padded_prefills"] >= 1 and y["left_padded_prefills"] >= 1
            assert x["forced_first_segment_calls"] == y["forced_first_segment_calls"]
            assert all(a[0] == b[0] for a, b in zip(x["segments"], y["segments"]))
            assert len(x["segments"]) == len(y["segments"]) == 2
            entry["first_segments_identical"] = True
            entry["later_segment_rows_changed"] = [
                a[1:] != b[1:] for a, b in zip(x["segments"], y["segments"])
            ]
        comparisons.append(entry)

        def clean(r):
            return {k: v for k, v in r.items() if k != "seconds"}

        repeats.append(
            dict(model=kind, case=x["case"], fixed_repeat_exact=clean(x) == clean(z))
        )
        print(
            kind,
            x["case"],
            "tokens",
            entry["generated_tokens_changed"],
            "leftpad",
            x["left_padded_prefills"],
            "delta",
            max(deltas, default=0),
            "repeat",
            repeats[-1]["fixed_repeat_exact"],
        )
assert all(r["fixed_repeat_exact"] for r in repeats)
output = dict(
    date="2026-10-11",
    run=args.run_url,
    harness_commit=args.harness_commit,
    numerical_probe="subset of raw decoder prefill logits; excludes unequal input prefills",
    comparisons=comparisons,
    fixed_repeats=repeats,
    records=records,
    random_build=json.loads(
        (root / "random/proto-whisper-tiny-slim/build.json").read_text()
    ),
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(output, indent=1) + "\n")
