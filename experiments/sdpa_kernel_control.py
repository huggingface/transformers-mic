"""Isolate the kernel change in Transformers #49517 using frozen MIC contracts.

The control keeps the new contiguous masks and forces MATH only for SDPA calls
whose position-bias mask previously had stride(-1) != 1. All other attention
calls keep their native dispatch. Contracts, baselines and tolerances are frozen.
"""

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import dependency_study

ROOT = Path(__file__).resolve().parents[1]


def previous_mask(position_bias, attention_mask, is_causal, query, key):
    """The parent commit's mask construction, before its final contiguous copy."""
    import torch

    minimum = torch.finfo(key.dtype).min
    if attention_mask is None:
        if not is_causal:
            return position_bias
        q_length, kv_length = query.shape[2], key.shape[2]
        causal = (
            torch.arange(q_length, device=key.device)[:, None]
            >= torch.arange(kv_length, device=key.device)[None, :]
        )
        return torch.where(
            causal.view(1, 1, q_length, kv_length), position_bias, minimum
        )
    if attention_mask.dtype == torch.bool:
        return torch.where(attention_mask, position_bias, minimum)
    return position_bias + attention_mask


@contextmanager
def math_control():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations import sdpa_attention

    mask_function = sdpa_attention.create_position_bias_mask
    attention_function = torch.nn.functional.scaled_dot_product_attention
    counts = {"masks_checked": 0, "math_calls": 0}

    def mask(*args, **kwargs):
        old = previous_mask(*args, **kwargs)
        current = mask_function(*args, **kwargs)
        assert torch.equal(old, current), (
            "Mask values changed in addition to their layout"
        )
        current._mic_previously_noncontiguous = old.stride(-1) != 1
        counts["masks_checked"] += 1
        return current

    def attention(*args, **kwargs):
        affected = getattr(
            kwargs.get("attn_mask"), "_mic_previously_noncontiguous", False
        )
        if affected:
            counts["math_calls"] += 1
        with sdpa_kernel(SDPBackend.MATH) if affected else nullcontext():
            return attention_function(*args, **kwargs)

    with (
        patch.object(sdpa_attention, "create_position_bias_mask", mask),
        patch.object(torch.nn.functional, "scaled_dot_product_attention", attention),
    ):
        yield counts


def observe(contract, variant, mode, out):
    import torch

    assert torch.cuda.is_available(), "This control requires CUDA"
    with math_control() if mode == "math" else nullcontext({}) as counts:
        dependency_study.observe(variant, out, contract=contract)
    result = json.loads(out.read_text())
    assert all(
        o["outputs"]["runtime"]["device"] == "cuda" for o in result["observations"]
    )
    if mode == "math":
        assert counts["math_calls"] > 0, "The control intercepted no affected calls"
    result.update(control=counts, mode=mode)
    out.write_text(json.dumps(result, indent=2) + "\n")


def study(out):
    import run

    out.mkdir(parents=True, exist_ok=True)
    frozen = dependency_study.frozen_hashes()
    provenance = {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "harness_sha": os.environ.get("GITHUB_SHA"),
        "transformers_sha": subprocess.check_output(
            ["git", "-C", "/tmp/transformers", "rev-parse", "HEAD"], text=True
        ).strip(),
        "frozen_hashes": frozen,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    samples = {}
    for contract in ("T5", "MusicgenMelody"):
        for variant in ("default", "bfloat16"):
            for mode in ("native", "math"):
                for repeat in (1, 2):
                    name = f"{contract}-{variant}-{mode}-{repeat}"
                    path = out / f"{name}.json"
                    with (out / f"{name}.log").open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--observe",
                                contract,
                                "--variant",
                                variant,
                                "--mode",
                                mode,
                                "--out",
                                str(path),
                            ],
                            cwd=ROOT,
                            stdout=log,
                            stderr=log,
                            check=True,
                        )
                    samples[contract, variant, mode, repeat] = json.loads(
                        path.read_text()
                    )
                    print(name, "completed", flush=True)
    rows, repeats = [], []
    for contract in ("T5", "MusicgenMelody"):
        manifest = run.load(
            ROOT / f"example/hub/{contract}-integration/integration.json", "integration"
        )
        rules = run.rules_for(
            manifest,
            ROOT / f"example/transformers/tests/contracts/expectations/{contract}/cuda",
        )
        for variant in ("default", "bfloat16"):
            for mode in ("native", "math"):
                first, second = [samples[contract, variant, mode, n] for n in (1, 2)]
                for record in first["records"]:
                    if record["status"] != "unsupported":
                        rows.append(
                            {
                                "contract": contract,
                                "target": record["target"],
                                "variant": variant,
                                "mode": mode,
                                "status": record["status"],
                                "failures": record.get("failures", []),
                            }
                        )
                assert [o["target"] for o in first["observations"]] == [
                    o["target"] for o in second["observations"]
                ]
                for a, b in zip(first["observations"], second["observations"]):
                    failures = run.compare(
                        rules, a["outputs"], b["outputs"], manifest["outputs"]
                    )
                    repeats.append(
                        {
                            "contract": contract,
                            "target": a["target"],
                            "variant": variant,
                            "mode": mode,
                            "exact": a["outputs"] == b["outputs"],
                            "failures": failures,
                        }
                    )
    report = {**provenance, "comparisons": rows, "repeats": repeats}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    assert dependency_study.frozen_hashes() == frozen, "Frozen files changed"
    assert len(rows) == 20, rows
    assert all(
        r["status"] == ("pass" if r["mode"] == "math" else "fail") for r in rows
    ), rows
    assert all(not r["failures"] for r in repeats), repeats
    print(
        "All 10 native failures reproduced; selective MATH restores all 10 old references. Fresh-process repeats agree."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--observe", choices=("T5", "MusicgenMelody"))
    parser.add_argument("--variant", choices=("default", "bfloat16"), default="default")
    parser.add_argument("--mode", choices=("native", "math"), default="native")
    args = parser.parse_args()
    if args.observe:
        observe(args.observe, args.variant, args.mode, args.out)
    else:
        study(args.out)
