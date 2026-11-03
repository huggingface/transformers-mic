"""The closest approved contract to a new checkpoint, as the planner's example.

A generic example (BERT) leaves the planner to invent every convention a large or
multimodal model needs, and each sampled draft invents them differently. Starting
from the approved contract whose checkpoint shares the most structure makes drafts
both better and more repeatable.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re

from huggingface_hub import hf_hub_download

from . import fleet


def signature(config):
    """What two configs can share: architecture class, model types, head."""
    architecture = (config.get("architectures") or [""])[0]
    types = {config.get("model_type")}
    for value in config.values():
        if isinstance(value, dict) and value.get("model_type"):
            types.add(value["model_type"])
    head = re.sub(r"^.*?(For\w+)$", r"\1", architecture) if "For" in architecture else ""
    try:  # the modality: input_features (speech), pixel_values (vision), input_ids (text)
        import transformers

        main_input = getattr(getattr(transformers, architecture, None), "main_input_name", None)
    except Exception:
        main_input = None
    return {
        "architecture": architecture,
        "model_type": config.get("model_type"),
        "types": {t for t in types if t},
        "head": head,
        "main_input": main_input,
        # deepseek_v32 ~ deepseek_v3, mistral_large4 ~ mistral3
        "family": (re.match(r"[a-z]+", config.get("model_type") or "") or [None])[0],
    }


def similarity(a, b):
    """Score and reason; 0 when nothing but the head is shared."""
    if a["architecture"] and a["architecture"] == b["architecture"]:
        return 100, f"same architecture {a['architecture']}"
    if a["model_type"] and a["model_type"] == b["model_type"]:
        return 50, f"same model_type {a['model_type']}"
    shared = sorted(a["types"] & b["types"])
    same_head = bool(a["head"]) and a["head"] == b["head"]
    # Whisper's head-only match was T5, a text model: its example never passes
    # decoder inputs with input_features. The input modality outranks the head.
    same_input = bool(a["main_input"]) and a["main_input"] == b["main_input"]
    same_family = bool(a["family"]) and a["family"] == b["family"]
    score = 10 * len(shared) + (9 if same_family else 0) + (8 if same_input else 0) + (5 if same_head else 0)
    if not shared and not same_head and not same_family:
        return 0, ""
    parts = ([f"shares {', '.join(shared)}"] if shared else []) + (
        [f"same {a['family']} family"] if same_family else []) + (
        [f"same {a['main_input']} input"] if same_input else []) + (
        [f"same {a['head']} head"] if same_head else [])
    return score, " and ".join(parts)


def candidates():
    """(fleet, contract name, integration pin, checkpoint pins) for both fleets."""
    rows = []
    for name in ("control", "prototypes"):
        lock = json.loads(Path(fleet.download(name)[0]).read_text())
        for contract, entry in sorted(lock["contracts"].items()):
            rows.append((name, contract, entry["integration"], entry["checkpoints"]))
    return rows


def _load(row):
    fleet_name, contract, integration, checkpoints = row
    try:
        repo, revision = next(iter(checkpoints.items()))
        config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
        manifest = json.loads(
            Path(
                hf_hub_download(integration["repo"], "integration.json", revision=integration["revision"])
            ).read_text()
        )
    except Exception:  # gated or removed checkpoints are not examples
        return None
    return fleet_name, contract, integration, config, manifest


def nearest_contract(config, rows=None, threads=16, model=None):
    """(example_plan, reason) for the closest approved contract, or None.

    TMIC_EXAMPLE_SKIP_SELF=1 skips contracts covering `model` itself, so the planner
    eval measures a new model's draft, not a copy of its approved contract.
    """
    target = signature(config)
    rows = rows if rows is not None else candidates()
    if model and os.environ.get("TMIC_EXAMPLE_SKIP_SELF"):
        rows = [row for row in rows if model not in row[3]]
    with ThreadPoolExecutor(threads) as pool:
        loaded = [r for r in pool.map(_load, rows) if r]
    best = None
    for fleet_name, contract, integration, other, manifest in loaded:
        score, reason = similarity(target, signature(other))
        # Ties prefer the reviewed control fleet, then the name, so the choice is stable.
        key = (score, fleet_name == "control", contract)
        if score and (best is None or key > best[0]):
            best = (key, contract, integration, manifest, reason)
    if best is None:
        return None
    _, contract, integration, manifest, reason = best
    entry = manifest["integrations"]["transformers"]["entry_point"]
    usage = Path(
        hf_hub_download(integration["repo"], entry, revision=integration["revision"])
    ).read_text()
    plan = {
        "name": contract,
        "manifest": manifest,
        "usage": usage,
        "rationale": f"Approved contract {contract}",
        "historical_candidates": [],
    }
    return plan, f"closest approved contract {contract} ({integration['repo']}): {reason}"
