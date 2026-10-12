"""Shrink layer counts by layer pattern instead of to a fixed count, and check it.

    python layer_patterns.py [--output results/layer_patterns.json]

For each real config, the rule keeps the first layer of every distinct layer kind,
in order. A layer's kind combines `layer_types[i]` with the dense/sparse choice
that MoE configs make per layer index (`first_k_dense_replace`, `moe_layer_freq`,
`mlp_only_layers`, `decoder_sparse_step`). Per-layer lists are re-indexed to the
kept layers. The rule is compared with today's fixed `num_hidden_layers = 2`.
Each shrunk config is built with small widths and random weights, then checked:
every kept kind is instantiated, every layer runs, and outputs, loss, and
gradients are finite.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

# repo, revision, and the width overrides that make each one a slim fixture.
CONFIGS = {
    "gemma3": (
        "google/gemma-3-1b-it",
        "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        dict(hidden_size=64, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=1, head_dim=16, sliding_window=4),
    ),
    "granite_hybrid": (
        "ibm-granite/granite-4.0-h-tiny",
        "791e0d3d28c86e106c9b6e0b4cecdee0375b6124",
        dict(hidden_size=64, intermediate_size=128, shared_intermediate_size=128,
             num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
             num_experts_per_tok=2, mamba_n_heads=8, mamba_d_head="auto",
             mamba_d_state=16, mamba_chunk_size=8),
    ),
    "deepseek_v2_lite": (
        "deepseek-ai/DeepSeek-V2-Lite",
        "604d5664dddd88a0433dbae533b7fe9472482de0",
        dict(hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
             num_attention_heads=4, num_key_value_heads=4, n_routed_experts=4,
             num_experts_per_tok=2, n_group=1, topk_group=1, kv_lora_rank=16,
             q_lora_rank=None, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16),
    ),
}


def is_sparse(config, i):
    """Whether layer i gets an MoE block, from the per-index rules MoE configs use."""
    if getattr(config, "first_k_dense_replace", None) is not None:
        return i >= config.first_k_dense_replace and i % getattr(config, "moe_layer_freq", 1) == 0
    if getattr(config, "mlp_only_layers", None) is not None:
        step = getattr(config, "decoder_sparse_step", 1)
        return i not in config.mlp_only_layers and (i + 1) % step == 0
    return None


def layer_kind(config, i):
    types = getattr(config, "layer_types", None)
    shared = getattr(config, "num_kv_shared_layers", 0)
    kv_shared = i >= config.num_hidden_layers - shared if shared else False
    return (types[i] if types else None, is_sparse(config, i), kv_shared)


def pattern_indices(config):
    kept, seen = [], set()
    for i in range(config.num_hidden_layers):
        kind = layer_kind(config, i)
        if kind not in seen:
            seen.add(kind)
            kept.append(i)
    return kept


def shrink_by_pattern(config, overrides):
    kept = pattern_indices(config)
    values = {**config.to_dict(), **overrides, "num_hidden_layers": len(kept)}
    for key, value in config.to_dict().items():
        # Per-layer lists follow the kept layers.
        if isinstance(value, list) and len(value) == config.num_hidden_layers:
            values[key] = [value[i] for i in kept]
    if getattr(config, "first_k_dense_replace", None) is not None:
        values["first_k_dense_replace"] = sum(
            i < config.first_k_dense_replace for i in kept
        )
    if getattr(config, "mlp_only_layers", None) is not None:
        values["mlp_only_layers"] = [n for n, i in enumerate(kept) if i in config.mlp_only_layers]
    return type(config).from_dict(values), kept


def shrink_fixed(config, overrides, layers=2):
    """Today's rule, as experiment.py applies it: assign num_hidden_layers on
    the loaded config, nothing else touched (widths as for the pattern rule)."""
    try:
        config = type(config).from_dict({**config.to_dict(), **overrides})
        config.num_hidden_layers = layers
        return config, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]


def built_kinds(model):
    """What each built layer actually is, read from the modules, not the config."""
    rows = []
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        mixer = (
            "mamba" if getattr(layer, "mamba", None) is not None
            else "sliding_attention" if getattr(attention, "is_sliding", False)
            else "attention"
        )
        mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
        rows.append({"mixer": mixer, "mlp": type(mlp).__name__})
    return rows


def run(config):
    try:
        return checked_run(config)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def checked_run(config):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    calls = [0] * len(model.model.layers)
    for n, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda m, a, o, n=n: calls.__setitem__(n, calls[n] + 1))
    ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids=ids, labels=ids)
    output.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return {
        "layers": len(model.model.layers),
        "parameters": sum(p.numel() for p in model.parameters()),
        "built": built_kinds(model),
        "every_layer_ran": all(calls),
        "logits_finite": bool(torch.isfinite(output.logits).all()),
        "loss": output.loss.item(),
        "gradients_finite": bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results" / "layer_patterns.json"
    )
    args = parser.parse_args()
    transformers.logging.set_verbosity_error()
    report = {"transformers": transformers.__version__, "torch": torch.__version__, "configs": {}}
    for name, (repo, revision, overrides) in CONFIGS.items():
        config = AutoConfig.from_pretrained(
            str(Path(hf_hub_download(repo, "config.json", revision=revision)).parent)
        )
        real_kinds = [layer_kind(config, i) for i in range(config.num_hidden_layers)]
        pattern, kept = shrink_by_pattern(config, overrides)
        fixed, fixed_error = shrink_fixed(config, overrides)
        row = {
            "source": [repo, revision],
            "real_layers": config.num_hidden_layers,
            "real_kinds": sorted({json.dumps(k) for k in real_kinds}),
            "pattern": {"kept_indices": kept, **run(pattern)},
            "fixed_2": run(fixed) if fixed else {"error": fixed_error},
        }
        for rule in ("pattern", "fixed_2"):
            built = {json.dumps([b["mixer"], b["mlp"]]) for b in row[rule].get("built", [])}
            row[rule]["distinct_built_kinds"] = sorted(built)
        report["configs"][name] = row
        print(json.dumps({name: {k: row[k] for k in ("real_layers", "real_kinds")}
                          | {r: {k: v for k, v in row[r].items() if k != "built"} for r in ("pattern", "fixed_2")}},
                         indent=1), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
