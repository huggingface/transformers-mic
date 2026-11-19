"""What an integration's runtime needs from a slim fixture's config, as config_overrides.

The generic shrink rule (experiments/sweep.py WIDTHS) targets Transformers: four
attention heads of width 16. Some runtimes cannot run that. vLLM's CPU attention only
has kernels for head sizes 32, 48, 64, ... (vllm/v1/attention/backends/cpu_attn.py), so a
fixture for a vLLM contract needs wider heads.

A profile declares those constraints for one integration. `violations` says why a config
cannot run there (empty: it can, e.g. an existing fixture can be reused as it is), and
`overrides` returns the smallest config_overrides that make a shrunk config fit, with a
reason, for `tmic fixture --backend`. A profile only ever adds reviewed
config_overrides, which the fixture key and build.json already record: published
fixtures and the contracts of other integrations are unchanged.

Integrations that run the fixture as it is built (transformers, transformers.js,
sentence_transformers) have an empty profile. sglang has none yet: its constraints are
not measured, and an empty profile would claim it has none.
"""

from dataclasses import dataclass

# vllm/v1/attention/backends/cpu_attn.py get_supported_head_sizes() at vLLM 0.34 (nightly 2026-11).
VLLM_CPU_HEAD_DIMS = frozenset({32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512})
# Layer kinds that dispatch through the Transformers attention interface (vLLM's
# Transformers backend requires every layer to; recurrent or linear-attention layers do not).
ATTENTION_LAYERS = frozenset({"full_attention", "sliding_attention", "chunked_attention"})


@dataclass(frozen=True)
class Profile:
    name: str
    integration: str
    # Bump when the constraints change. Not part of any key yet: profiles only add
    # config_overrides, which carry their own reason.
    version: int = 1
    head_dims: frozenset | None = None  # allowed attention head widths; None: any
    layer_types: frozenset | None = None  # allowed `layer_types` entries; None: any
    note: str = ""


PROFILES = {
    p.name: p
    for p in (
        Profile("transformers", "transformers"),
        Profile("transformers.js", "transformers.js"),
        Profile("sentence_transformers", "sentence_transformers"),
        Profile(
            "vllm-cpu",
            "vllm",
            head_dims=VLLM_CPU_HEAD_DIMS,
            layer_types=ATTENTION_LAYERS,
            note="vLLM's Transformers backend on CPU",
        ),
    )
}


def profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown serving profile {name!r}; known: {', '.join(sorted(PROFILES))}") from None


def _attention(config):
    """(dotted prefix, sub-config) holding the language model's attention fields."""
    text = config.get("text_config")
    if isinstance(text, dict) and text.get("num_attention_heads"):
        return "text_config.", text
    return "", config


def _head_dim(cfg):
    if cfg.get("head_dim"):
        return cfg["head_dim"]
    heads, hidden = cfg.get("num_attention_heads"), cfg.get("hidden_size")
    return hidden // heads if heads and hidden else None


def violations(config, prof):
    """Why `config` cannot run with this profile; empty when it can as it is."""
    prefix, cfg = _attention(config)
    problems = []
    if prof.layer_types is not None:
        other = sorted({t for t in cfg.get("layer_types") or () if t not in prof.layer_types})
        if other:
            problems.append(f"{prefix}layer_types has {other}: {prof.name} needs attention in every layer")
    if prof.head_dims is not None:
        width = _head_dim(cfg)
        if width is not None and width not in prof.head_dims:
            problems.append(f"{prefix}head width {width} is not supported by {prof.name} ({sorted(prof.head_dims)})")
    return problems


def overrides(config, prof):
    """(config_overrides fields, reason) that make `config` fit the profile, or None if it already fits.

    Widens the heads by the smallest factor k that gives a supported width, keeping
    heads x width (so q/o projection shapes): heads / k, head width x k. KV heads are
    divided by k too when they can be, which keeps the k/v projection shapes as well;
    otherwise they stay as they are (MQA keeps its single KV head) and the k/v
    projections grow. Weights sized by the head width (Qwen3's q/k norms) always grow.
    Raises ValueError for what widening cannot fix (e.g. linear-attention layers).
    """
    problems = violations(config, prof)
    if not problems:
        return None
    prefix, cfg = _attention(config)
    width = _head_dim(cfg)
    if any("layer_types" in p for p in problems) or width is None or prof.head_dims is None:
        raise ValueError("; ".join(problems))
    heads, kv = cfg["num_attention_heads"], cfg.get("num_key_value_heads")
    factors = [k for k in (2, 4, 8, 16) if heads % k == 0 and width * k in prof.head_dims]
    if not factors:
        raise ValueError(f"{prefix}head width {width} x {heads} heads: no widening reaches {prof.name}'s widths")
    keeps_shapes = [k for k in factors if not kv or kv % k == 0]
    k = (keeps_shapes or factors)[0]
    fields = {f"{prefix}num_attention_heads": heads // k}
    if kv:
        new_kv = kv // k if kv % k == 0 else kv
        if (heads // k) % new_kv:
            new_kv = 1
        fields[f"{prefix}num_key_value_heads"] = new_kv
    if cfg.get("head_dim"):
        fields[f"{prefix}head_dim"] = width * k
    shapes = (
        "keeps the q/k/v/o projection shapes"
        if keeps_shapes
        else "keeps the q/o projection shapes; the k/v projections grow"
    ) + " (weights sized by the head width, e.g. q/k norms, grow)"
    reason = (
        f"{prof.note or prof.name} supports head widths {sorted(prof.head_dims)[:4]}...; the shrink rule gives "
        f"{width}. {heads} heads of {width} -> {heads // k} of {width * k} {shapes}. "
        f"(serving profile {prof.name} v{prof.version})"
    )
    return fields, reason
