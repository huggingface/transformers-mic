"""Can the fixture recipe shrink every Transformers architecture? Configs only.

    python sweep.py prefetch            # resolve and download each doc checkpoint's config.json
    python sweep.py run [--jobs 6]      # shrink, build, forward, backward, one process per type
    python sweep.py summary

For every model type with a documentation checkpoint (`@auto_docstring(checkpoint=...)`
on its config class), load the real config, shrink it with one generic rule
(widths by field name in every sub-config, depth by layer pattern, MoE to 4
experts), build the model's primary head with random weights, and run forward
and backward on synthetic inputs built from the config. Each type stops at
the first stage that fails, with the error recorded. No weights or processors
are downloaded, so this measures the generic shrink rule, not processor paths.
"""

import argparse
import glob
import inspect
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).parent
SOURCES = HERE / "results" / "sweep_sources.json"
RESULTS = HERE / "results" / "sweep.json"

# Generic width rule: field name -> fixture value, applied in every (sub-)config.
WIDTHS = {
    "hidden_size": 64, "intermediate_size": 128, "d_model": 64, "d_ff": 128,
    "embed_dim": 64, "out_hidden_size": 64, "ffn_dim": 128, "encoder_ffn_dim": 128, "decoder_ffn_dim": 128,
    "num_attention_heads": 4, "num_heads": 4, "encoder_attention_heads": 4,
    "decoder_attention_heads": 4, "head_dim": 16, "d_kv": 16, "attention_head_dim": 16,
    # DeBERTa-v2 names its head size and pooler width explicitly.
    "attention_head_size": 16, "pooler_hidden_size": 64,
    # Per-layer embeddings (Qwen4-Exp) are vocab-sized: shrink their width too.
    "ple_embed_dim": 16,
    "moe_intermediate_size": 32, "shared_intermediate_size": 128,
    "shared_expert_intermediate_size": 128, "projection_dim": 32,
    "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "mamba_d_state": 16, "state_size": 16, "ssm_state_size": 16, "mamba_n_heads": 8,
    "mamba_d_head": "auto", "mamba_chunk_size": 8, "chunk_size": 8,
    # GPT-2, XLM, DistilBERT, and others name the same widths differently.
    "n_embd": 64, "n_inner": 128, "n_head": 4, "emb_dim": 64, "dim": 64, "hidden_dim": 128,
    "n_heads": 4,
}
EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
DEPTHS = ("num_layers", "num_decoder_layers", "encoder_layers", "decoder_layers", "n_layer", "n_layers")
# Primary head, first match wins: the usage path a contract would cover first.
HEADS = (
    "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",
    "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
    "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
    "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES",
    "MODEL_FOR_MASKED_LM_MAPPING_NAMES",
    "MODEL_FOR_CTC_MAPPING_NAMES",
    "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES",
    "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
    "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",
    "MODEL_MAPPING_NAMES",
)
# Fast integration tests in transformers#49103.
FAST_49103 = (
    "bert", "clip", "detr", "internvl", "llama", "llava", "musicgen", "paligemma", "qwen2",
    "qwen2_5_vl", "segformer", "smolvlm", "t5", "video_llama_3", "vit", "wav2vec2", "whisper",
)


def doc_checkpoints(transformers_src):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    by_class = {name: model_type for model_type, name in CONFIG_MAPPING_NAMES.items()}
    found = {}
    pattern = re.compile(
        r'@auto_docstring\(\s*checkpoint="([^"]+)"[^)]*\)\s*(?:@[^\n]+\n\s*)*class (\w+)\('
    )
    for path in glob.glob(f"{transformers_src}/models/*/configuration_*.py"):
        for match in pattern.finditer(Path(path).read_text()):
            if match.group(2) in by_class:
                found[by_class[match.group(2)]] = match.group(1)
    return {t: found.get(t) for t in CONFIG_MAPPING_NAMES}


def prefetch(args):
    import transformers
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    sources = doc_checkpoints(Path(transformers.__file__).parent)

    def types_in(config):
        """model_type of a raw config.json and of every nested config in it."""
        found = {config.get("model_type")}
        for value in config.values():
            if isinstance(value, dict) and "model_type" in value:
                found |= types_in(value)
        return found

    def try_repo(model_type, repo):
        sha = api.model_info(repo).sha
        config = json.loads(Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
        if model_type not in types_in(config):
            raise ValueError(f"config.json has model_type {config.get('model_type')}")
        return sha

    def fetch(item):
        model_type, repo = item
        row = {"doc_checkpoint": repo}
        if repo is None:
            row["doc_status"] = "no_doc_checkpoint"
        else:
            try:
                row.update(repo=repo, revision=try_repo(model_type, repo), source="doc", status="ok")
                row["doc_status"] = "ok"
                return model_type, row
            except Exception as exc:
                row["doc_status"] = (
                    "wrong_model_type" if isinstance(exc, ValueError) else type(exc).__name__
                )
        # Fall back to the most downloaded ungated Transformers model with this type.
        try:
            candidates = api.list_models(
                filter=[model_type, "transformers"], sort="downloads", limit=10, expand=["gated"]
            )
            for candidate in candidates:
                if candidate.gated:
                    continue
                try:
                    sha = try_repo(model_type, candidate.id)
                except Exception:
                    continue
                row.update(repo=candidate.id, revision=sha, source="hub", status="ok")
                return model_type, row
        except Exception as exc:
            row["error"] = str(exc)[:200]
        row["status"] = "no_checkpoint"
        return model_type, row

    with ThreadPoolExecutor(16) as pool:
        rows = dict(pool.map(fetch, sources.items()))
    SOURCES.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print({s: sum(r["status"] == s for r in rows.values()) for s in {r["status"] for r in rows.values()}})


def sub_configs(config):
    """The config and every nested config, depth first."""
    from transformers import PretrainedConfig

    yield config
    for key in getattr(config, "sub_configs", {}) or {}:
        value = getattr(config, key, None)
        if isinstance(value, PretrainedConfig):
            yield from sub_configs(value)


def layer_pattern(config):
    """Indices to keep: the first layer of each distinct kind (see layer_patterns.py)."""
    sys.path.insert(0, str(HERE))
    from layer_patterns import pattern_indices

    return pattern_indices(config)


def shrink(config):
    """Apply the generic rule in place, innermost configs first, then rebuild each."""
    from transformers import PretrainedConfig

    def rebuilt(cfg):
        values = cfg.to_dict()
        # A slim fixture holds dense random weights, whatever the checkpoint's format.
        values.pop("quantization_config", None)
        # Random fixture weights need a signal: Unlimited-OCR's SAM tower inits with
        # 1e-10, which vanishes to 1e-37 over the tower and makes images inert.
        init = values.get("initializer_range")
        if isinstance(init, float) and 0 < init < 1e-3:
            values["initializer_range"] = 0.02
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            if isinstance(sub, PretrainedConfig):
                values[key] = rebuilt(sub).to_dict()
        for key, value in WIDTHS.items():
            # 0 is a choice, not a width (GLM-5.3's NoPE qk_rope_head_dim).
            if key in values and values[key] not in (None, 0) and not isinstance(values[key], (list, dict)):
                values[key] = value
        if "num_key_value_heads" in values and values["num_key_value_heads"]:
            heads = cfg.num_attention_heads if hasattr(cfg, "num_attention_heads") else None
            values["num_key_value_heads"] = 4 if heads and values["num_key_value_heads"] >= heads else 2
        # MLA models whose rotary tables are sized by head_dim (DeepSeek-V3, Kimi
        # K2.5) set it to qk_rope_head_dim; keep them equal after shrinking.
        if cfg.to_dict().get("head_dim") and cfg.to_dict().get("head_dim") == cfg.to_dict().get("qk_rope_head_dim"):
            values["head_dim"] = values["qk_rope_head_dim"]
        # SigLIP's text tower projects to projection_size == hidden_size, which
        # must meet the vision tower's width; keep them equal after shrinking.
        if cfg.to_dict().get("projection_size") and cfg.to_dict().get("projection_size") == cfg.to_dict().get("hidden_size"):
            values["projection_size"] = values["hidden_size"]
        if values.get("q_lora_rank"):
            values["q_lora_rank"] = 16
        for key in EXPERTS:
            if isinstance(values.get(key), int) and values[key] > 4:
                values[key] = 4
        if isinstance(values.get("num_experts_per_tok"), int):
            values["num_experts_per_tok"] = min(values["num_experts_per_tok"], 2)
        for key in ("n_group", "topk_group"):
            if isinstance(values.get(key), int):
                values[key] = 1
        for key in DEPTHS:
            if isinstance(values.get(key), int):
                values[key] = min(values[key], 2)
        if isinstance(values.get("hidden_sizes"), list):
            values["hidden_sizes"] = [16 * 2**i for i in range(len(values["hidden_sizes"]))]
        if isinstance(values.get("depths"), list):
            values["depths"] = [1] * len(values["depths"])
        # Per-stage head counts must divide the shrunk per-stage widths (PVT's
        # hidden_sizes, Swin's embed_dim doubling per stage): keep the largest
        # head count, no more than the original, that does.
        # A Swin whose last stage is the model width (CLAP's audio tower:
        # patch_embeds_hidden_size doubled per stage == hidden_size) keeps that link.
        if isinstance(values.get("patch_embeds_hidden_size"), int) and isinstance(values.get("depths"), list):
            last = 2 ** (len(values["depths"]) - 1)
            if cfg.patch_embeds_hidden_size * last == cfg.hidden_size:
                values["patch_embeds_hidden_size"] = 16
                values["hidden_size"] = 16 * last
                if values.get("projection_hidden_size") == cfg.hidden_size:
                    values["projection_hidden_size"] = values["hidden_size"]
        stages = values.get("hidden_sizes")
        if not isinstance(stages, list) and isinstance(values.get("embed_dim"), int) and isinstance(values.get("depths"), list):
            stages = [values["embed_dim"] * 2**i for i in range(len(values["depths"]))]
        if not isinstance(stages, list) and isinstance(values.get("patch_embeds_hidden_size"), int) and isinstance(values.get("depths"), list):
            stages = [values["patch_embeds_hidden_size"] * 2**i for i in range(len(values["depths"]))]
        if isinstance(stages, list):
            for key in ("num_attention_heads", "num_heads"):
                heads = values.get(key)
                if isinstance(heads, list) and len(heads) == len(stages):
                    values[key] = [max(d for d in range(1, h + 1) if w % d == 0) for w, h in zip(stages, heads)]
        # Layer-indexed backbones name outputs after the real depth; let them default.
        # Stage-based ones (depths per stage) keep their stage count, so keep them.
        if not isinstance(values.get("depths"), list):
            for key in ("out_features", "out_indices"):
                if values.get(key) is not None:
                    values[key] = None
        # A detection encoder reads the backbone stages it was built for.
        backbone = values.get("backbone_config")
        if (
            isinstance(values.get("encoder_in_channels"), list)
            and isinstance(backbone, dict)
            and isinstance(backbone.get("hidden_sizes"), list)
            and backbone.get("out_indices")
        ):
            values["encoder_in_channels"] = [backbone["hidden_sizes"][i - 1] for i in backbone["out_indices"]]
        # Stage-based models (depths per stage) count stages in num_hidden_layers: keep them.
        if isinstance(values.get("num_hidden_layers"), int) and not isinstance(values.get("depths"), list):
            kept = layer_pattern(cfg) if hasattr(cfg, "num_hidden_layers") else [0, 1]
            kept = kept if len(kept) > 1 else list(range(min(2, values["num_hidden_layers"])))
            sys.path.insert(0, str(HERE))
            from layer_patterns import derived_types

            if derived_types(cfg):
                # Kinds follow the layer index, so only a prefix keeps them: up to each kind's first layer.
                kept = list(range(max(kept) + 1))
            for key, value in cfg.to_dict().items():
                if isinstance(value, list) and len(value) == values["num_hidden_layers"]:
                    values[key] = [value[i] for i in kept if i < len(value)]
            if getattr(cfg, "num_kv_shared_layers", None) is not None:
                values["num_kv_shared_layers"] = sum(i >= cfg.num_hidden_layers - cfg.num_kv_shared_layers for i in kept)
            # Leading dense layers: keep at least one MoE layer, or the fixture has no
            # experts at all (LFM2-MoE: 2 dense of 24, 2 layers kept).
            for key in ("first_k_dense_replace", "num_dense_layers"):
                if getattr(cfg, key, None) is not None:
                    values[key] = min(sum(i < getattr(cfg, key) for i in kept), len(kept) - 1)
            per_layer = values.get("per_layer_config")
            if isinstance(per_layer, dict):
                # Keys are layer indices, sometimes zero-padded (Gemma 4's "04").
                values["per_layer_config"] = {
                    str(n): value for n, i in enumerate(kept)
                    for key, value in per_layer.items() if int(key) == i
                }
            if getattr(cfg, "mlp_only_layers", None):
                values["mlp_only_layers"] = [n for n, i in enumerate(kept) if i in cfg.mlp_only_layers]
            # A layer-id list naming a layer that was dropped moves to the kept
            # layer of the same kind (Qwen4-Exp's PLE layer must stay linear attention).
            kinds = getattr(cfg, "layer_types", None)
            if isinstance(kinds, list) and len(kinds) == cfg.num_hidden_layers:
                def moved(ids):
                    same = {kinds[i]: n for n, i in reversed(list(enumerate(kept)))}
                    return sorted({kept.index(i) if i in kept else same[kinds[i]] for i in ids if kinds[i] in same})

                if isinstance(values.get("ple_layer_ids"), list):  # one-indexed
                    values["ple_layer_ids"] = [n + 1 for n in moved([i - 1 for i in values["ple_layer_ids"]])]
                linear = values.get("linear_attn_config")
                if isinstance(linear, dict):
                    # GLM-5.3 lists its linear and full attention layers by index.
                    for key in ("kda_layers", "full_attn_layers"):
                        if isinstance(linear.get(key), list):
                            linear[key] = moved(linear[key])
                    for key, value in (("num_heads", 4), ("head_dim", 16)):
                        if isinstance(linear.get(key), int):
                            linear[key] = value
            values["num_hidden_layers"] = len(kept)
        # CLIPSeg's decoder reads vision hidden states extract_layers (3, 6, 9): keep
        # one entry each, within the shrunk vision depth.
        vision = values.get("vision_config")
        if isinstance(values.get("extract_layers"), list) and isinstance(vision, dict) and vision.get("num_hidden_layers"):
            depth, top = vision["num_hidden_layers"], max(values["extract_layers"]) + 1
            values["extract_layers"] = [min(i * depth // top, depth - 1) for i in values["extract_layers"]]
        # DPT reads one hidden state per backbone_out_indices entry, one per neck
        # width: keep a layer per entry, renumbered, and shrink the neck.
        outs = values.get("backbone_out_indices")
        if isinstance(outs, list) and isinstance(values.get("num_hidden_layers"), int) and not values.get("is_hybrid"):
            values["num_hidden_layers"] = len(outs)
            values["backbone_out_indices"] = list(range(len(outs)))
            if isinstance(values.get("neck_hidden_sizes"), list):
                values["neck_hidden_sizes"] = [16 * 2 ** min(i, 2) for i in range(len(values["neck_hidden_sizes"]))]
            if isinstance(values.get("fusion_hidden_size"), int):
                values["fusion_hidden_size"] = 64
        if isinstance(values.get("intermediate_size"), list):
            values["intermediate_size"] = [min(size, 128) for size in values["intermediate_size"]]
        if values.get("upsample_groups") and values.get("hidden_size"):
            from math import gcd
            values["upsample_groups"] = gcd(values["upsample_groups"], values["hidden_size"])
        # These models need projection_dim == the text width (OWL's class head contracts
        # text query embeds against it); CLIP's projection is its own joint space.
        if values.get("model_type") in ("paligemma", "owlv2", "owlvit"):
            values["projection_dim"] = values["text_config"]["hidden_size"]
        # BLT's local encoder/decoder name the global transformer's width
        # (hidden_size_global), and the checkpoint sizes the global transformer to
        # the encoder's cross_attn_k patch outputs so no extra projection is built.
        if values.get("model_type") == "blt":
            glob = values["global_config"]
            if cfg.global_config.hidden_size == cfg.encoder_config.hidden_size * cfg.cross_attn_k:
                glob["hidden_size"] = values["encoder_config"]["hidden_size"] * values["cross_attn_k"]
                if glob.get("head_dim"):  # attention splits hidden_size; rotary reads head_dim
                    glob["head_dim"] = glob["hidden_size"] // glob["num_attention_heads"]
            glob["encoder_cross_output_size"] = None  # recomputed from the widths on load
            for key in ("encoder_config", "decoder_config"):
                values[key]["hidden_size_global"] = glob["hidden_size"]
        # A sub-config projecting into the language model (Janus's vision aligner,
        # Qwen3-ASR's audio output_dim, Llama 4's vision projector) must keep
        # projecting to its resized hidden size.
        text = getattr(cfg, "text_config", None)
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            ratio = getattr(sub, "pixel_shuffle_ratio", None)
            if (
                ratio
                and isinstance(values.get(key), dict)
                and sub.intermediate_size == sub.hidden_size / ratio**2
            ):
                # Llama 4's pixel shuffle turns hidden_size channels into
                # hidden_size / ratio**2, which its adapter MLP reads as intermediate_size.
                values[key]["intermediate_size"] = int(values[key]["hidden_size"] / ratio**2)
            for field in ("projection_dim", "output_dim", "vision_output_dim", "projector_input_dim", "projector_output_dim"):
                if (
                    key != "text_config"
                    and isinstance(values.get(key), dict)
                    and getattr(sub, field, None) is not None
                    and getattr(sub, field) == getattr(text, "hidden_size", None)
                ):
                    # BART-style text configs (Florence-2) name their width d_model.
                    text_values = values["text_config"]
                    values[key][field] = text_values.get("hidden_size") or text_values["d_model"]
        # CSM's depth decoder projects the backbone's hidden states (backbone_hidden_size).
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            if (
                isinstance(values.get(key), dict)
                and getattr(sub, "backbone_hidden_size", None) == getattr(cfg, "hidden_size", None)
                and values.get("hidden_size")
            ):
                values[key]["backbone_hidden_size"] = values["hidden_size"]
        # A sub-config's channel list that ends at a sibling's width feeds that sibling
        # (Unlimited-OCR's SAM downsample_channels [512, 1024] are the CLIP encoder's
        # patch embeddings): scale it to the sibling's resized width.
        subs = {key: getattr(cfg, key, None) for key in getattr(cfg, "sub_configs", {}) or {}}
        for key, sub in subs.items():
            channels = getattr(sub, "downsample_channels", None)
            if not (isinstance(channels, (list, tuple)) and channels and isinstance(values.get(key), dict)):
                continue
            for other, sibling in subs.items():
                width = getattr(sibling, "hidden_size", None)
                if other != key and width == channels[-1] and isinstance(values.get(other), dict):
                    resized = values[other]["hidden_size"]
                    values[key]["downsample_channels"] = [c * resized // width for c in channels]
        # Multimodal RoPE partitions must still cover half the resized head.
        rope = values.get("rope_parameters")
        if isinstance(rope, dict) and rope.get("mrope_section"):
            sections = rope["mrope_section"]
            half_head = values.get("head_dim", values["hidden_size"] // values["num_attention_heads"]) // 2
            scaled = [size * half_head / sum(sections) for size in sections]
            kept = [int(size) for size in scaled]
            for i in sorted(range(len(kept)), key=lambda i: scaled[i] - kept[i], reverse=True)[:half_head - sum(kept)]:
                kept[i] += 1
            rope["mrope_section"] = kept
        # LongRoPE keeps one factor per rotary frequency; resample to the resized head.
        if isinstance(rope, dict) and isinstance(rope.get("short_factor"), list):
            # MLA models (MiniCPM3, DeepSeek) rotate only qk_rope_head_dim channels.
            head = (
                values.get("qk_rope_head_dim")
                or values.get("head_dim")
                or values["hidden_size"] // values["num_attention_heads"]
            )
            partial = rope.get("partial_rotary_factor", values.get("partial_rotary_factor", 1.0))
            half = int(head * partial) // 2
            for key in ("short_factor", "long_factor"):
                factors = rope.get(key)
                if isinstance(factors, list) and len(factors) != half:
                    rope[key] = [factors[i * len(factors) // half] for i in range(half)]
        return type(cfg).from_dict(values)

    return rebuilt(config)


def primary_head(config):
    from transformers.models.auto import modeling_auto

    others = sorted(n for n in dir(modeling_auto) if n.endswith("_MAPPING_NAMES") and n not in HEADS)
    for name in (*HEADS, *others):
        mapping = getattr(modeling_auto, name, {})
        if not isinstance(mapping, dict):
            continue
        if config.model_type in mapping:
            classes = mapping[config.model_type]
            name_ = classes[0] if isinstance(classes, (list, tuple)) else classes
            if not name_.endswith("Config"):  # some *_MAPPING_NAMES map to configs
                return name, name_
    return None, None


def head_config(model_class, config):
    """The config the head is built from: a text-only head of a composite model
    (Qwen3.5's causal LM) takes its text config, as AutoModel would pass it."""
    expected = getattr(model_class, "config_class", None)
    if expected is None or isinstance(config, expected):
        return config
    text = config.get_text_config()
    return text if isinstance(text, expected) else config


def synthetic_inputs(model, config):
    """Inputs for main_input_name and its usual companions, sized from the config."""
    import torch

    def find(key, default=None):
        for cfg in sub_configs(config):
            value = getattr(cfg, key, None)
            if value is not None:
                return value
        return default

    name = model.main_input_name
    batch, length = 2, 16
    inputs = {}
    if name == "input_ids":
        vocab = find("vocab_size", 1000)
        inputs["input_ids"] = torch.randint(3, min(vocab, 1000), (batch, length))
        inputs["attention_mask"] = torch.ones(batch, length, dtype=torch.long)
        if config.is_encoder_decoder:
            inputs["decoder_input_ids"] = inputs["input_ids"].clone()
    elif name == "pixel_values":
        size = find("image_size", 224)
        size = size if isinstance(size, int) else (size.get("height") if isinstance(size, dict) else size[0])
        inputs["pixel_values"] = torch.randn(batch, find("num_channels", 3), size, size)
    elif name == "input_features":
        inputs["input_features"] = torch.randn(batch, find("num_mel_bins", 80), 3000)
        if config.is_encoder_decoder:
            inputs["decoder_input_ids"] = torch.randint(3, 100, (batch, 4))
    elif name == "input_values":
        inputs["input_values"] = torch.randn(batch, 16000)
    else:
        raise NotImplementedError(f"no synthetic inputs for {name}")
    # Dual encoders (CLIP-like) take text and pixels together; models that splice
    # image tokens into the text need a processor and get text only here.
    accepted = inspect.signature(model.forward).parameters
    spliced = any(find(k) is not None for k in ("image_token_id", "image_token_index"))
    vision = getattr(config, "vision_config", None)
    if name == "input_ids" and "pixel_values" in accepted and vision is not None and not spliced:
        size = getattr(vision, "image_size", 224)
        size = size if isinstance(size, int) else (size.get("height") if isinstance(size, dict) else size[0])
        inputs["pixel_values"] = torch.randn(batch, getattr(vision, "num_channels", 3), size, size)
    # Text-audio dual encoders (CLAP) take mel features beside the text, as the
    # feature extractor shapes them: (batch, channels, frames, mel bins).
    audio = getattr(config, "audio_config", None)
    if name == "input_ids" and "input_features" in accepted and audio is not None and getattr(audio, "spec_size", None):
        inputs["input_features"] = torch.randn(batch, 1, 1001, audio.num_mel_bins)
        if "is_longer" in accepted:  # fusion models flag clips longer than the window
            inputs["is_longer"] = torch.zeros(batch, 1, dtype=torch.bool)
    return inputs


def run_one(model_type):
    import torch
    import transformers
    from transformers import AutoConfig

    transformers.logging.set_verbosity_error()
    torch.manual_seed(0)
    source = json.loads(SOURCES.read_text())[model_type]
    row = {"model_type": model_type, "repo": source.get("repo"), "revision": source.get("revision")}
    stage = "load_config"
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(source["repo"], "config.json", revision=source["revision"])
        config = AutoConfig.from_pretrained(str(Path(path).parent))
        row["config_model_type"] = config.model_type
        if config.model_type != model_type:
            # The doc checkpoint is a composite model; use its nested config of this type.
            nested = [c for c in sub_configs(config) if c.model_type == model_type]
            if not nested:
                row["stage"] = "sub_config"
                return row
            config = nested[0]
            row["from_parent"] = True
        stage = "shrink"
        small = shrink(config)
        stage = "build"
        head, class_name = primary_head(small)
        row["head"] = head
        if class_name is None:
            # Only reachable inside a composite model, not a usage path of its own.
            row["stage"] = "component"
            return row
        small = head_config(getattr(transformers, class_name), small)
        model = getattr(transformers, class_name)._from_config(small, dtype=torch.float32)
        row["parameters"] = sum(p.numel() for p in model.parameters())
        row["layers"] = getattr(small, "num_hidden_layers", None)
        stage = "inputs"
        inputs = synthetic_inputs(model, small)
        row["main_input"] = model.main_input_name
        row["inputs"] = sorted(inputs)
        # Multimodal models get text only here; their image/audio path is not exercised.
        row["text_only_multimodal"] = "input_ids" in inputs and any(
            getattr(small, k, None) is not None for k in ("vision_config", "audio_config")
        )
        stage = "forward"
        start = time.perf_counter()
        output = model(**inputs)
        row["forward_seconds"] = time.perf_counter() - start
        tensors = [v for v in output.values() if isinstance(v, torch.Tensor) and v.is_floating_point()]
        row["finite"] = all(bool(torch.isfinite(t).all()) for t in tensors)
        stage = "backward"
        target = output.get("logits")
        target = target if target is not None else tensors[0]
        target.float().pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        row["finite"] = row["finite"] and bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        row["stage"] = "ok"
    except Exception as exc:
        row["stage"] = stage
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return row


def run(args):
    sources = json.loads(SOURCES.read_text())
    types = args.types or sorted(t for t, s in sources.items() if s["status"] == "ok")

    def one(model_type):
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "one", model_type],
                capture_output=True, text=True, timeout=args.timeout,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None
            if proc.returncode == 0 and line:
                return json.loads(line)
            return {"model_type": model_type, "stage": "crash", "error": proc.stderr[-400:]}
        except subprocess.TimeoutExpired:
            return {"model_type": model_type, "stage": "timeout"}

    # A run over named types updates those rows and keeps the rest.
    rows = json.loads(RESULTS.read_text())["rows"] if args.types and RESULTS.exists() else {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for row in pool.map(one, types):
            rows[row["model_type"]] = row
            print(f"{row['model_type']:40} {row['stage']:12} {row.get('error', '')[:100]}", flush=True)
    for model_type, source in sources.items():
        if model_type not in rows and not args.types:
            rows[model_type] = {"model_type": model_type, "stage": source["status"]}
    report = {"transformers": __import__("transformers").__version__, "rows": rows}
    RESULTS.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    summary(args)


def summary(args):
    rows = json.loads(RESULTS.read_text())["rows"]
    counts = {}
    for row in rows.values():
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    print(json.dumps(dict(sorted(counts.items(), key=lambda kv: -kv[1])), indent=1))
    print("49103:", {t: rows.get(t, {}).get("stage") for t in FAST_49103})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prefetch", "run", "one", "summary"))
    parser.add_argument("types", nargs="*")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.command == "one":
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch

        torch.set_num_threads(1)
        print(json.dumps(run_one(args.types[0]), default=str))
    else:
        {"prefetch": prefetch, "run": run, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
